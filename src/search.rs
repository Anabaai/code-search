use crate::embeddings::EmbeddingModel;
use crate::scanner::scan_repository;
use crate::store::VectorStore;
use anyhow::Result;
use std::path::Path;
use std::collections::HashSet;

pub struct Searcher {
    model: EmbeddingModel,
}

impl Searcher {
    pub fn new() -> Result<Self> {
        Ok(Self {
            model: EmbeddingModel::new()?,
        })
    }

    pub async fn search(&self, repo_path: &str, query: &str, max_lines: usize, exclude: Vec<String>, limit: usize) -> Result<Vec<crate::store::SearchResult>> {
        let path = Path::new(repo_path);
        if !path.exists() {
            return Err(anyhow::anyhow!("Repository path not found: {}", repo_path));
        }

        let db_path = path.join(".code-search");
        let db_path_str = db_path.to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid unicode path: {:?}", db_path))?;
        let store = VectorStore::new(db_path_str).await?;

        // 1. Scan Repository
        eprintln!("Scanning repository: {}", repo_path);
        
        let (tx, rx) = crossbeam_channel::unbounded();
        let repo_path_owned = repo_path.to_string();
        let exclude_owned = exclude.clone();
        
        std::thread::spawn(move || {
            scan_repository(&repo_path_owned, tx, max_lines, exclude_owned);
        });
        
        // Collect all chunks
        let current_chunks: Vec<_> = rx.iter().collect();
        eprintln!("Found {} chunks in repository.", current_chunks.len());

        // 2. Fetch Existing Index Metadata
        let indexed_metadata = store.get_indexed_metadata().await?;
        eprintln!("Found {} files in existing index.", indexed_metadata.len());

        // 3. Compute Diffs
        // Identify files to add/update
        // A chunk is part of a file. We track file-level mtimes.
        // If ANY chunk of a file changes, we re-index the whole file (since chunks might shift).
        // `current_chunks` is flat list of chunks. Group by file?
        // Actually, `scan_repository` returns fresh chunks.
        // We just need to check if the file for a chunk needs update.
        
        let mut files_to_reindex = HashSet::new();
        let mut seen_files_in_scan = HashSet::new();

        // Check for modifications/additions
        // Optimization: scan_repository returns all chunks. 
        // We can just iterate chunks.
        for chunk in &current_chunks {
            seen_files_in_scan.insert(chunk.file_path.clone());
            
            if let Some(&indexed_mtime) = indexed_metadata.get(&chunk.file_path) {
                // If mtime changed (newer OR older), re-index.
                if chunk.mtime != indexed_mtime {
                    files_to_reindex.insert(chunk.file_path.clone());
                }
            } else {
                // New file
                files_to_reindex.insert(chunk.file_path.clone());
            }
        }
        
        // Identify removed files
        let mut files_to_remove = Vec::new();
        for indexed_path in indexed_metadata.keys() {
            if !seen_files_in_scan.contains(indexed_path) {
                files_to_remove.push(indexed_path.clone());
            }
        }

        // 4. Handle Deletions
        if !files_to_remove.is_empty() {
             eprintln!("Removing {} deleted files from index...", files_to_remove.len());
             store.delete_files(&files_to_remove).await?;
        }

        // 5. Handle Upserts (Re-indexing)
        // Filter chunks to only those belonging to files_to_reindex
        let chunks_to_upsert: Vec<_> = current_chunks.into_iter()
            .filter(|c| files_to_reindex.contains(&c.file_path))
            .collect();
        
        if !chunks_to_upsert.is_empty() {
            eprintln!("Re-indexing {} chunks from {} files...", chunks_to_upsert.len(), files_to_reindex.len());
            
            let texts: Vec<String> = chunks_to_upsert.iter().map(|c| c.content.clone()).collect();
             
             // Batch embedding
             let mut all_embeddings = Vec::new();
             let total_chunks = texts.len();
             let mut processed = 0;
             eprintln!("Generating embeddings for {} chunks...", total_chunks);
             
             for chunk_batch in texts.chunks(32) {
                 let embeddings = self.model.embed_batch(chunk_batch)?;
                 all_embeddings.extend(embeddings);
                 processed += chunk_batch.len();
                 if processed % 320 == 0 || processed == total_chunks {
                    eprintln!("Processed {}/{} chunks...", processed, total_chunks);
                 }
             }
             
             // Note: upsert in store.rs now handles deletion of old versions of these files if needed
             // But we are passing new chunks. `upsert` logic should handle overwrite/append correctly.
             // Our updated `upsert` does: remove existing rows for these files -> append new rows.
             store.upsert(&chunks_to_upsert, &all_embeddings).await?;
        } else {
            eprintln!("Index is up to date. Skipping embedding.");
        }
        
        // Cleanup old versions (optimization)
        let _ = store.cleanup().await;

        // 6. Search (Hybrid: Recall + Rerank)
        // Recall: effective_limit = max(limit * 3, 50) to ensure we have enough candidates
        let fetch_limit = std::cmp::max(limit * 3, 50);
        let query_embedding = self.model.embed_batch(&[query.to_string()])?;
        
        // Fetch candidates (vector search)
        let mut candidates = store.search(&query_embedding[0], fetch_limit).await?;
        
        // Rerank: Apply keyword boost
        // We do a simple case-insensitive check. If query is in content, boost score.
        // Boost magnitude: +0.25 (arbitrary, sufficient to push "ok" semantic match above "good" semantic match)
        let query_lower = query.to_lowercase();
        
        for candidate in &mut candidates {
            if candidate.content.to_lowercase().contains(&query_lower) {
                // Determine boost. Exact match is very strong signal for "code search".
                // But we must capture it.
                // Logically: 
                // - Semantic score ranges usually 0.5 ~ 0.8 for unrelated-but-similar, 0.8~1.0 for good.
                // - A boost of 0.5 essentially guarantees it floats to top unless baseline was terrible.
                candidate.score += 0.5;
            }
        }
        
        // Sort by new score (descending)
        // search returns already sorted by vector score, but we modified scores.
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Truncate to original limit
        if candidates.len() > limit {
            candidates.truncate(limit);
        }

        Ok(candidates)
    }
}
