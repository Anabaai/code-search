use crate::embeddings::EmbeddingModel;
use crate::scanner::{scan_repository, process_file, FileEntry, FileChunk};
use crate::store::VectorStore;
use crate::text_index::TextIndex;
use anyhow::Result;
use std::path::Path;
use std::collections::{HashSet, HashMap};
use rayon::prelude::*;

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

        // 1. Scan Repository (Metadata only)
        eprintln!("Scanning repository: {}", repo_path);
        
        let (tx, rx) = crossbeam_channel::unbounded();
        let repo_path_owned = repo_path.to_string();
        let exclude_owned = exclude.clone();
        
        let repo_path_for_scan = repo_path_owned.clone();
        
        std::thread::spawn(move || {
            scan_repository(&repo_path_for_scan, tx, exclude_owned);
        });
        
        // Collect all file entries
        let current_entries: Vec<FileEntry> = rx.iter().collect();
        eprintln!("Found {} files in repository.", current_entries.len());

        // 2. Fetch Existing Index Metadata
        let indexed_metadata = store.get_indexed_metadata().await?;
        eprintln!("Found {} files in existing index.", indexed_metadata.len());

        // 3. Compute Diffs
        let mut files_to_reindex = Vec::new();
        let mut seen_files_in_scan = HashSet::new();

        // Check for modifications/additions
        for entry in &current_entries {
            seen_files_in_scan.insert(entry.path.clone());
            
            if let Some(&indexed_mtime) = indexed_metadata.get(&entry.path) {
                // If mtime changed (newer OR older), re-index.
                if entry.mtime != indexed_mtime {
                    files_to_reindex.push(entry);
                }
            } else {
                // New file
                files_to_reindex.push(entry);
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
        if !files_to_reindex.is_empty() {
            eprintln!("Re-indexing {} files...", files_to_reindex.len());
            
            // Parallel processing of files to generate chunks
            let chunks_to_upsert: Vec<FileChunk> = files_to_reindex.par_iter()
                .filter_map(|entry| {
                     let full_path = Path::new(&repo_path_owned).join(&entry.path); // Use repo_path_owned
                     process_file(&full_path, &repo_path_owned, max_lines).ok()
                })
                .flatten()
                .collect();

            if !chunks_to_upsert.is_empty() {
                eprintln!("Generated {} chunks from {} files.", chunks_to_upsert.len(), files_to_reindex.len());
                
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
                 
                 store.upsert(&chunks_to_upsert, &all_embeddings).await?;
                 
                 // Update Text Index
                 let tantivy_path = path.join(".code-search/text_index");
                 let text_index = TextIndex::load_or_create(tantivy_path.to_str().unwrap())?;
                 
                 for chunk in &chunks_to_upsert {
                     let _ = text_index.index_text(&chunk.file_path, &chunk.content);
                 }
                 text_index.save("")?; // Path ignored
            }
        } else {
            eprintln!("Index is up to date. Skipping embedding.");
        }
        
        // Cleanup old versions (optimization)
        let _ = store.cleanup().await;

        // 6. Search (Hybrid: Recall + Rerank)
        // Load Text Index
        let tantivy_path = path.join(".code-search/text_index");
        let text_index = TextIndex::load_or_create(tantivy_path.to_str().unwrap())?;
        
        // Vector Search
        let fetch_limit = std::cmp::max(limit * 3, 50);
        let query_embedding = self.model.embed_batch(&[query.to_string()])?;
        let vector_results = store.search(&query_embedding[0], fetch_limit).await?;
        
        // Text Search
        let text_results = text_index.search(query);
        
        // RRF Fusion
        // Map: FilePath -> (VectorRank, TextRank)
        let mut rankings: HashMap<String, (Option<usize>, Option<usize>)> = HashMap::new();
        
        // Vector Ranks (0-indexed)
        for (rank, res) in vector_results.iter().enumerate() {
            rankings.entry(res.file_path.clone())
                .and_modify(|e| e.0 = Some(rank))
                .or_insert((Some(rank), None));
        }
        
        // Text Ranks
        for (rank, (path, _score)) in text_results.iter().enumerate() {
             rankings.entry(path.clone())
                .and_modify(|e| e.1 = Some(rank))
                .or_insert((None, Some(rank)));
        }
        
        let k = 60.0;
        let mut fused_scores: Vec<(String, f32)> = rankings.iter().map(|(path, (r_vec, r_text))| {
            let score_vec = if let Some(r) = r_vec { 1.0 / (k + *r as f32) } else { 0.0 };
            let score_text = if let Some(r) = r_text { 1.0 / (k + *r as f32) } else { 0.0 };
            (path.clone(), score_vec + score_text)
        }).collect();
        
        // Sort by RRF score
        fused_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Select top candidates
        // let top_paths: HashSet<String> = fused_scores.iter().take(limit * 2).map(|(p, _): &(String, f32)| p.clone()).collect();
        
        // Filter candidates to return full objects
        // We only have full content for Vector Results currently (loaded from DB).
        // Text index doesn't store content (optimization).
        // So we prioritized vector results, but if a text result is NOT in vector results, we might miss it.
        // However, `vector_results` has content. `text_results` acts as a booster/filter.
        // If a file is ONLY in text results, we can't show it unless we read file (expensive).
        // Compromise: We only re-rank the `vector_results` + highly ranked text results if possible?
        // Actually, let's just use RRF to re-order `vector_results`.
        // If a Top Text Result is missing from Vector Results, we might want to fetch it?
        // For now, let's just RRF re-rank the `vector_results` combined with text signal.
        // Wait, if it's not in vector_results (fetch_limit), we don't have the chunk content.
        // We can fetch from store by ID? LanceDB supports it.
        // But our `store` API is limited.
        // Let's stick to: RRF re-ranking of the retrieved candidates from Vector Store.
        // We used `fetch_limit` (limit * 3).
        
        let mut candidates = vector_results;
        
        for candidate in &mut candidates {
            // Check text rank
            if let Some((_, Some(text_rank))) = rankings.get(&candidate.file_path) {
                // Boost score based on text rank
                // Simple additive boost? Or replace score with RRF?
                // Let's add RRF component to the existing score?
                // Existing score: 0.0-1.0.
                // RRF score: ~0.03 max.
                // Let's scale RRF.
                 let rrf_boost = 1.0 / (k + *text_rank as f32);
                 candidate.score += rrf_boost * 10.0; // Significant boost
            }
        }
        
        // Rerank: Apply keyword boost (existing logic)
        let query_lower = query.to_lowercase();
        
        for candidate in &mut candidates {
            if candidate.content.to_lowercase().contains(&query_lower) {
                candidate.score += 0.1;
            }
        }
        
        // Filter low scores
        candidates.retain(|c| c.score > 0.01);

        // Sort by new score (descending)
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Diversity: Limit chunks per file (Max 3)
        let mut file_counts = std::collections::HashMap::new();
        let mut diverse_candidates = Vec::new();
        
        for candidate in candidates {
            let count = file_counts.entry(candidate.file_path.clone()).or_insert(0);
            if *count < 3 {
                diverse_candidates.push(candidate);
                *count += 1;
            }
            if diverse_candidates.len() >= limit {
                break;
            }
        }
        
        Ok(diverse_candidates)
    }

    pub async fn index_file(&self, path: &Path, root: &str, max_lines: usize) -> Result<()> {
         eprintln!("Indexing updated file: {:?}", path);
         let db_path = Path::new(root).join(".code-search");
         let db_path_str = db_path.to_str()
             .ok_or_else(|| anyhow::anyhow!("Invalid unicode path: {:?}", db_path))?;
         
         // Note: Opening store for every file event is not ideal for high throughput,
         // but fine for interactive editing (watch mode).
         let store = VectorStore::new(db_path_str).await?;

         let relative_path = pathdiff::diff_paths(path, root)
            .unwrap_or(path.to_path_buf())
            .to_string_lossy()
            .to_string();

         if !path.exists() {
             eprintln!("File deleted: {}", relative_path);
             store.delete_files(&[relative_path]).await?;
             return Ok(());
         }
         
         // Only process if it is a supported code file
         if !crate::scanner::should_process_file(path) {
             return Ok(());
         }

         // Process file
         match process_file(path, root, max_lines) {
             Ok(chunks) => {
                 if chunks.is_empty() {
                     // Empty file or no code
                     // Should we delete it if it existed? Yes.
                     store.delete_files(&[relative_path]).await?;
                     return Ok(());
                 }
                 
                 let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
                 let embeddings = self.model.embed_batch(&texts)?;
                 
                 // Reuse upsert which handles deleting old chunks for this file
                 store.upsert(&chunks, &embeddings).await?;
                 
                 // Update Text Index
                 let tantivy_path = Path::new(root).join(".code-search/text_index");
                 {
                    // Accessing text_index via Searcher might be cleaner if we cached it.
                    // But here we load/save to ensure persistence.
                    // TODO: Optimize by keeping in memory and saving periodically?
                    let text_index = TextIndex::load_or_create(tantivy_path.to_str().unwrap())?;
                    let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
                    for text in texts {
                         let _ = text_index.index_text(&relative_path, &text);
                    }
                    text_index.save("")?;
                 }
                 
                 eprintln!("Updated index for: {} ({} chunks)", relative_path, chunks.len());
             },
             Err(e) => {
                 eprintln!("Failed to process file {:?}: {}", path, e);
             }
         }
         
         Ok(())
    }
}
