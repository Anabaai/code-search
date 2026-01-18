use anyhow::Result;
use arrow_array::{
    FixedSizeListArray, Float32Array, Int32Array, Int64Array, RecordBatch, RecordBatchIterator,
    StringArray,
    types::Float32Type,
    Array,
};
use arrow_schema::{DataType, Field, Schema};
use futures::StreamExt;
use lancedb::{connect, Connection};
use lancedb::query::{ExecutableQuery, QueryBase, Select}; // Import Select
use lancedb::arrow::SendableRecordBatchStream; 
use std::sync::Arc;
use std::collections::HashMap;
use crate::scanner::FileChunk;

const EMBEDDING_DIM: i32 = 384;

pub struct VectorStore {
    conn: Connection,
    table_name: String,
}

impl VectorStore {
    pub async fn new(path: &str) -> Result<Self> {
        let conn = connect(path).execute().await?;
        Ok(Self {
            conn,
            table_name: "code_chunks".to_string(),
        })
    }

    pub async fn get_indexed_metadata(&self) -> Result<HashMap<String, u64>> {
        let mut map = HashMap::new();
        
        let table = match self.conn.open_table(&self.table_name).execute().await {
            Ok(t) => t,
            Err(_) => return Ok(map),
        };

        // select needs Select enum
        let selection = Select::Columns(vec!["file_path".to_string(), "mtime".to_string()]);
        let stream_result = table.query().select(selection).limit(1_000_000).execute().await;
        
        let mut stream: SendableRecordBatchStream = match stream_result {
            Ok(s) => s,
            Err(_) => return Ok(map), // Schema mismatch or other error
        };

        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            
            let file_path_col: &Arc<dyn Array> = batch.column_by_name("file_path")
                .ok_or(anyhow::anyhow!("Missing file_path"))?;
            let file_paths: &StringArray = file_path_col.as_any().downcast_ref::<StringArray>()
                .ok_or(anyhow::anyhow!("Invalid file_path type"))?;
            
            let mtime_col: &Arc<dyn Array> = batch.column_by_name("mtime")
                .ok_or(anyhow::anyhow!("Missing mtime"))?;
            let mtimes: &Int64Array = mtime_col.as_any().downcast_ref::<Int64Array>()
                .ok_or(anyhow::anyhow!("Invalid mtime type"))?;

            for i in 0..batch.num_rows() {
                let path = file_paths.value(i).to_string();
                let mtime = mtimes.value(i) as u64;
                map.insert(path, mtime);
            }
        }
        Ok(map)
    }

    pub async fn upsert(&self, chunks: &[FileChunk], embeddings: &[Vec<f32>]) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }
        eprintln!("Upserting {} chunks into LanceDB...", chunks.len());

        let schema = Arc::new(Schema::new(vec![
            Field::new("file_path", DataType::Utf8, false),
            Field::new("chunk_index", DataType::Int32, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("line_start", DataType::Int32, false),
            Field::new("line_end", DataType::Int32, false),
            Field::new("mtime", DataType::Int64, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    EMBEDDING_DIM,
                ),
                false,
            ),
        ]));

        let file_paths = StringArray::from(
            chunks.iter().map(|c| c.file_path.clone()).collect::<Vec<_>>()
        );
        let chunk_indices = Int32Array::from(
            chunks.iter().map(|c| c.chunk_index as i32).collect::<Vec<_>>()
        );
        let contents = StringArray::from(
            chunks.iter().map(|c| c.content.clone()).collect::<Vec<_>>()
        );
        let line_starts = Int32Array::from(
            chunks.iter().map(|c| c.line_start as i32).collect::<Vec<_>>()
        );
        let line_ends = Int32Array::from(
            chunks.iter().map(|c| c.line_end as i32).collect::<Vec<_>>()
        );
        let mtimes = Int64Array::from(
            chunks.iter().map(|c| c.mtime as i64).collect::<Vec<_>>()
        );

        let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            embeddings.iter().map(|e| Some(e.iter().map(|x| Some(*x)))),
            EMBEDDING_DIM,
        );

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(file_paths),
                Arc::new(chunk_indices),
                Arc::new(contents),
                Arc::new(line_starts),
                Arc::new(line_ends),
                Arc::new(mtimes),
                Arc::new(vectors),
            ],
        )?;

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        
        match self.conn.open_table(&self.table_name).execute().await {
            Ok(table) => {
                 let unique_files: Vec<String> = chunks.iter()
                    .map(|c| c.file_path.clone())
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();
                 
                 if !unique_files.is_empty() {
                     let filter = unique_files.iter()
                        .map(|f| format!("'{}'", f))
                        .collect::<Vec<_>>()
                        .join(", ");
                     let predicate = format!("file_path IN ({})", filter);
                     let _ = table.delete(&predicate).await;
                 }
                 table.add(batches).execute().await?;
            },
            Err(_) => {
                self.conn.create_table(&self.table_name, batches).execute().await?;
            }
        }
        Ok(())
    }

    pub async fn delete_files(&self, file_paths: &[String]) -> Result<()> {
        if file_paths.is_empty() { return Ok(()); }
        
        let table = match self.conn.open_table(&self.table_name).execute().await {
            Ok(t) => t,
            Err(_) => return Ok(()),
        };
        
        let filter = file_paths.iter()
            .map(|f| format!("'{}'", f))
            .collect::<Vec<_>>()
            .join(", ");
        let predicate = format!("file_path IN ({})", filter);
        table.delete(&predicate).await?;
        Ok(())
    }

    pub async fn search(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
        let table = match self.conn.open_table(&self.table_name).execute().await {
            Ok(t) => t,
            Err(_) => return Ok(vec![]),
        };
        
        let mut results: SendableRecordBatchStream = table
            .vector_search(query_embedding.to_vec())?
            .limit(limit)
            .execute()
            .await?;

        let mut search_results = Vec::new();

        while let Some(batch_result) = results.next().await {
            let batch = batch_result?;

            let file_path_col: &Arc<dyn Array> = batch.column_by_name("file_path")
                .ok_or(anyhow::anyhow!("Missing file_path"))?;
            let file_paths: &StringArray = file_path_col.as_any().downcast_ref::<StringArray>()
                .ok_or(anyhow::anyhow!("Invalid file_path"))?;
            
            let chunk_index_col: &Arc<dyn Array> = batch.column_by_name("chunk_index")
                .ok_or(anyhow::anyhow!("Missing chunk_index"))?;
            let chunk_indices: &Int32Array = chunk_index_col.as_any().downcast_ref::<Int32Array>()
                .ok_or(anyhow::anyhow!("Invalid chunk_index"))?;
            
            let content_col: &Arc<dyn Array> = batch.column_by_name("content")
                .ok_or(anyhow::anyhow!("Missing content"))?;
            let contents: &StringArray = content_col.as_any().downcast_ref::<StringArray>()
                .ok_or(anyhow::anyhow!("Invalid content"))?;
            
            let line_start_col: &Arc<dyn Array> = batch.column_by_name("line_start")
                .ok_or(anyhow::anyhow!("Missing line_start"))?;
            let line_starts: &Int32Array = line_start_col.as_any().downcast_ref::<Int32Array>()
                .ok_or(anyhow::anyhow!("Invalid line_start"))?;
            
            let line_end_col: &Arc<dyn Array> = batch.column_by_name("line_end")
                .ok_or(anyhow::anyhow!("Missing line_end"))?;
            let line_ends: &Int32Array = line_end_col.as_any().downcast_ref::<Int32Array>()
                .ok_or(anyhow::anyhow!("Invalid line_end"))?;

            let dist_col = batch.column_by_name("_distance");
            // Handle optional distance column
            let distances: Option<&Float32Array> = if let Some(col) = dist_col {
                col.as_any().downcast_ref::<Float32Array>()
            } else {
                None
            };
            
            for i in 0..batch.num_rows() {
                let dist = if let Some(d_arr) = distances {
                     d_arr.value(i)
                } else {
                    0.0
                };
                // Assuming L2 distance on normalized vectors (range 0.0 to 2.0)
                // Map to 0.0 - 1.0 similarity score
                let score = (1.0 - (dist / 2.0)).max(0.0);

                search_results.push(SearchResult {
                    file_path: file_paths.value(i).to_string(),
                    chunk_index: chunk_indices.value(i) as usize,
                    content: contents.value(i).to_string(),
                    line_start: line_starts.value(i) as usize,
                    line_end: line_ends.value(i) as usize,
                    score, 
                });
            }
        }

        Ok(search_results)
    }

    pub async fn cleanup(&self) -> Result<()> {
         // Cleanup old versions to prevent disk bloat.
         // Lancedb 0.14 uses `optimize` with `OptimizeAction::Prune`.
         // We keep versions from the last 1 hour.
         let table = match self.conn.open_table(&self.table_name).execute().await {
            Ok(t) => t,
            Err(_) => return Ok(()),
        };

        use lancedb::table::OptimizeAction;
        
        // 1. Prune old versions
        match table.optimize(OptimizeAction::Prune { 
            older_than: Some(chrono::Duration::hours(1)), 
            delete_unverified: Some(false),
            error_if_tagged_old_versions: Some(false)
        }).await {
             Ok(_) => {
                 eprintln!("Storage cleanup (Prune) completed.");
             }
             Err(e) => {
                 eprintln!("Storage cleanup warning: {}", e);
             }
        }
        
        // 2. Compact files (merge small fragments)
        match table.optimize(OptimizeAction::Compact { 
            options: lancedb::table::CompactionOptions::default(), 
            remap_options: None 
        }).await {
            Ok(_) => {
                eprintln!("Storage compaction completed.");
            }
            Err(e) => {
                eprintln!("Storage compaction warning: {}", e);
            }
        }

        Ok(())
    }
}

pub struct SearchResult {
    pub file_path: String,
    #[allow(dead_code)]
    pub chunk_index: usize,
    pub content: String,
    pub line_start: usize,
    pub line_end: usize,
    pub score: f32,
}
