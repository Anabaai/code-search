use anyhow::Result;
use std::path::Path;
use std::sync::{Arc, RwLock};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Schema, TEXT, STORED, STRING, Field, Value};
use tantivy::{doc, Index, IndexWriter, Term, TantivyDocument};
use tantivy::directory::MmapDirectory;

pub struct TextIndex {
    index: Index,
    writer: Arc<RwLock<IndexWriter>>,
    path_field: Field,
    content_field: Field,
}

impl TextIndex {
    pub fn load_or_create(path_str: &str) -> Result<Self> {
        let index_path = Path::new(path_str);
        if !index_path.exists() {
             std::fs::create_dir_all(index_path)?;
        }

        let mut schema_builder = Schema::builder();
        // Use STRING for path (exact match, untokenized)
        let path_field = schema_builder.add_text_field("path", STRING | STORED);
        let content_field = schema_builder.add_text_field("content", TEXT);
        let schema = schema_builder.build();

        let dir = MmapDirectory::open(index_path)?;
        let index = Index::open_or_create(dir, schema.clone())?;
        
        // 50MB buffer
        let writer = index.writer(50_000_000)?;

        Ok(Self {
            index,
            writer: Arc::new(RwLock::new(writer)),
            path_field,
            content_field,
        })
    }

    pub fn save(&self, _path: &str) -> Result<()> {
        // Commit changes. Path arg is ignored as Tantivy manages its own dir.
        let mut writer = self.writer.write().unwrap();
        writer.commit()?;
        Ok(())
    }

    pub fn index_text(&self, file_path: &str, content: &str) -> Result<()> {
        let writer = self.writer.write().unwrap();
        
        // Delete existing document for this path to support updates
        let term = Term::from_field_text(self.path_field, file_path);
        writer.delete_term(term);
        
        writer.add_document(tantivy::doc!(
            self.path_field => file_path,
            self.content_field => content,
        ))?;
        
        Ok(())
    }
    
    pub fn search(&self, query_str: &str) -> Vec<(String, f32)> {
        let reader = match self.index.reader_builder()
            .try_into() {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Failed to get reader: {}", e);
                    return vec![];
                }
            };
            
        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.content_field]);
        
        let query = match query_parser.parse_query(query_str) {
            Ok(q) => q,
            Err(_) => return vec![], // Invalid query
        };
        
        // Get top 50 results
        let top_docs = match searcher.search(&query, &TopDocs::with_limit(50)) {
            Ok(docs) => docs,
            Err(_) => return vec![],
        };
        
        let mut results = Vec::new();
        for (score, doc_address) in top_docs {
            let retrieved_doc: TantivyDocument = match searcher.doc(doc_address) {
                Ok(doc) => doc,
                Err(_) => continue,
            };
            
            let path_val_opt = retrieved_doc.get_first(self.path_field);
            if let Some(path_val) = path_val_opt {
                let path_opt: Option<&str> = path_val.as_str();
                if let Some(path) = path_opt {
                     results.push((path.to_string(), score));
                }
            }
        }
        
        results
    }
}
