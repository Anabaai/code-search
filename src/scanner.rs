use anyhow::Result;
use ignore::WalkBuilder;
use std::fs;
use std::path::Path;
use std::time::SystemTime;
use tree_sitter::{Parser, Query, QueryCursor};

use crossbeam_channel::Sender;

#[derive(Debug, Clone)]
pub struct FileChunk {
    pub file_path: String,
    pub chunk_index: usize,
    pub content: String,
    pub line_start: usize,
    pub line_end: usize,
    pub mtime: u64,
}

#[derive(Debug, Clone)]
pub struct FileEntry {
    pub path: String,
    pub mtime: u64,
}

pub fn scan_repository(root_path: &str, tx: Sender<FileEntry>, exclude: Vec<String>) {
    let mut builder = WalkBuilder::new(root_path);
    builder
        .hidden(false)
        .git_ignore(true)
        .add_custom_ignore_filename(".codesearchignore");
    
    if !exclude.is_empty() {
        let mut overrides = ignore::overrides::OverrideBuilder::new(root_path);
        for pattern in exclude {
            // "!" prefix means ignore in OverrideBuilder
            let p = if pattern.starts_with("!") { pattern } else { format!("!{}", pattern) };
            let _ = overrides.add(&p);
        }
        if let Ok(ov) = overrides.build() {
            builder.overrides(ov);
        }
    }

    // Ensure .code-search/ is in .gitignore
    ensure_gitignore(root_path);

    let root_path_owned = root_path.to_string();

    builder.build_parallel().run(|| {
        let tx = tx.clone();
        let root = root_path_owned.clone();
        Box::new(move |result| {
            if let Ok(entry) = result {
                let path = entry.path();
                // Explicitly filter common noise directories
                if path.components().any(|c| c.as_os_str() == "target" || c.as_os_str() == ".git" || c.as_os_str() == "node_modules") {
                    return ignore::WalkState::Continue;
                }

                if entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                    if should_process_file(path) {
                        // Calculate mtime for change detection
                        let mtime = match fs::metadata(path) {
                            Ok(metadata) => metadata.modified()
                                .unwrap_or(SystemTime::UNIX_EPOCH)
                                .duration_since(SystemTime::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs(),
                            Err(_) => 0,
                        };

                        let relative_path = pathdiff::diff_paths(path, &root)
                            .unwrap_or(path.to_path_buf())
                            .to_string_lossy()
                            .to_string();

                        let _ = tx.send(FileEntry {
                            path: relative_path,
                            mtime,
                        });
                    }
                }
            }
            ignore::WalkState::Continue
        })
    });
}

const VALID_EXTENSIONS: &[&str] = &[
    "rs", "py", "js", "ts", "jsx", "tsx", "go", "java", "cpp", "c", "h", "hpp", "php", "rb", "cs", 
    "md", "txt", "json", "yml", "yaml", "toml"
];

pub fn should_process_file(path: &Path) -> bool {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    VALID_EXTENSIONS.contains(&ext)
}

pub fn process_file(path: &Path, root_path: &str, max_lines: usize) -> Result<Vec<FileChunk>> {
    let content = fs::read_to_string(path)?;
    let metadata = fs::metadata(path)?;
    let mtime = metadata.modified()
        .unwrap_or(SystemTime::UNIX_EPOCH)
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs();

    let relative_path = pathdiff::diff_paths(path, root_path)
        .unwrap_or(path.to_path_buf())
        .to_string_lossy()
        .to_string();

    // Try AST chunking first
    if let Some(chunks) = chunk_with_tree_sitter(path, &content, &relative_path, mtime, max_lines) {
        return Ok(chunks);
    }

    // Fallback to heuristic
    Ok(chunk_with_heuristic(&content, &relative_path, mtime, max_lines))
}

fn chunk_with_tree_sitter(path: &Path, content: &str, relative_path: &str, mtime: u64, max_lines: usize) -> Option<Vec<FileChunk>> {
    let ext = path.extension()?.to_str()?;
    
    let (language, query_str) = match ext {
        "rs" => (tree_sitter_rust::language(), 
            r#"
            (line_comment) @comment
            (block_comment) @comment
            (function_item) @func
            (type_item) @type
            (struct_item) @struct
            (enum_item) @enum
            (trait_item) @trait
            (mod_item) @mod
            (macro_definition) @macro
            "#),
        "py" => (tree_sitter_python::language(), 
            r#"
            (comment) @comment
            (function_definition) @func
            (class_definition) @class
            "#),
        "go" => (tree_sitter_go::language(), 
            r#"
            (comment) @comment
            (function_declaration) @func
            (method_declaration) @method
            (type_declaration) @type
            "#),
        "js" | "jsx" | "mjs" | "cjs" => (tree_sitter_javascript::language(), 
            r#"
            (comment) @comment
            (function_declaration) @func
            (method_definition) @method
            (arrow_function) @arrow
            (class_declaration) @class
            "#),
        "ts" => (tree_sitter_typescript::language_typescript(), 
            r#"
            (comment) @comment
            (function_declaration) @func
            (method_definition) @method
            (arrow_function) @arrow
            (interface_declaration) @interface
            (class_declaration) @class
            (enum_declaration) @enum
            "#),
        "tsx" => (tree_sitter_typescript::language_tsx(), 
            r#"
            (comment) @comment
            (function_declaration) @func
            (method_definition) @method
            (arrow_function) @arrow
            (interface_declaration) @interface
            (class_declaration) @class
            (jsx_element) @jsx
            "#),
        "java" => (tree_sitter_java::language(), 
            r#"
            (line_comment) @comment
            (block_comment) @comment
            (method_declaration) @method
            (class_declaration) @class
            (interface_declaration) @interface
            "#),
        "cpp" | "cc" | "cxx" | "h" | "hpp" => (tree_sitter_cpp::language(), 
            r#"
            (comment) @comment
            (function_definition) @func
            (class_specifier) @class
            "#),
        "php" => (unsafe { std::mem::transmute(tree_sitter_php::language_php()) }, 
            r#"
            (comment) @comment
            (function_definition) @func
            (method_declaration) @method
            (class_declaration) @class
            "#),
        "rb" => (tree_sitter_ruby::language(), 
            r#"
            (comment) @comment
            (method) @method
            (class) @class
            (module) @module
            "#),
        "cs" => (tree_sitter_c_sharp::language(), 
            r#"
            (comment) @comment
            (method_declaration) @method
            (class_declaration) @class
            (interface_declaration) @interface
            "#),
        _ => return None,
    };
    
    let mut parser = Parser::new();
    if let Err(_) = parser.set_language(&language) {
        return None;
    }

    let tree = parser.parse(content, None)?;
    let query = Query::new(&language, query_str).ok()?;
    
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, tree.root_node(), content.as_bytes());
    
    // Store (start_byte, end_byte, is_comment)
    struct NodeSpan {
        start_byte: usize,
        end_byte: usize,
        start_row: usize,
        end_row: usize,
        is_comment: bool,
    }

    let check_is_comment = |idx: u32| -> bool {
        let name: &str = query.capture_names()[idx as usize].as_ref();
        name == "comment"
    };

    let mut spans = Vec::new();
    while let Some(m) = matches.next() {
        for capture in m.captures {
             let is_comment = check_is_comment(capture.index);
             let range = capture.node.range();
             spans.push(NodeSpan {
                 start_byte: range.start_byte,
                 end_byte: range.end_byte,
                 start_row: range.start_point.row,
                 end_row: range.end_point.row,
                 is_comment,
             });
        }
    }
    
    // Sort by position
    spans.sort_by_key(|s| s.start_byte);
    
    // Merge logic
    // We want to merge a sequence of comments into the FOLLOWING code block if they are adjacent (contiguous lines).
    // Actually, simple logic:
    // Iterate spans.
    // If comment: extend "current_comment_block".
    // If code: 
    //    check if adjacent to current_comment_block. 
    //    If yes, start = comment_block.start. 
    //    Emit merged chunk.
    //    Reset comment block.
    // If code not adjacent: emit code chunk, reset comment block.
    // If comment not adjacent to previous comment: reset comment block to this comment.
    
    let mut merged_chunks = Vec::new();
    
    struct CodeChunk {
        start_row: usize,
        end_row: usize,
        start_byte: usize,
        end_byte: usize,
    }

    let mut comment_start_row: Option<usize> = None;
    let mut comment_start_byte: Option<usize> = None;
    let mut last_comment_end_row: Option<usize> = None;

    for span in spans {
        if span.is_comment {
            // Check if contiguous with previous comment
            let is_contiguous = match last_comment_end_row {
                Some(end_r) => span.start_row <= end_r + 1, // allow 0 or 1 line gap? strict adjacency means `span.start_row == end_r + 1` or `== end_r` (same line?)
                // Usually comments are on separate lines.
                // Let's allow max 1 empty line? Or strict?
                // Strict: `span.start_row == end_r + 1`.
                None => true,
            };

            if is_contiguous {
                 if comment_start_row.is_none() {
                     comment_start_row = Some(span.start_row);
                     comment_start_byte = Some(span.start_byte);
                 }
                 last_comment_end_row = Some(span.end_row);
            } else {
                 // Break in comment chain, start new
                 comment_start_row = Some(span.start_row);
                 comment_start_byte = Some(span.start_byte);
                 last_comment_end_row = Some(span.end_row);
            }
        } else {
            // It is a code block
            let mut final_start_row = span.start_row;
            let mut final_start_byte = span.start_byte;
            
            // Check if we have a pending comment block immediately before this code
            if let Some(comment_end) = last_comment_end_row {
                if span.start_row <= comment_end + 1 {
                    // It is adjacent! Include comments.
                    if let Some(c_start) = comment_start_row {
                        final_start_row = c_start;
                    }
                     if let Some(c_byte) = comment_start_byte {
                        final_start_byte = c_byte;
                    }
                }
            }
            
            // Add chunk (duplicates handled later or we just emit)
            merged_chunks.push(CodeChunk {
                start_row: final_start_row,
                end_row: span.end_row,
                start_byte: final_start_byte,
                end_byte: span.end_byte,
            });
            
            // Reset comments
            comment_start_row = None;
            comment_start_byte = None;
            last_comment_end_row = None;
        }
    }
    
    // Deduplicate exact chunks (nested nodes might trigger multiple times, though we only capture top-levels usually)
    // Actually our queries capture specific nodes.
    // Filter duplicates
    merged_chunks.dedup_by(|a, b| a.start_byte == b.start_byte && a.end_byte == b.end_byte);

    let mut file_chunks = Vec::new();
    let mut idx = 0;

    for chunk in merged_chunks {
        let start_line = chunk.start_row + 1;
        let end_line = chunk.end_row + 1;
        
        if start_line > end_line { continue; }

        let chunk_lines = end_line - start_line + 1;
        
        if chunk.end_byte > content.len() { continue; } 
        let chunk_bytes = &content.as_bytes()[chunk.start_byte..chunk.end_byte];
        let chunk_text = String::from_utf8_lossy(chunk_bytes).to_string();
        
        if chunk_lines > max_lines {
             // Split huge function using heuristic fallback
             let sub_chunks = chunk_with_heuristic(&chunk_text, relative_path, mtime, max_lines);
             for mut sub in sub_chunks {
                 sub.line_start += start_line - 1;
                 sub.line_end += start_line - 1;
                 sub.chunk_index = idx; 
                 file_chunks.push(sub);
                 idx += 1;
             }
        } else {
             file_chunks.push(FileChunk {
                 file_path: relative_path.to_string(),
                 chunk_index: idx,
                 content: chunk_text,
                 line_start: start_line,
                 line_end: end_line,
                 mtime,
             });
             idx += 1;
        }
    }
    
    Some(file_chunks)
}
    
fn chunk_with_heuristic(content: &str, relative_path: &str, mtime: u64, max_lines: usize) -> Vec<FileChunk> {
    let lines: Vec<&str> = content.lines().collect();
    let mut chunks = Vec::new();
    
    // params
    let min_chunk_size = 10;
    let max_chunk_size = max_lines; 
    let overlap = if max_lines > 16 { 8 } else { max_lines / 2 };

    let line_count = lines.len();

    if line_count <= max_chunk_size {
        chunks.push(FileChunk {
            file_path: relative_path.to_string(),
            chunk_index: 0,
            content: content.to_string(),
            line_start: 1,
            line_end: line_count,
            mtime,
        });
    } else {
        let mut start_line = 0;
        let mut idx = 0;
        
        while start_line < line_count {
            let mut end_line = start_line + min_chunk_size;
            if end_line > line_count { end_line = line_count; }

            let mut hit_limit = false;

            while end_line < line_count {
               if end_line - start_line >= max_chunk_size {
                   hit_limit = true;
                   break;
               }

               let line = lines[end_line];
               let trimmed = line.trim_start();
               let is_definition = 
                   trimmed.starts_with("fn ") || trimmed.starts_with("pub fn ") || 
                   trimmed.starts_with("async fn ") || trimmed.starts_with("pub async fn ") ||
                   trimmed.starts_with("impl ") || trimmed.starts_with("struct ") ||
                   trimmed.starts_with("enum ") || trimmed.starts_with("mod ") ||
                   trimmed.starts_with("type ") || trimmed.starts_with("trait ") ||
                   trimmed.starts_with("class ") || trimmed.starts_with("def ") || // python
                   trimmed.starts_with("func "); // go/swift
                
               if is_definition && (end_line - start_line >= min_chunk_size) {
                   break;
               }
               
               end_line += 1;
            }

            let chunk_lines = &lines[start_line..end_line];
            let chunk_content = chunk_lines.join("\n");
            
            if !chunk_content.trim().is_empty() {
                chunks.push(FileChunk {
                    file_path: relative_path.to_string(),
                    chunk_index: idx,
                    content: chunk_content,
                    line_start: start_line + 1,
                    line_end: end_line,
                    mtime,
                });
                idx += 1;
            }

            if hit_limit {
                start_line = std::cmp::max(start_line + 1, end_line - overlap);
            } else {
                start_line = end_line;
            }
        }
    }
    chunks
}

fn ensure_gitignore(root_path: &str) {
    let gitignore_path = std::path::Path::new(root_path).join(".gitignore");
    let entry = ".code-search/";

    if gitignore_path.exists() {
        if let Ok(content) = fs::read_to_string(&gitignore_path) {
            if !content.contains(entry) {
                use std::io::Write;
                if let Ok(mut file) = fs::OpenOptions::new().append(true).open(&gitignore_path) {
                    let _ = writeln!(file, "\n{}", entry);
                }
            }
        }
    } else {
         use std::io::Write;
         if let Ok(mut file) = fs::File::create(&gitignore_path) {
             let _ = writeln!(file, "{}", entry);
         }
    }
}
