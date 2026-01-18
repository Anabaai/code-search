# Architecture Guide: Code Search (Rust Implementation)

## Overview

**code-search-mcp** is an ultra-fast semantic code search tool built in Rust that enables natural language queries across code repositories using vector embeddings. This is a high-performance Rust reimplementation of the original JavaScript version, with two execution modes:

1. **CLI Mode**: Direct command-line semantic search
2. **MCP Server Mode**: Model Context Protocol server for AI assistant integration

**Key Design Philosophy**: High-performance, incremental indexing, and zero persistent background processes. Everything runs on-demand with intelligent caching.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Entry Points                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  main.rs (CLI)            │           mcp.rs (MCP Server)                │
│  ──────────────────────   │           ───────────────────────────       │
│  • CLI parsing (clap)     │           • rmcp protocol handler            │
│  • Direct query mode      │           • Tool routing & dispatch          │
│  • Search subcommand      │           • Lazy Searcher initialization    │
└───────────┬───────────────────────────────┬─────────────────────────────┘
            │                               │
            └───────────────┬───────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Searcher (search.rs)                             │
│  ────────────────────────────────────────────────────────────────────  │
│  Orchestrates the entire search pipeline:                               │
│  1. Scan Repository (scanner.rs)                                        │
│  2. Compute Diffs (incremental indexing)                                │
│  3. Generate Embeddings (embeddings.rs)                                 │
│  4. Upsert to Vector Store (store.rs)                                   │
│  5. Hybrid Search (Recall + Rerank)                                     │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Scanner    │   │  Embeddings  │   │  VectorStore │
│  (scanner)   │   │(embeddings)  │   │   (store)    │
├──────────────┤   ├──────────────┤   ├──────────────┤
│• File walk   │   │• Candle BERT │   │• LanceDB     │
│• Tree-sitter │   │• MiniLM-L6   │   │• Arrow schema│
│• Chunking    │   │• Batch proc. │   │• Vector search│
└──────────────┘   └──────────────┘   └──────────────┘
```

---

## Component Deep-Dive

### 1. Entry Points (`main.rs` / `mcp.rs`)

#### main.rs - CLI Entry Point
```rust
// Dual-mode execution via clap CLI parsing
Cli {
    mcp: bool,              // --mcp flag for server mode
    command: Option<Commands>,  // search subcommand
    direct_query: Option<String>,  // positional query arg
}
```

**Flow:**
1. Parse CLI arguments
2. If `--mcp`: Launch MCP server
3. Otherwise: Initialize Searcher → search → print results

**Key Functions:**
- `main()`: Entry point, async runtime setup
- CLI limit resolution: CLI Arg > Env Var > Default (10)

#### mcp.rs - MCP Server Entry Point
```rust
pub struct McpServer {
    tool_router: ToolRouter<Self>,
    searcher: Arc<Mutex<Option<Searcher>>>,  // Lazy initialization
}
```

**Flow:**
1. Listen on stdio (MCP transport)
2. Route "search" tool calls to Searcher
3. Lazy model loading on first search

**Key Functions:**
- `run_mcp_server()`: Spawns stdio MCP server
- `search()`: Tool handler, delegates to Searcher

---

### 2. Searcher (`search.rs`) - The Orchestrator

**Responsibility**: Coordinates the entire search pipeline with incremental indexing.

```rust
pub struct Searcher {
    model: EmbeddingModel,
}

pub async fn search(&self, repo_path, query, max_lines, exclude, limit) -> Vec<SearchResult>
```

**Pipeline Flow (search.rs:19-161):**

```
1. Scan Repository  ──►  FileChunk stream (via crossbeam channel)
                            │
2. Fetch Metadata  ──►    HashMap<String, u64>  // file_path → mtime
                            │
3. Compute Diffs  ──►     • files_to_reindex: HashSet<String>
                          • files_to_remove: Vec<String>
                            │
4. Handle Deletions  ──►  store.delete_files(&files_to_remove)
                            │
5. Batch Embeddings  ──►  model.embed_batch(chunks) // 32 chunks/batch
                            │
6. Upsert  ──►            store.upsert(&chunks, &embeddings)
                            │
7. Hybrid Search  ──►     • Recall: limit * 3 candidates
                          • Rerank: Keyword boost (+0.5 if query in content)
                          • Truncate to original limit
```

**Incremental Indexing Strategy:**
- Tracks file modification times (mtime)
- Only re-indexes changed files
- Removes deleted files from index
- Hybird recall + rerank for quality results

---

### 3. Scanner (`scanner.rs`) - File Discovery & Chunking

**Responsibility**: Walks directory tree, filters files, and creates semantic chunks.

```rust
pub struct FileChunk {
    pub file_path: String,
    pub chunk_index: usize,
    pub content: String,
    pub line_start: usize,
    pub line_end: usize,
    pub mtime: u64,  // For incremental indexing
}
```

**Two-Stage Chunking Strategy (scanner.rs:80-100):**

```
1. AST-Based Chunking (tree-sitter)
   ├── Language-specific queries
   ├── Captures functions, classes, traits, etc.
   ├── Falls back if parsing fails or file too large

2. Heuristic Chunking (fallback)
   ├── Min 10 lines, max_lines parameter
   ├── Detects definition boundaries (fn, class, etc.)
   ├── Overlap: max_lines / 2 for context preservation
```

**Supported Languages (AST):**
- Rust, Python, Go, JavaScript/TypeScript/TSX, Java, C++, PHP, Ruby, C#

**Ignore Mechanism:**
- Uses `ignore` crate with `.gitignore` support
- Custom `.codesearchignore` file support
- CLI `--exclude` glob patterns
- Auto-adds `.code-search/` to `.gitignore`

---

### 4. Embeddings (`embeddings.rs`) - Vector Generation

**Responsibility**: Generates 384-dimensional vectors using sentence-transformers.

```rust
pub struct EmbeddingModel {
    model: BertModel,           // Candle BERT model
    tokenizer: Tokenizer,       // HuggingFace tokenizers
    device: Device,             // CPU-only for portability
}
```

**Model Configuration:**
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Device**: CPU (portability, no GPU requirement)
- **Framework**: Candle (Rust ML framework)

**Embedding Pipeline (embeddings.rs:47-94):**

```
1. Tokenize batch (tokenizer.encode_batch)
   │
2. Convert to Tensors (token_ids, attention_mask)
   │
3. BERT forward pass (model.forward)
   │
4. Mean pooling with attention mask
   │
5. L2 normalization
   │
6. Return Vec<Vec<f32>> (batch of embeddings)
```

**Batch Processing:**
- Default batch size: 32 chunks
- Progress logging every 320 chunks

---

### 5. Vector Store (`store.rs`) - LanceDB Integration

**Responsibility**: Manages vector database operations with LanceDB.

```rust
pub struct VectorStore {
    conn: Connection,        // LanceDB connection
    table_name: String,      // "code_chunks"
}

pub struct SearchResult {
    pub file_path: String,
    pub chunk_index: usize,
    pub content: String,
    pub line_start: usize,
    pub line_end: usize,
    pub score: f32,          // 1.0 - distance (similarity)
}
```

**Schema (Arrow/LanceDB):**
```rust
Schema::new(vec![
    Field::new("file_path", DataType::Utf8, false),
    Field::new("chunk_index", DataType::Int32, false),
    Field::new("content", DataType::Utf8, false),
    Field::new("line_start", DataType::Int32, false),
    Field::new("line_end", DataType::Int32, false),
    Field::new("mtime", DataType::Int64, false),
    Field::new("vector", DataType::FixedSizeList(Float32, 384), false),
])
```

**Key Operations:**

1. **get_indexed_metadata()**: Fetch all file_path → mtime mappings
2. **upsert()**: Delete old versions → Insert new chunks
3. **delete_files()**: Remove deleted files
4. **search()**: Vector similarity search with distance→score conversion
5. **cleanup()**: Prune old versions, compact fragments

**Storage Location:**
`.code-search/` directory (auto-added to `.gitignore`)

---

## Data Flow Diagram

```
┌─────────────┐
│   Query     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  Searcher::search()                                          │
│  ─────────────────────────────────────────────────────────  │
│  1. scan_repository() ──► crossbeam channel ──► FileChunks  │
│  2. get_indexed_metadata() ──► HashMap<path, mtime>         │
│  3. Compute diffs (new/modified/deleted files)              │
│  4. delete_files() for deletions                            │
│  5. embed_batch() for modified chunks (32 at a time)        │
│  6. upsert() new chunks                                     │
│  7. vector_search() with hybrid recall+rerank               │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│  Results    │
└─────────────┘
```

---

## Concurrency Model

**Parallel File Scanning:**
- `WalkBuilder::build_parallel()` for directory traversal
- `crossbeam_channel::unbounded()` for chunk streaming
- Thread spawn for scanner, async/await for embeddings

**Async/Await:**
- Tokio runtime for async operations
- LanceDB operations are async
- Embedding generation is sync (CPU-bound)

```
Main Thread
    │
    ├──► spawn thread ──► scan_repository ──► crossbeam channel
    │
    └──► rx.iter() ──► collect chunks ──► embed_batch
```

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Embedding Dimensions | 384 |
| Batch Size | 32 chunks |
| Chunk Size | Default 60 lines (configurable) |
| Overlap | max_lines / 2 |
| Model | all-MiniLM-L6-v2 (sentence-transformers) |
| Backend | Candle (CPU-only) |
| Vector DB | LanceDB (embedded) |

**Incremental Indexing:**
- First run: Full scan + embedding (slowest)
- Subsequent runs: Only changed files re-indexed
- mtime-based diff detection

---

## Key Design Decisions

### 1. **Candle over PyTorch/TensorFlow**
- Pure Rust, no Python dependency
- CPU-only for portability
- Sufficient performance for embedding inference

### 2. **LanceDB over Vector DBs**
- Embedded, file-based (no server required)
- Arrow-based for zero-copy efficiency
- Built-in versioning for incremental updates

### 3. **Tree-sitter for Chunking**
- Language-aware chunk boundaries
- Better semantic coherence than line-based
- Fallback to heuristics for unsupported languages

### 4. **Hybrid Recall + Rerank**
- Recall: 3x limit to ensure coverage
- Rerank: Keyword boost (+0.5) for exact matches
- Balances semantic + lexical relevance

### 5. **MCP Protocol**
- Stdio transport for simplicity
- Lazy model loading on first search
- Stateless server design

---

## Configuration

### Environment Variables
- `CODE_SEARCH_LIMIT`: Default result limit (default: 10)

### CLI Arguments
```
code-search [OPTIONS] [QUERY]

OPTIONS:
    --mcp                  Run in MCP server mode
    -h, --help             Print help

SUBCOMMANDS:
    search                 Search the codebase
        --path <PATH>      Repository path [default: .]
        --max-lines <N>    Max lines per chunk [default: 60]
        --exclude <PATTERN> Glob patterns to exclude
        --limit <N>        Limit results count
```

### MCP Tool
```json
{
  "name": "search",
  "description": "Perform a semantic code search",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "repository_path": {"type": "string"}
    },
    "required": ["query"]
  }
}
```

---

## Dependency Overview

### Core
- `anyhow`: Error handling
- `clap`: CLI parsing
- `tokio`: Async runtime
- `serde`/`serde_json`: Serialization

### ML
- `candle-core`/`candle-nn`/`candle-transformers`: ML framework
- `tokenizers`: HuggingFace tokenizers
- `hf-hub`: Model downloads

### Database
- `lancedb`: Vector database
- `arrow-array`/`arrow-schema`: Arrow format

### Code Analysis
- `tree-sitter`: Parser library
- `tree-sitter-*`: Language grammars

### Utilities
- `ignore`: .gitignore handling
- `crossbeam-channel`: Multi-producer multi-consumer channels
- `rayon`: Parallelism
- `pathdiff`: Relative path computation

---

## File Structure

```
src/
├── main.rs          # CLI entry point
├── mcp.rs           # MCP server implementation
├── search.rs        # Search orchestrator
├── scanner.rs       # File scanning and chunking
├── embeddings.rs    # Embedding generation
└── store.rs         # LanceDB integration
```

---

## Common Patterns

### Error Handling
- `anyhow::Result` for application code
- `?` operator for propagation
- Context with `.context()`

### Async/Await
- `async fn` for LanceDB operations
- `tokio::sync::Mutex` for shared state
- `crossbeam_channel` for thread communication

### Resource Management
- RAII for LanceDB connection cleanup
- Arrow arrays for zero-copy data transfer
- Mmap for model weights (`VarBuilder::from_mmaped_safetensors`)

---

## Future Considerations

1. **GPU Support**: Add GPU device option for Candle
2. **Streaming Embeddings**: Process chunks while scanning
3. **Index Compression**: Reduce disk footprint
4. **Hybrid Search**: Add BM25 lexical scoring
5. **Caching**: Embedding cache for identical chunks
6. **More Languages**: Additional tree-sitter grammars

---

## Comparison with JavaScript Version

| Feature | JS Version | Rust Version |
|---------|------------|--------------|
| ML Framework | @huggingface/transformers (WASM) | Candle (native) |
| Chunking | Heuristic only | AST + Heuristic |
| Indexing | Always full | Incremental (mtime) |
| Search | Vector only | Hybrid (recall + rerank) |
| Parallelism | Single-threaded | Parallel scanning |
| Dependencies | npm | Cargo |

---

## Troubleshooting

### Model Download Issues
- Models cached in `~/.cache/huggingface/`
- Check network connectivity for first run

### Large Repository Performance
- First run is slow (full index)
- Subsequent runs faster (incremental)
- Consider `--max-lines` tuning

### Memory Usage
- Embedding generation: ~500MB for model
- LanceDB: Grows with repository size
- Cleanup with `store.cleanup()`
