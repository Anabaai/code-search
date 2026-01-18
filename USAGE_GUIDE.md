# Code-Search Usage Guide

## Overview

**code-search** is a semantic code search tool that uses vector embeddings to enable natural language queries across code repositories. This Rust implementation features:

- **AST-based chunking** for better semantic boundaries
- **Incremental indexing** - only re-indexes changed files
- **Hybrid search** - vector similarity + keyword boosting + diversity ranking
- **Zero external dependencies** - runs entirely locally

## Quick Start

### Installation

```bash
# Clone the repository
cd rust_impl

# Build the project
cargo build --release

# The binary will be at ./target/release/code-search
```

### Basic Usage

```bash
# Search in current directory
cargo run -- "your search query"

# Or using the built binary
./target/release/code-search "authentication flow"

# Search in a specific path
./target/release/code-search "error handling" --path /path/to/repo

# Limit results to 20
./target/release/code-search --limit 20 "database connection"

# Exclude patterns
./target/release/code-search "config" --exclude "*.test.*" --exclude "vendor/"
```

### MCP Server Mode

```bash
# Start MCP server (listens on stdio)
./target/release/code-search --mcp

# Or via cargo
cargo run -- --mcp
```

## Command-Line Interface

### Arguments

```
code-search [OPTIONS] [QUERY] [COMMAND]

OPTIONS:
    --mcp              Start in MCP server mode
    -h, --help         Print help
    -V, --version      Print version

ARGS:
    <QUERY>            Direct search query (alternative to `search` subcommand)

COMMANDS:
    search <QUERY>     Perform a semantic code search
    help               Print this message
```

### Search Command Options

```
code-search search <QUERY> [OPTIONS]

OPTIONS:
    -p, --path <PATH>         Repository path [default: .]
    -m, --max-lines <NUM>     Maximum lines per chunk [default: 60]
    -e, --exclude <PATTERN>   Exclude patterns (can be used multiple times)
    -l, --limit <NUM>         Max results [default: 10 or CODE_SEARCH_LIMIT env var]
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CODE_SEARCH_LIMIT` | Default number of search results | 10 |

## Configuration Files

### Ignore Patterns

The search respects these ignore files (in order of precedence):

1. **`.codesearchignore`** - Project-specific search ignores
2. **`.gitignore`** - Git ignore patterns

The `.code-search/` directory (vector database storage) is automatically added to `.gitignore`.

## How It Works

### Search Pipeline

```
1. Scan Repository
   │ Walk directory (parallel)
   │ Apply ignore patterns
   └─► Stream FileEntry

2. Detect Changes
   │ Compare file mtimes with index
   ├─► files_to_reindex (new/modified)
   └─► files_to_remove (deleted)

3. Process Changes
   │ Delete removed files from index
   ├─► Chunk files (AST-based or heuristic)
   ├─► Generate embeddings (batch size: 32)
   └─► Upsert to LanceDB

4. Execute Search
   ├─► Recall: Fetch limit × 3 candidates
   ├─► Rerank: Boost if query in content (+0.1)
   ├─► Filter: Retain scores > 0.01
   ├─► Diversity: Max 3 chunks per file
   └─► Sort by relevance score

5. Return Results
```

### Chunking Strategy

**1. AST-Based Chunking (Preferred)**
- Uses tree-sitter for language-aware parsing
- Captures: functions, classes, traits, methods, interfaces, etc.
- Supported languages: Rust, Python, Go, JavaScript/TypeScript, Java, C++, PHP, Ruby, C#

**2. Heuristic Chunking (Fallback)**
- Min 10 lines, max `--max-lines` (default: 60)
- Detects definition boundaries (fn, class, impl, struct, def, etc.)
- Overlap: `max_lines / 2` for context preservation

### Supported File Extensions

```
Source Code:
  rs, py, js, ts, jsx, tsx, go, java, cpp, c, h, hpp, php, rb, cs

Markup/Config:
  md, txt, json, yml, yaml, toml
```

### Embedding Model

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Framework**: Candle (pure Rust, CPU-only)
- **Storage**: Cached in `~/.cache/huggingface/`

### Vector Database

- **Engine**: LanceDB (embedded, file-based)
- **Location**: `.code-search/`
- **Format**: Apache Arrow (zero-copy efficiency)

## Search Result Format

```
1. src/auth/login.rs:42:58 (score: 0.87)
--------------------------------------------------
pub async fn login(username: &str, password: &str) -> Result<Session> {
    // Authenticate user credentials
    let user = authenticate(username, password).await?;
    // ... rest of function
}
--------------------------------------------------
```

Each result includes:
- **Rank**: Relevance order
- **Location**: `file_path:line_start:line_end`
- **Score**: Similarity score (0-1, higher is better)
- **Content**: Actual code snippet

## Usage Examples

### Finding Code by Intent

```bash
# "Show me how errors are handled"
./target/release/code-search "error handling"

# "Find database connection code"
./target/release/code-search "database connection"

# "Where is authentication implemented?"
./target/release/code-search "authentication login"

# "Show me configuration loading"
./target/release/code-search "config load"
```

### Refactoring Exploration

```bash
# Find similar functions
./target/release/code-search "parse user input"

# Locate test coverage
./target/release/code-search "test authentication"

# Find API endpoints
./target/release/code-search "http handler"
```

### Learning a Codebase

```bash
# Understand the architecture
./target/release/code-search "main entry point"

# Find data models
./target/release/code-search "struct user data"

# Locate async operations
./target/release/code-search "async await"
```

### Advanced Filtering

```bash
# Exclude tests
./target/release/code-search "api endpoint" --exclude "*test*" --exclude "*spec*"

# Smaller chunks for more precision
./target/release/code-search "algorithm" --max-lines 40

# More results from larger codebase
./target/release/code-search "utility" --limit 30
```

## MCP Integration

The tool can run as an MCP server for integration with AI coding assistants like Claude Code.

### Starting the Server

```bash
./target/release/code-search --mcp
```

### MCP Tool Definition

```json
{
  "name": "search",
  "description": "Perform a semantic code search. Returns a list of relevant code chunks ranked by similarity.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The search query (natural language or code snippet)"
      },
      "repository_path": {
        "type": "string",
        "description": "Path to the repository to search (default: current directory)"
      }
    },
    "required": ["query"]
  }
}
```

### MCP Usage Example

When running as an MCP server, the tool accepts search requests via stdio:

```json
{
  "method": "tools/call",
  "params": {
    "name": "search",
    "arguments": {
      "query": "how is authentication implemented?",
      "repository_path": "/path/to/project"
    }
  }
}
```

## Performance Considerations

### First Run

- Downloads model (~100MB) to `~/.cache/huggingface/`
- Scans entire repository
- Generates embeddings for all files
- **Expected time**: 10-60 seconds depending on codebase size

### Subsequent Runs

- Uses cached model
- Only re-indexes changed files (mtime-based)
- **Expected time**: 1-5 seconds for small changes

### Optimization Tips

1. **Use appropriate chunk sizes**:
   - Smaller (`--max-lines 40`): More precise results, more chunks
   - Larger (`--max-lines 100`): More context, fewer chunks

2. **Exclude unnecessary directories**:
   ```bash
   ./target/release/code-search "query" --exclude "node_modules/" --exclude "target/"
   ```

3. **Use `.codesearchignore`** for permanent excludes:
   ```
   # .codesearchignore
   *.generated.rs
   vendor/
   dist/
   ```

## Troubleshooting

### Model Download Fails

Check network connection and HuggingFace accessibility:
```bash
# Verify cache directory
ls ~/.cache/huggingface/hub/

# Clear cache if needed
rm -rf ~/.cache/huggingface/hub/
```

### Slow Performance

1. **First run is expected to be slow** - model download + full indexing
2. **Check file count** - very large repos (>100k files) take longer
3. **Reduce chunk size** - `--max-lines 40` for faster processing

### Out of Memory

The tool is optimized for typical codebases. For extremely large repositories:

1. **Use exclude patterns** to reduce scope
2. **Search subdirectories** instead of entire repo
3. **Increase swap space** if system memory is limited

### No Results Found

1. **Verify path**: Ensure `--path` points to valid directory
2. **Check ignores**: `.gitignore` patterns may exclude files
3. **Broader query**: Use more general search terms
4. **Supported languages**: Verify file extensions are supported

## Development

### Building from Source

```bash
# Debug build (faster compilation)
cargo build

# Release build (optimized binary)
cargo build --release

# Run tests
cargo test

# Run with logging
RUST_LOG=debug cargo run -- "query"
```

### Project Structure

```
src/
├── main.rs         # CLI entry point
├── mcp.rs          # MCP server implementation
├── search.rs       # Search orchestrator
├── scanner.rs      # File discovery and chunking
├── embeddings.rs   # BERT embedding generation
└── store.rs        # LanceDB vector operations
```

### Key Dependencies

| Category | Crates |
|----------|--------|
| ML/AI | `candle-core`, `candle-transformers`, `tokenizers`, `hf-hub` |
| Database | `lancedb`, `arrow-array`, `arrow-schema` |
| Parsing | `tree-sitter` + language grammars |
| Parallelism | `rayon`, `crossbeam-channel` |
| CLI | `clap`, `anyhow`, `tokio` |

## Comparison with JavaScript Version

| Feature | JavaScript | Rust |
|---------|-----------|------|
| ML Framework | @huggingface/transformers (WASM) | Candle (native) |
| Chunking | Heuristic only | AST + Heuristic |
| Indexing | Always full | Incremental (mtime-based) |
| Search | Vector only | Hybrid (recall + rerank) |
| Performance | WASM overhead | Native speed |
| Chunk Size | 1000 lines | 60 lines (default) |

## License

This project follows the same license as the parent code-search repository.
