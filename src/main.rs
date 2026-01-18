mod embeddings;

mod mcp;
pub mod scanner;
pub mod search;
mod store;
mod text_index;


use clap::{Parser, Subcommand};
use mcp::run_mcp_server;
use search::Searcher;


#[derive(Parser)]
#[command(name = "code-search")]
#[command(version = "0.1.0")]
#[command(about = "Semantic code search tool with MCP support")]
struct Cli {
    /// Run in MCP (Model Context Protocol) server mode
    #[arg(long)]
    mcp: bool,

    /// Optional subcommand (if not using MCP mode)
    #[command(subcommand)]
    command: Option<Commands>,
    
    /// Direct query argument (fallback if no subcommand)
    #[arg(index = 1)]
    direct_query: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Search the codebase
    Search {
        /// Search query
        query: String,
        
        /// Repository path
        #[arg(short, long, default_value = ".")]
        path: String,

        /// Max lines per chunk
        #[arg(long, default_value_t = 60)]
        max_lines: usize,

        /// Glob patterns to exclude
        #[arg(long)]
        exclude: Vec<String>,

        /// Limit results count
        #[arg(long)]
        limit: Option<usize>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if cli.mcp {
        // Run MCP Server
        run_mcp_server().await?;
    } else {
        // CLI Mode
        let (query, path, max_lines, exclude, limit) = match cli.command {
            Some(Commands::Search { query, path, max_lines, exclude, limit }) => (query, path, max_lines, exclude, limit),
            None => {
                if let Some(q) = cli.direct_query {
                    (q, std::env::current_dir()?.to_string_lossy().to_string(), 60, vec![], None)
                } else {
                    // Print help if no args
                    use clap::CommandFactory;
                    Cli::command().print_help()?;
                    return Ok(());
                }
            }
        };

        // Determine limit: CLI Arg > Env Var > Default (10)
        let limit = limit.unwrap_or_else(|| {
            std::env::var("CODE_SEARCH_LIMIT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10)
        });

        eprintln!("Initializing searcher (loading model)...");
        let searcher = Searcher::new()?;
        
        eprintln!("Searching for '{}' in '{}' (limit: {})...", query, path, limit);
        let results = searcher.search(&path, &query, max_lines, exclude, limit).await?;
        
        if results.is_empty() {
            println!("No results found.");
        } else {
            for (i, result) in results.iter().enumerate() {
                println!("\n{}. {}:{}:{} (score: {:.2})", 
                    i + 1, result.file_path, result.line_start, result.line_end, result.score);
                println!("--------------------------------------------------");
                println!("{}", result.content);
                println!("--------------------------------------------------");
            }
        }
    }

    Ok(())
}
