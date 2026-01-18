use anyhow::{Context, Result};
use rmcp::{
    model::{CallToolResult, Content, ListToolsResult, ErrorData, ErrorCode, CallToolRequestParam, PaginatedRequestParam},
    service::{ServiceExt, RequestContext, RoleServer},
    tool, tool_router,
    handler::server::{
        ServerHandler,
        router::tool::ToolRouter,
        wrapper::Parameters,
    },
    // RmcpError,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use crate::search::Searcher;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct SearchArgs {
    pub query: String,
    pub repository_path: Option<String>,
}

#[derive(Clone)]
pub struct McpServer {
    tool_router: ToolRouter<Self>,
    searcher: Arc<Mutex<Option<Searcher>>>,
}

#[tool_router]
impl McpServer {
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
            searcher: Arc::new(Mutex::new(None)),
        }
    }

    #[tool(name = "search", description = "Perform a semantic code search. Returns a list of relevant code chunks with their file path, line numbers, and similarity score.")]
    async fn search(&self, args: Parameters<SearchArgs>) -> Result<CallToolResult, ErrorData> {
        let query = &args.0.query;
        let path = args.0.repository_path.as_deref().unwrap_or(".");
        
        eprintln!("Searching for '{}' in '{}'...", query, path);

        let mut searcher_guard = self.searcher.lock().await;
        
        if searcher_guard.is_none() {
             eprintln!("Initializing searcher (loading model)...");
            let searcher = Searcher::new().map_err(|e| {
                ErrorData {
                    code: ErrorCode(-32000),
                    message: format!("Failed to initialize searcher: {}", e).into(),
                    data: None
                }
            })?;
            *searcher_guard = Some(searcher);
        }
        
        let searcher = searcher_guard.as_mut().unwrap();

        let limit = std::env::var("CODE_SEARCH_LIMIT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);

        let results = searcher.search(path, query, 60, vec![], limit).await.map_err(|e| {
             ErrorData {
                code: ErrorCode(-32000),
                message: format!("Search failed: {}", e).into(),
                data: None
             }
        })?;

        let mut text_output = String::new();
        if results.is_empty() {
            text_output.push_str("No results found.");
        } else {
            for result in results {
                 text_output.push_str(&format!(
                    "{}:{}:{} (score: {:.2})\n",
                    result.file_path, result.line_start, result.line_end, result.score
                ));
                text_output.push_str("--------------------------------------------------\n");
                text_output.push_str(&result.content);
                 text_output.push_str("\n--------------------------------------------------\n\n");
            }
        }

        Ok(CallToolResult::success(vec![Content::text(text_output)]))
    }
}

impl ServerHandler for McpServer {
    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParam>,
        _ctx: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        let tools = self.tool_router.list_all();
        Ok(ListToolsResult {
            tools,
            next_cursor: None,
            meta: None,
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParam,
        _ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        // Manual dispatch since ToolRouter delegation is proving difficult with private fields/traits
        if request.name == "search" {
             let args: SearchArgs = if let Some(args_map) = request.arguments {
                 serde_json::from_value(serde_json::Value::Object(args_map)).map_err(|e| {
                     ErrorData {
                         code: ErrorCode(-32602), // Invalid params
                         message: format!("Invalid arguments: {}", e).into(),
                         data: None
                     }
                 })?
             } else {
                 return Err(ErrorData {
                     code: ErrorCode(-32602),
                     message: "Missing arguments".into(),
                     data: None
                 });
             };

             return self.search(Parameters(args)).await;
        }

        Err(ErrorData {
            code: ErrorCode(-32601), // Method not found
            message: format!("Tool not found: {}", request.name).into(),
            data: None
        })
    }
}

pub async fn run_mcp_server() -> Result<()> {
    let server = McpServer::new();
    let transport = rmcp::transport::io::stdio();
    
    server.serve(transport).await.context("MCP server failed")?;
    
    Ok(())
}
