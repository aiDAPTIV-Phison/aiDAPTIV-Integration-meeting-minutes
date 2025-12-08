use reqwest::{header, Client};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};
use futures_util::StreamExt;

// Generic structure for OpenAI-compatible API chat messages
#[derive(Debug, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

// Generic structure for OpenAI-compatible API chat requests
#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
}

// Generic structure for OpenAI-compatible API chat responses
#[derive(Deserialize, Debug)]
pub struct ChatResponse {
    pub choices: Vec<Choice>,
}

#[derive(Deserialize, Debug)]
pub struct Choice {
    pub message: MessageContent,
}

#[derive(Deserialize, Debug)]
pub struct MessageContent {
    pub content: String,
}

// Streaming chat request structure (used internally for summary generation)
#[derive(Debug, Serialize)]
struct StreamingChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

// Streaming response chunk structure (OpenAI-compatible format)
#[derive(Deserialize, Debug)]
struct StreamChatChunk {
    choices: Vec<StreamChoice>,
}

#[derive(Deserialize, Debug)]
struct StreamChoice {
    delta: StreamDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct StreamDelta {
    content: Option<String>,
}

// Claude streaming event types
#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum ClaudeStreamEvent {
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { delta: ClaudeDelta },
    #[serde(rename = "error")]
    Error { error: ClaudeError },
    #[serde(other)]
    Other,
}

#[derive(Deserialize, Debug)]
struct ClaudeDelta {
    #[serde(rename = "type")]
    delta_type: String,
    text: Option<String>,
}

#[derive(Deserialize, Debug)]
struct ClaudeError {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

// Claude-specific request structure
#[derive(Debug, Serialize)]
pub struct ClaudeRequest {
    pub model: String,
    pub max_tokens: u32,
    pub system: String,
    pub messages: Vec<ChatMessage>,
}

// Claude-specific response structure
#[derive(Deserialize, Debug)]
pub struct ClaudeChatResponse {
    pub content: Vec<ClaudeChatContent>,
}

#[derive(Deserialize, Debug)]
pub struct ClaudeChatContent {
    pub text: String,
}

// ============================================================================
// TTFT LOGGING UTILITIES
// ============================================================================

/// Log unified TTFT metrics in a clean 3-line format
///
/// # Arguments
/// * `ttft_us` - Time to first token in microseconds
/// * `total_time_us` - Total streaming time in microseconds
fn log_ttft_metrics(ttft_us: Option<u64>, total_time_us: u64) {
    if let Some(ttft) = ttft_us {
        let t1_to_tn_us = total_time_us - ttft;
        let ttft_ms = ttft as f64 / 1000.0;
        let t1_to_tn_ms = t1_to_tn_us as f64 / 1000.0;
        let total_ms = total_time_us as f64 / 1000.0;
        let ttft_ratio = (ttft as f64 / total_time_us as f64) * 100.0;

        info!("⏱️  Timing: t0->t1: {:.0}ms | t1->tn: {:.0}ms | total: {:.0}ms",
              ttft_ms, t1_to_tn_ms, total_ms);
        info!("⏱️  TTFT: {:.0}ms ({:.1}% of total)", ttft_ms, ttft_ratio);
        info!("⏱️  Tokens/time: t1->tn took {:.1}x longer than TTFT",
              t1_to_tn_ms / ttft_ms.max(1.0));
    }
}

// ============================================================================
// LLM PROVIDER TYPES
// ============================================================================

/// LLM Provider enumeration for multi-provider support
#[derive(Debug, Clone, PartialEq)]
pub enum LLMProvider {
    OpenAI,
    Claude,
    Groq,
    Ollama,
    OpenRouter,
    OpenAICompatible,
}

impl LLMProvider {
    /// Parse provider from string (case-insensitive)
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(Self::OpenAI),
            "claude" => Ok(Self::Claude),
            "groq" => Ok(Self::Groq),
            "ollama" => Ok(Self::Ollama),
            "openrouter" => Ok(Self::OpenRouter),
            "openai-compatible" => Ok(Self::OpenAICompatible),
            _ => Err(format!("Unsupported LLM provider: {}", s)),
        }
    }
}

/// Summary generation result with timing metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryResult {
    pub content: String,
    pub ttft_us: Option<u64>,    // Time to first token in microseconds
    pub total_time_us: u64,       // Total generation time in microseconds (tn - t0)
}

/// Generates a summary using the specified LLM provider
///
/// # Arguments
/// * `client` - Reqwest HTTP client (reused for performance)
/// * `provider` - The LLM provider to use
/// * `model_name` - The specific model to use (e.g., "gpt-4", "claude-3-opus")
/// * `api_key` - API key for the provider (not needed for Ollama)
/// * `system_prompt` - System instructions for the LLM
/// * `user_prompt` - User query/content to process
/// * `ollama_endpoint` - Optional custom Ollama endpoint (defaults to localhost:11434)
/// * `openai_compatible_endpoint` - Optional custom OpenAI-compatible endpoint
/// * `stream` - Whether to use streaming mode (enables TTFT tracking)
///
/// # Returns
/// SummaryResult with content, ttft_us (if stream=true), and total_time_us
pub async fn generate_summary(
    client: &Client,
    provider: &LLMProvider,
    model_name: &str,
    api_key: &str,
    system_prompt: &str,
    user_prompt: &str,
    ollama_endpoint: Option<&str>,
    openai_compatible_endpoint: Option<&str>,
    stream: bool,
) -> Result<SummaryResult, String> {
    use std::time::Instant;

    // Route to streaming or non-streaming based on parameter
    if stream {
        _generate_summary_streaming(
            client,
            provider,
            model_name,
            api_key,
            system_prompt,
            user_prompt,
            ollama_endpoint,
            openai_compatible_endpoint,
        )
        .await
    } else {
        _generate_summary_non_streaming(
            client,
            provider,
            model_name,
            api_key,
            system_prompt,
            user_prompt,
            ollama_endpoint,
            openai_compatible_endpoint,
        )
        .await
    }
}

/// Internal: Non-streaming summary generation (original logic, ttft_us is always None)
async fn _generate_summary_non_streaming(
    client: &Client,
    provider: &LLMProvider,
    model_name: &str,
    api_key: &str,
    system_prompt: &str,
    user_prompt: &str,
    ollama_endpoint: Option<&str>,
    openai_compatible_endpoint: Option<&str>,
) -> Result<SummaryResult, String> {
    use std::time::Instant;
    let request_start_time = Instant::now();

    let (api_url, mut headers) = match provider {
        LLMProvider::OpenAI => (
            "https://api.openai.com/v1/chat/completions".to_string(),
            header::HeaderMap::new(),
        ),
        LLMProvider::Groq => (
            "https://api.groq.com/openai/v1/chat/completions".to_string(),
            header::HeaderMap::new(),
        ),
        LLMProvider::OpenRouter => (
            "https://openrouter.ai/api/v1/chat/completions".to_string(),
            header::HeaderMap::new(),
        ),
        LLMProvider::Ollama => {
            let host = ollama_endpoint
                .map(|s| s.to_string())
                .unwrap_or_else(|| "http://localhost:11434".to_string());
            (
                format!("{}/v1/chat/completions", host),
                header::HeaderMap::new(),
            )
        },
        LLMProvider::Claude => {
            let mut header_map = header::HeaderMap::new();
            header_map.insert(
                "x-api-key",
                api_key
                    .parse()
                    .map_err(|_| "Invalid API key format".to_string())?,
            );
            header_map.insert(
                "anthropic-version",
                "2023-06-01"
                    .parse()
                    .map_err(|_| "Invalid anthropic version".to_string())?,
            );
            ("https://api.anthropic.com/v1/messages".to_string(), header_map)
        },
        LLMProvider::OpenAICompatible => {
            let base_url = openai_compatible_endpoint
                .ok_or("OpenAI Compatible endpoint not configured")?
                .trim_end_matches('/');
            (
                format!("{}/v1/chat/completions", base_url),
                header::HeaderMap::new(),
            )
        }
    };

    // Add authorization header for non-Claude providers
    if provider != &LLMProvider::Claude {
        // OpenAI Compatible might not need API key (local services)
        if !api_key.is_empty() || provider != &LLMProvider::OpenAICompatible {
            headers.insert(
                header::AUTHORIZATION,
                format!("Bearer {}", api_key)
                    .parse()
                    .map_err(|_| "Invalid authorization header".to_string())?,
            );
        }
    }
    headers.insert(
        header::CONTENT_TYPE,
        "application/json"
            .parse()
            .map_err(|_| "Invalid content type".to_string())?,
    );

    // Build request body based on provider
    let request_body = if provider != &LLMProvider::Claude {
        serde_json::json!(ChatRequest {
            model: model_name.to_string(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt.to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: user_prompt.to_string(),
                }
            ],
        })
    } else {
        serde_json::json!(ClaudeRequest {
            system: system_prompt.to_string(),
            model: model_name.to_string(),
            max_tokens: 2048,
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: user_prompt.to_string(),
            }]
        })
    };

    info!("🐞 LLM Request to {}: model={} (non-streaming)", provider_name(provider), model_name);

    // Send request
    let response = client
        .post(api_url)
        .headers(headers)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| format!("Failed to send request to LLM: {}", e))?;

    if !response.status().is_success() {
        let error_body = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(format!("LLM API request failed: {}", error_body));
    }

    // Parse response based on provider
    if provider == &LLMProvider::Claude {
        let chat_response = response
            .json::<ClaudeChatResponse>()
            .await
            .map_err(|e| format!("Failed to parse LLM response: {}", e))?;

        info!("🐞 LLM Response received from Claude");

        let content = chat_response
            .content
            .get(0)
            .ok_or("No content in LLM response")?
            .text
            .trim();

        let total_time_us = request_start_time.elapsed().as_micros() as u64;
        info!("⏱️ Summary generation total time: {:.2}ms", total_time_us as f64 / 1000.0);

        // For non-streaming, we don't have TTFT but can measure total time
        Ok(SummaryResult {
            content: content.to_string(),
            ttft_us: None, // Non-streaming doesn't capture TTFT
            total_time_us,
        })
    } else {
        let chat_response = response
            .json::<ChatResponse>()
            .await
            .map_err(|e| format!("Failed to parse LLM response: {}", e))?;

        info!("🐞 LLM Response received from {}", provider_name(provider));

        let content = chat_response
            .choices
            .get(0)
            .ok_or("No content in LLM response")?
            .message
            .content
            .trim();

        let total_time_us = request_start_time.elapsed().as_micros() as u64;
        info!("⏱️ Summary generation total time: {:.2}ms", total_time_us as f64 / 1000.0);

        Ok(SummaryResult {
            content: content.to_string(),
            ttft_us: None, // Non-streaming doesn't capture TTFT
            total_time_us,
        })
    }
}

/// Internal: Streaming summary generation (consumes stream internally, captures TTFT)
///
/// This function uses streaming to capture TTFT metrics but accumulates all tokens
/// internally before returning, so the caller receives a complete response.
async fn generate_summary_streaming(
    client: &Client,
    provider: &LLMProvider,
    model_name: &str,
    api_key: &str,
    system_prompt: &str,
    user_prompt: &str,
    ollama_endpoint: Option<&str>,
    openai_compatible_endpoint: Option<&str>,
) -> Result<SummaryResult, String> {
    use std::time::Instant;

    // Route to appropriate streaming handler based on provider
    if provider == &LLMProvider::Claude {
        _generate_summary_streaming_claude(
            client,
            model_name,
            api_key,
            system_prompt,
            user_prompt,
        )
        .await
    } else {
        _generate_summary_streaming(
            client,
            provider,
            model_name,
            api_key,
            system_prompt,
            user_prompt,
            ollama_endpoint,
            openai_compatible_endpoint,
        )
        .await
    }
}

/// Internal: Streaming handler for OpenAI-compatible providers (reuses chat streaming logic)
async fn _generate_summary_streaming(
    client: &Client,
    provider: &LLMProvider,
    model_name: &str,
    api_key: &str,
    system_prompt: &str,
    user_prompt: &str,
    ollama_endpoint: Option<&str>,
    openai_compatible_endpoint: Option<&str>,
) -> Result<SummaryResult, String> {
    use std::time::Instant;
    let request_start_time = Instant::now();

    // Determine API endpoint (reuse logic from stream_chat_openai_compatible)
    let (api_url, mut headers) = match provider {
        LLMProvider::OpenAI => (
            "https://api.openai.com/v1/chat/completions".to_string(),
            header::HeaderMap::new(),
        ),
        LLMProvider::Groq => (
            "https://api.groq.com/openai/v1/chat/completions".to_string(),
            header::HeaderMap::new(),
        ),
        LLMProvider::OpenRouter => (
            "https://openrouter.ai/api/v1/chat/completions".to_string(),
            header::HeaderMap::new(),
        ),
        LLMProvider::Ollama => {
            let host = ollama_endpoint
                .map(|s| s.to_string())
                .unwrap_or_else(|| "http://localhost:11434".to_string());
            (
                format!("{}/v1/chat/completions", host),
                header::HeaderMap::new(),
            )
        }
        LLMProvider::OpenAICompatible => {
            let base_url = openai_compatible_endpoint
                .ok_or("OpenAI Compatible endpoint not configured")?
                .trim_end_matches('/');
            (
                format!("{}/chat/completions", base_url),
                header::HeaderMap::new(),
            )
        }
        _ => return Err("Unsupported provider for OpenAI-compatible streaming".to_string()),
    };

    // Add authorization header
    if !api_key.is_empty() || provider != &LLMProvider::OpenAICompatible {
        headers.insert(
            header::AUTHORIZATION,
            format!("Bearer {}", api_key)
                .parse()
                .map_err(|_| "Invalid authorization header".to_string())?,
        );
    }
    headers.insert(
        header::CONTENT_TYPE,
        "application/json"
            .parse()
            .map_err(|_| "Invalid content type".to_string())?,
    );

    // Build streaming request body
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: system_prompt.to_string(),
        },
        ChatMessage {
            role: "user".to_string(),
            content: user_prompt.to_string(),
        },
    ];

    let request_body = StreamingChatRequest {
        model: model_name.to_string(),
        messages,
        stream: true,
        temperature: None,
        top_p: None,
        max_tokens: Some(2048),
    };

    info!("🚀 Sending streaming summary request to {}: {}", provider_name(provider), api_url);

    // Send request
    let response = client
        .post(&api_url)
        .headers(headers)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| format!("Failed to send streaming request: {}", e))?;

    if !response.status().is_success() {
        let error_body = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(format!("Streaming API request failed: {}", error_body));
    }

    // Process streaming response - accumulate tokens internally
    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut accumulated_content = String::new();
    let mut ttft_us: Option<u64> = None;
    let mut first_token_received = false;

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                // Convert bytes to string
                let chunk_str = String::from_utf8_lossy(&chunk);
                buffer.push_str(&chunk_str);

                // Process complete lines (SSE format: "data: {...}\n\n")
                while let Some(line_end) = buffer.find("\n\n") {
                    let line = buffer[..line_end].to_string();
                    buffer = buffer[line_end + 2..].to_string();

                    // Parse SSE line
                    for sse_line in line.lines() {
                        if sse_line.starts_with("data: ") {
                            let data = &sse_line[6..]; // Skip "data: " prefix

                            // Check for [DONE] signal
                            if data.trim() == "[DONE]" {
                                info!("🏁 Received [DONE] signal for summary");
                                break;
                            }

                            // Parse JSON chunk
                            match serde_json::from_str::<StreamChatChunk>(data) {
                                Ok(parsed_chunk) => {
                                    // Extract content delta
                                    if let Some(choice) = parsed_chunk.choices.first() {
                                        if let Some(content) = &choice.delta.content {
                                            if !content.is_empty() {
                                                // Calculate TTFT on first token
                                                if !first_token_received {
                                                    first_token_received = true;
                                                    ttft_us = Some(request_start_time.elapsed().as_micros() as u64);
                                                    info!("⏱️ First token received for summary: {:.2}ms",
                                                          ttft_us.unwrap() as f64 / 1000.0);
                                                }

                                                // Accumulate content internally
                                                accumulated_content.push_str(content);
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    warn!("⚠️ Failed to parse streaming chunk: {} | Data: {}", e, data);
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                return Err(format!("Stream error: {}", e));
            }
        }
    }

    // Calculate total time
    let total_time_us = request_start_time.elapsed().as_micros() as u64;

    // Log unified TTFT metrics
    log_ttft_metrics(ttft_us, total_time_us);

    info!("✅ Summary streaming completed: {} chars, TTFT: {:?}μs",
          accumulated_content.len(), ttft_us);

    Ok(SummaryResult {
        content: accumulated_content,
        ttft_us,
        total_time_us,
    })
}

/// Internal: Streaming handler for Claude (reuses Claude streaming logic)
async fn _generate_summary_streaming_claude(
    client: &Client,
    model_name: &str,
    api_key: &str,
    system_prompt: &str,
    user_prompt: &str,
) -> Result<SummaryResult, String> {
    use std::time::Instant;
    let request_start_time = Instant::now();

    // Build headers
    let mut headers = header::HeaderMap::new();
    headers.insert(
        "x-api-key",
        api_key
            .parse()
            .map_err(|_| "Invalid API key format".to_string())?,
    );
    headers.insert(
        "anthropic-version",
        "2023-06-01"
            .parse()
            .map_err(|_| "Invalid anthropic version".to_string())?,
    );
    headers.insert(
        header::CONTENT_TYPE,
        "application/json"
            .parse()
            .map_err(|_| "Invalid content type".to_string())?,
    );

    // Build streaming request body
    let request_body = serde_json::json!({
        "model": model_name,
        "max_tokens": 2048,
        "messages": vec![ChatMessage {
            role: "user".to_string(),
            content: user_prompt.to_string(),
        }],
        "system": system_prompt,
        "stream": true,
    });

    let api_url = "https://api.anthropic.com/v1/messages";
    info!("🚀 Sending streaming summary request to Claude: {}", api_url);

    // Send request
    let response = client
        .post(api_url)
        .headers(headers)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| format!("Failed to send streaming request to Claude: {}", e))?;

    if !response.status().is_success() {
        let error_body = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(format!("Claude streaming API request failed: {}", error_body));
    }

    // Process streaming response - accumulate tokens internally
    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut accumulated_content = String::new();
    let mut ttft_us: Option<u64> = None;
    let mut first_token_received = false;

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                let chunk_str = String::from_utf8_lossy(&chunk);
                buffer.push_str(&chunk_str);

                // Process complete lines (SSE format: "event: ...\ndata: {...}\n\n")
                while let Some(line_end) = buffer.find("\n\n") {
                    let block = buffer[..line_end].to_string();
                    buffer = buffer[line_end + 2..].to_string();

                    // Parse SSE block
                    let mut data: Option<String> = None;

                    for line in block.lines() {
                        if line.starts_with("data: ") {
                            data = Some(line[6..].to_string());
                        }
                    }

                    if let Some(data_str) = data {
                        match serde_json::from_str::<ClaudeStreamEvent>(&data_str) {
                            Ok(event) => {
                                match event {
                                    ClaudeStreamEvent::ContentBlockDelta { delta } => {
                                        // Extract text from delta if available
                                        if let Some(text) = delta.text {
                                            // Calculate TTFT on first token
                                            if !first_token_received && !text.is_empty() {
                                                first_token_received = true;
                                                ttft_us = Some(request_start_time.elapsed().as_micros() as u64);
                                                info!("⏱️ First token received for Claude summary: {:.2}ms",
                                                      ttft_us.unwrap() as f64 / 1000.0);
                                            }

                                            // Accumulate content internally
                                            accumulated_content.push_str(&text);
                                        }
                                    }
                                    ClaudeStreamEvent::Error { error } => {
                                        return Err(format!("Claude error: {}", error.message));
                                    }
                                    _ => {}
                                }
                            }
                            Err(e) => {
                                warn!("⚠️ Failed to parse Claude streaming event: {} | Data: {}", e, data_str);
                            }
                        }
                    }
                }
            }
            Err(e) => {
                return Err(format!("Claude stream error: {}", e));
            }
        }
    }

    // Calculate total time
    let total_time_us = request_start_time.elapsed().as_micros() as u64;

    // Log unified TTFT metrics
    log_ttft_metrics(ttft_us, total_time_us);

    info!("✅ Claude summary streaming completed: {} chars, TTFT: {:?}μs",
          accumulated_content.len(), ttft_us);

    Ok(SummaryResult {
        content: accumulated_content,
        ttft_us,
        total_time_us,
    })
}

/// Helper function to get provider name for logging
fn provider_name(provider: &LLMProvider) -> &str {
    match provider {
        LLMProvider::OpenAI => "OpenAI",
        LLMProvider::Claude => "Claude",
        LLMProvider::Groq => "Groq",
        LLMProvider::Ollama => "Ollama",
        LLMProvider::OpenRouter => "OpenRouter",
        LLMProvider::OpenAICompatible => "OpenAI Compatible",
    }
}
