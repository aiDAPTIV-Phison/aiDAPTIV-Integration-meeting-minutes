# LRC Import Token Validation Design

## Problem

Importing large LRC files (e.g., 5000+ lines, ~240KB) causes the Meetily app to crash with `STATUS_BREAKPOINT`. The crash occurs because:

1. All transcript segments are rendered simultaneously in `TranscriptView` using `<motion.div>` with no virtualization
2. ~5000 animated DOM nodes + ~10000 paragraph elements + tooltips exhaust the WebView2 renderer memory

Beyond the crash, importing LRC files that exceed the LLM's context window is pointless — the summary and chat features cannot process them effectively.

## Solution

Validate the LRC file's estimated token count against the current LLM model's context window **before** writing to the database. Reject files that exceed the limit with a clear error message.

## Design

### Validation Location

All validation happens in `api_import_lrc` (Rust backend, `frontend/src-tauri/src/lrc/commands.rs`). The frontend requires zero changes — existing error handling already displays the error via toast.

### Token Limit by Provider

| Provider        | Token Limit                          | Source                |
|-----------------|--------------------------------------|-----------------------|
| `ollama`        | Dynamic from model metadata - 300    | `METADATA_CACHE`      |
| `llamacpp`      | 16000                                | Fixed                 |
| `claude`        | 100000                               | Fixed                 |
| `groq`          | 100000                               | Fixed                 |
| `openrouter`    | 100000                               | Fixed                 |
| Not configured  | 4000                                 | Conservative fallback |

Ollama fallback (if metadata fetch fails): 4000 tokens.

### Validation Flow

```
api_import_lrc(file_content):
  1. parse_lrc(file_content)                    // existing
  2. Concatenate all line texts → total_text
  3. estimated_tokens = rough_token_count(total_text)  // reuse from summary::processor
  4. model_config = SettingsRepository::get_model_config(pool)
  5. token_limit = determine_limit(model_config)       // see table above
  6. IF estimated_tokens > token_limit → return Err(descriptive message)
  7. ELSE → proceed with DB write                      // existing
```

### Error Message

```
LRC file too large: ~{estimated_tokens} tokens (model limit: {token_limit}).
Please reduce the file size or switch to a model with a larger context window.
```

This message surfaces through the existing `LRCImport.tsx` toast error handling — no frontend changes needed.

### Helper Function

New `get_token_limit` async function in `lrc/commands.rs`:

```rust
async fn get_token_limit(pool: &SqlitePool) -> usize {
    match SettingsRepository::get_model_config(pool).await {
        Ok(Some(config)) => match config.provider.as_str() {
            "ollama" => {
                match METADATA_CACHE.get_or_fetch(
                    &config.model,
                    config.ollama_endpoint.as_deref()
                ).await {
                    Ok(metadata) => metadata.context_size.saturating_sub(300),
                    Err(_) => 4000,
                }
            }
            "llamacpp" => 16000,
            "claude" | "groq" | "openrouter" => 100000,
            _ => 4000,
        },
        _ => 4000,
    }
}
```

### Token Estimation

Reuse the existing `rough_token_count` from `summary::processor`:

```rust
pub fn rough_token_count(s: &str) -> usize {
    (s.chars().count() as f64 / 4.0).ceil() as usize
}
```

The estimation is based on all LRC line texts joined together (not including timestamps or metadata).

## Files Changed

| File | Change |
|------|--------|
| `frontend/src-tauri/src/lrc/commands.rs` | Add token validation logic (~30 lines) |

No new dependencies. No frontend changes. No new Tauri commands.

## Reused Components

- `summary::processor::rough_token_count` — token estimation
- `database::repositories::setting::SettingsRepository::get_model_config` — read current model config
- `ollama::metadata::METADATA_CACHE` (only for Ollama provider) — dynamic context size
- `LRCImport.tsx` existing error toast — displays the rejection message

## Testing

1. Import a small LRC file (<100 lines) → should succeed
2. Import a large LRC file (5000+ lines) with Ollama (small context) → should show error toast
3. Import a large LRC file with Claude provider → should succeed (100K limit)
4. Import with no model configured → should reject with 4000 token fallback limit
5. Import with llamacpp provider → should reject above 16000 tokens
