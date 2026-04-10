use super::parse_lrc;
use crate::api::TranscriptSegment;
use crate::database::repositories::{setting::SettingsRepository, transcript::TranscriptsRepository};
use crate::ollama::metadata::ModelMetadataCache;
use crate::state::AppState;
use crate::summary::rough_token_count;
use chrono::Utc;
use log::{error as log_error, info as log_info, warn as log_warn};
use once_cell::sync::Lazy;
use sqlx::SqlitePool;
use std::time::Duration;
use tauri::{AppHandle, Runtime, State};
use uuid::Uuid;

static METADATA_CACHE: Lazy<ModelMetadataCache> =
    Lazy::new(|| ModelMetadataCache::new(Duration::from_secs(300)));

/// Determine the token limit for LRC import based on the configured LLM provider.
async fn get_token_limit(pool: &SqlitePool) -> usize {
    match SettingsRepository::get_model_config(pool).await {
        Ok(Some(config)) => match config.provider.as_str() {
            "ollama" => {
                match METADATA_CACHE
                    .get_or_fetch(&config.model, config.ollama_endpoint.as_deref())
                    .await
                {
                    Ok(metadata) => {
                        let limit = metadata.context_size.saturating_sub(300);
                        log_info!(
                            "Ollama model '{}' context size: {} tokens (limit: {})",
                            config.model,
                            metadata.context_size,
                            limit
                        );
                        limit
                    }
                    Err(e) => {
                        log_warn!(
                            "Failed to fetch Ollama context size for '{}': {}. Using fallback 4000",
                            config.model,
                            e
                        );
                        4000
                    }
                }
            }
            "llamacpp" => 16000,
            "claude" | "groq" | "openrouter" => 100_000,
            other => {
                log_warn!("Unknown provider '{}', using fallback limit 4000", other);
                4000
            }
        },
        Ok(None) => {
            log_warn!("No model config found, using conservative token limit 4000");
            4000
        }
        Err(e) => {
            log_warn!("Failed to read model config: {}. Using fallback limit 4000", e);
            4000
        }
    }
}

/// Import LRC file and create a new meeting with transcripts
#[tauri::command]
pub async fn api_import_lrc<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, AppState>,
    file_content: String,
) -> Result<String, String> {
    log_info!("api_import_lrc called");

    // Parse LRC content
    let parse_result = parse_lrc(&file_content)?;
    log_info!(
        "Parsed LRC file: {} lines, title: {:?}",
        parse_result.lines.len(),
        parse_result.metadata.title
    );

    // Validate token count against the current model's context window
    let pool = state.db_manager.pool();
    let total_text: String = parse_result
        .lines
        .iter()
        .map(|l| l.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let estimated_tokens = rough_token_count(&total_text);
    let token_limit = get_token_limit(pool).await;

    log_info!(
        "LRC token estimate: {} tokens, model limit: {}",
        estimated_tokens,
        token_limit
    );

    if estimated_tokens > token_limit {
        return Err(format!(
            "LRC file too large: ~{} tokens (model limit: {}). \
             Please reduce the file size or switch to a model with a larger context window.",
            estimated_tokens, token_limit
        ));
    }

    // Generate meeting ID
    let meeting_id = format!("meeting-{}", Uuid::new_v4());

    // Use metadata title or default
    let meeting_title = parse_result
        .metadata
        .title
        .clone()
        .unwrap_or_else(|| format!("LRC Import {}", Utc::now().format("%Y-%m-%d %H:%M:%S")));

    // Convert LRC lines to TranscriptSegment
    let mut transcript_segments = Vec::new();
    let lines = &parse_result.lines;

    for (i, line) in lines.iter().enumerate() {
        let audio_start_time = line.time_seconds;

        // Calculate end time: use next line's start time, or add 5 seconds for last line
        let audio_end_time = if i + 1 < lines.len() {
            lines[i + 1].time_seconds
        } else {
            // Last line: add 5 seconds as default duration
            audio_start_time + 5.0
        };

        let duration = (audio_end_time - audio_start_time).max(0.0);

        // Generate wall-clock timestamp (use current time as base)
        let timestamp = Utc::now().format("%H:%M:%S").to_string();

        transcript_segments.push(TranscriptSegment {
            id: format!("lrc-{}", i),
            text: line.text.clone(),
            timestamp,
            audio_start_time: Some(audio_start_time),
            audio_end_time: Some(audio_end_time),
            duration: Some(duration),
        });
    }

    log_info!(
        "Created {} transcript segments for meeting {}",
        transcript_segments.len(),
        meeting_id
    );

    // Save to database using existing repository
    match TranscriptsRepository::save_transcript(
        pool,
        &meeting_title,
        &transcript_segments,
        None, // No folder path for LRC imports
    )
    .await
    {
        Ok(saved_meeting_id) => {
            log_info!(
                "Successfully imported LRC as meeting: {} ({})",
                meeting_title,
                saved_meeting_id
            );
            Ok(saved_meeting_id)
        }
        Err(e) => {
            log_error!("Failed to save LRC import to database: {}", e);
            Err(format!("Failed to save LRC import: {}", e))
        }
    }
}
