/// Chat module - handles real-time chat with LLM about meeting transcripts
///
/// This module provides:
/// - Commands for sending chat messages with streaming responses
/// - Commands for retrieving meeting context
/// - Commands for chat history persistence
/// - Integration with existing LLM providers
/// - System prompt construction and message processing

pub mod commands;
pub mod processor;

// Re-export Tauri commands for use in lib.rs
pub use commands::{
    chat_clear_history, chat_get_history, chat_get_meeting_context, chat_save_message,
    chat_send_message, __cmd__chat_clear_history, __cmd__chat_get_history,
    __cmd__chat_get_meeting_context, __cmd__chat_save_message, __cmd__chat_send_message,
};
