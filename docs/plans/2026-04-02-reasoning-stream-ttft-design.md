# Reasoning Stream Display & TTFT Fix Design

## Problem

When using a thinking/reasoning model (e.g. DeepSeek R1 distills) via **llama.cpp server** with `--reasoning-format deepseek --jinja`, the server streams `reasoning_content` in the SSE delta **before** `content`. The Meetily app only processes `delta.content`, causing:

1. **UI appears frozen** during the entire reasoning phase (no tokens emitted to frontend)
2. **TTFT is inaccurate** — measured from first `content` token, not first streamed output

## Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Reasoning field name | `reasoning_content` | Matches DeepSeek API / llama.cpp convention |
| TTFT definition | First non-empty `reasoning_content` OR `content` (whichever comes first) | Reflects actual time-to-first-useful-output |
| Token event | Extend existing `llm:chat:token` with `token_type` field | Avoids adding a separate event channel |
| UI display | Collapsible reasoning block above answer | Shows progress without overwhelming the answer |
| Persistence | Only save `content` to DB; reasoning is ephemeral | Keeps DB lean; reasoning is transient debug info |
| Summary path | Parse `reasoning_content` internally but only accumulate `content` for final summary | Summary output should be clean answer only |

## Changes

### Rust — `llm_client.rs`

#### 1. `StreamDelta` struct — add `reasoning_content`

```rust
pub struct StreamDelta {
    pub content: Option<String>,
    pub role: Option<String>,
    pub reasoning_content: Option<String>,  // NEW
}
```

#### 2. `StreamTokenPayload` — add `token_type`

```rust
pub struct StreamTokenPayload {
    pub request_id: String,
    pub content_delta: String,
    pub token_type: String,  // NEW: "reasoning" | "content"
}
```

#### 3. `stream_chat_openai_compatible` — emit reasoning tokens + fix TTFT

- Check `delta.reasoning_content` first; if non-empty, emit with `token_type: "reasoning"`
- Check `delta.content` second; if non-empty, emit with `token_type: "content"`
- TTFT: record on **first non-empty** from either field

#### 4. `generate_summary_streaming` — skip reasoning for summary

- Parse `reasoning_content` from chunks (so serde doesn't fail)
- Only accumulate `content` into `accumulated_content`
- TTFT: still record from first output of either field

### Frontend — `ChatPanel.tsx`

#### 1. `Message` interface — add `reasoning_content`

```typescript
interface Message {
  // ... existing fields
  reasoning_content?: string;  // NEW: accumulated reasoning text
}
```

#### 2. Event listener — handle `token_type`

- Maintain `streamingReasoningRef` alongside `streamingContentRef`
- On `llm:chat:token`: append to reasoning or content ref based on `token_type`
- Update message with both fields

#### 3. UI — collapsible reasoning block

- If `reasoning_content` exists, show a collapsible `<details>` block above the answer
- Label: "Thinking..." (during stream) / "Reasoning" (after complete)
- Content text styled in muted color, smaller font
- Auto-collapse when streaming completes

## Files Modified

| File | Change |
|------|--------|
| `frontend/src-tauri/src/summary/llm_client.rs` | StreamDelta, StreamTokenPayload, streaming logic |
| `frontend/src/components/MeetingDetails/ChatPanel.tsx` | Message type, event listener, UI rendering |

## Not Changed

- Server config (`--reasoning-format deepseek --jinja` stays)
- DB schema (no new columns)
- Claude streaming path (separate protocol, out of scope)
- Summary display (reasoning filtered out before display)
