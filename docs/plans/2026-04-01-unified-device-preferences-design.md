# Unified Device Preferences Design

## Problem

Audio device settings (Microphone, System Audio, Recording Mode) could be configured from two places:

1. **Home page "Device" popup** (top bar) — stored in `localStorage`
2. **Settings page "Recordings" tab** — stored via Rust backend (but not actually persisted)

This caused three conflicts:
- **Dual storage**: `localStorage` vs Rust backend, with unclear priority
- **`recordingMode` lost on override**: Backend prefs didn't include `recordingMode`, causing it to reset
- **No actual persistence**: Rust `save_recording_preferences` only logged, never wrote to disk

## Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Storage location | Rust backend only (`tauri-plugin-store`) | Single source of truth, persists across app restarts |
| Sync mechanism | Both UIs read/write same Rust backend | They're on separate Next.js routes; remounting reloads latest values |
| Scope | Audio Device + Recording Mode only | Language and Model don't have dual-entry conflicts |
| Settings page behavior | Keep full DeviceSelection UI | Users can configure from both places equally |

## Changes

### Rust Backend (`recording_preferences.rs`)

- Added `preferred_mic_device`, `preferred_system_device`, `recording_mode` fields to `RecordingPreferences` struct
- Implemented actual persistence using `tauri_plugin_store::StoreExt` (store file: `recording_preferences.json`)
- `load_recording_preferences`: reads from store, falls back to defaults
- `save_recording_preferences`: serializes to store and calls `store.save()`
- Backward compatible: `#[serde(default)]` handles missing fields in old data

### Frontend `page.tsx`

- Removed `loadSelectedDevicesFromStorage()` (localStorage reader)
- Removed `useEffect` that wrote `selectedDevices` to `localStorage`
- Kept existing `loadDevicePreferences` useEffect, added `recordingMode` field
- Device popup "Done" button now writes to Rust backend via `set_recording_preferences`

### Frontend `RecordingSettings.tsx`

- Added `recording_mode` to `RecordingPreferences` interface
- `handleDeviceChange` now includes `recordingMode` when saving
- `DeviceSelection` receives `recordingMode` from stored preferences

## Data Flow (After)

```
App Startup
  └─→ useEffect: invoke('get_recording_preferences')
       └─→ setSelectedDevices({ micDevice, systemDevice, recordingMode })

Home Page Device Popup "Done"
  └─→ invoke('set_recording_preferences', { ...prefs, mic, system, mode })
  └─→ Rust: store.set() → store.save() → disk

Settings Page Mount
  └─→ invoke('get_recording_preferences')
       └─→ setPreferences({ ...prefs including recording_mode })

Settings Page Device Change
  └─→ invoke('set_recording_preferences', { ...prefs, mic, system, mode })
  └─→ Rust: store.set() → store.save() → disk
```

## Files Modified

| File | Change |
|------|--------|
| `frontend/src-tauri/src/audio/recording_preferences.rs` | Add fields, implement store persistence |
| `frontend/src/app/page.tsx` | Remove localStorage, save to backend on Done |
| `frontend/src/components/RecordingSettings.tsx` | Add recordingMode to interface and DeviceSelection |
