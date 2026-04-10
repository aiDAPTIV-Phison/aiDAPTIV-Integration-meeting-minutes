# Design Spec

## _Feature Name_

LRC Import Timestamp Disorder Robustness Fix

---

## _Description_

### 功能目的

LRC（Lyric）格式的字幕檔案在實際場景中（尤其是 YouTube 自動生成字幕、第三方轉換工具匯出）常出現 timestamp 非時間順序的情況。目前系統在 Rust parser 中使用 `partial_cmp().unwrap()` 進行排序，當 f64 值為 NaN 時會直接 panic，導致 Tauri 後端 process 崩潰，WebView 顯示 `STATUS_BREAKPOINT` 錯誤頁。此功能修復的對象是所有使用 LRC Import 功能的使用者。

### 核心行為

1. Parser（`lrc/mod.rs`）在解析後對所有行依 `time_seconds` 安全排序（使用 `total_cmp`，NaN 安全）
2. Parser 對 `minutes * 60` 做溢位防護（`checked_mul`），避免 debug mode overflow panic
3. `commands.rs` 在計算 `duration = end - start` 後，clamp 到 `0.0` 以上，防止負數 duration
4. DB 查詢（`meeting.rs`）在取出 transcript 時加上 `ORDER BY audio_start_time ASC, rowid ASC`，確保回傳順序一致
5. 測試涵蓋亂序 timestamp 與重複 timestamp 兩種情境

### 關鍵狀態變化

- 原本 parser 遇到 NaN timestamp → panic → crash；修復後回傳 `Err` 並由前端顯示 toast 錯誤訊息
- 原本亂序 LRC 排序後 `audio_end_time < audio_start_time` 可能發生（重複 timestamp 情境）→ duration 為負數存入 DB；修復後 clamp 為 `0.0`
- DB 查詢原本無 ORDER BY，回傳順序由 SQLite 內部決定；修復後保證按 `audio_start_time` 升序回傳

### 輸入輸出

- **輸入**：`.lrc` 格式文字檔案，timestamp 可能亂序、重複、或含有超大 minutes 值
- **輸出**：
  - 成功：依時間順序排列的 `TranscriptSegment` 陣列，存入 DB 並回傳 meeting_id
  - 失敗（minutes overflow）：回傳 `Err("Timestamp minutes overflow: {minutes}")` 字串

### 效能目標

- Parser 排序：`total_cmp` 為 O(n log n)，與原本行為相同，無效能退步
- `checked_mul`：單次整數乘法，無可量測開銷
- 預期同時使用人數：1（桌面應用，單人操作）

### UI 流程

1. 使用者點選 Import LRC 按鈕 → 選擇 `.lrc` 檔案 → 點 "Import LRC File"
2. 後端解析並排序 → 若 overflow 錯誤，前端 `LRCImport.tsx` 的 catch block 顯示 toast 錯誤訊息
3. 匯入成功 → 跳轉至 meeting-details 頁面，transcript 按時間順序顯示

---

## _Use Cases_

### 正常流程

```
Given 使用者有一個 timestamp 亂序的 LRC 檔案
When 使用者匯入該檔案
Then 系統自動排序後存入 DB，meeting-details 頁面依時間順序顯示所有 transcript
And 每個 segment 的 duration 均 >= 0.0
```

```
Given 使用者有一個 timestamp 有序的正常 LRC 檔案
When 使用者匯入該檔案
Then 行為與修復前相同，正常顯示
```

### 取消流程

N/A — LRC import 為單次同步操作，沒有中途取消機制。

### 錯誤情境

**輸入錯誤**

```
Given LRC 檔案中有 minutes 值極大（如 [99999:00.00]）導致 u32 overflow
When parser 計算 minutes * 60
Then 回傳 Err("Timestamp minutes overflow: 99999")
And 前端顯示 toast 錯誤：Failed to import LRC file
```

```
Given LRC 檔案完全沒有有效 timestamp 行
When parser 解析
Then 回傳 Err("No valid LRC lines found in file")
And 前端顯示 toast 錯誤
```

**重複 timestamp**

```
Given LRC 中有兩行使用相同的 timestamp（如 [00:10.00]）
When parser 排序並計算 duration
Then duration = 0.0（clamp 後）
And 兩行均正常存入 DB，顯示順序以 rowid 為 tiebreaker
```

**資源不足**

N/A — parser 為純記憶體運算，LRC 匯入已有 token limit 防護擋住過大檔案。

**錯誤注入**

N/A — 此為桌面本地應用，無網路依賴，不適用外部服務中斷情境。

**敏感詞 / 政治用語 / 非法格式**

- LRC 內容為 transcript 文字，由使用者自行負責；系統不做內容過濾
- 非法 LRC 格式（無法匹配 timestamp regex 的行）會被靜默跳過，不造成 crash

### 錯誤碼

| 錯誤情境 | 錯誤訊息 |
|---------|---------|
| minutes overflow | `Timestamp minutes overflow: {minutes}` |
| 無有效 LRC 行 | `No valid LRC lines found in file` |
| DB 寫入失敗 | `Failed to save LRC import: {sqlx error}` |
| token 超限 | `LRC file too large: ~{N} tokens (model limit: {M})` |

針對敏感詞、政治用語：N/A（不做內容過濾）

### 權限情境

- 無需特殊權限；LRC import 對所有使用者開放
- 沒有 auth token 機制；為本地桌面應用
- N/A — 無登入/登出、token 過期情境

### 環境情境

- 支援 Windows（主要）、macOS、Linux（Tauri 跨平台）
- 最低硬體：與 Meetily 基本需求相同（無額外需求）
- 無瀏覽器/裝置相關限制（Tauri desktop app）

**並行使用**

- 不支援多人同時匯入（桌面單人應用），無 race condition 問題

**連續使用**

- parser 每次 import 都建立新的 `Vec<LrcLine>`，不保留狀態，無記憶體洩漏風險

**負載情境**

- 單筆 LRC 匯入，無高並發情境
- token limit 防護已限制單筆最大處理量

**安全性問題**

- LRC 內容不會執行，無 injection 風險
- 本地 SQLite，無 SQL injection 風險（使用 parameterized query）

### UI 情境

- **操作路徑**：主頁 → Import 按鈕 → Modal → 選檔 → Import LRC File 按鈕
- **Loading 狀態**：按鈕顯示 spinner + "Importing..."，按鈕 disabled
- **錯誤訊息顯示**：`toast.error("Failed to import LRC file", { description: error.message })`
- **空狀態**：N/A（modal 需先選擇檔案才能 submit）
- **RWD 行為**：Modal 固定 max-w-md，手機螢幕正常顯示

### 離線模式

完整支援離線操作——LRC import 為純本地操作，無網路依賴。

### 安裝升級

N/A — 此為邏輯修復，不涉及 DB schema 變更，無需 migration。

---

## _Limits_

### 功能限制

- 一次只能匯入一個 LRC 檔案
- LRC 行數上限受 token limit 間接限制（依當前 model context window）
- `minutes` 欄位上限：`u32::MAX / 60 ≈ 71,582,788` 分鐘（實際上 checked_mul 在此值以上才 error）

### 資料限制

- **數量限制**：單檔行數無硬性上限，受 token limit 間接限制
- **大小限制**：無明確檔案大小限制（受 token limit 保護）
- **型態限制**：僅接受 `.lrc` 副檔名
- **格式限制**：timestamp 格式需符合 `[mm:ss.xx]`、`[mm:ss]` 或 `[mmm:ss.xx]`；不符合的行靜默跳過

### 資源限制

- DB 存滿：由 SQLite 回傳錯誤，前端顯示 `Failed to save LRC import`
- 記憶體：LRC 檔案完整載入記憶體後解析，token limit 防護已限制最大輸入

### 取消限制

- 無取消機制；import 為同步操作，不支援中途取消

### 權限限制

- 無角色權限模型（本地單人桌面應用）
- 無認證機制，無 token 過期情境

### 環境限制

- 不支援瀏覽器（Tauri desktop only）
- 最低 macOS：13+（受 Meetily 整體限制，非此功能特有）
- 不支援同時多個 import 操作

### 不支援情境

- 不支援同時匯入多個 LRC 檔案
- 不支援中途取消匯入
- 不支援跨裝置同步
- 不支援修改已匯入的 LRC 資料（匯入後視為一般 meeting，走正常編輯流程）

### UI 限制

- 不支援拖曳多個檔案同時匯入（只取第一個）
- Modal 不支援鍵盤快捷鍵關閉（需點 X 按鈕）

### 安裝升級限制

- N/A — 純邏輯修復，無 DB schema 變更，支援直接升級，無需資料 migration
