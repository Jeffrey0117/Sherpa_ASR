# Sherpa_ASR

**離線中文語音辨識伺服器 — 零雲端、零 API key、開箱即用**

基於 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) Paraformer 模型，加上多層優化，讓語音辨識又快又準。

```python
from sherpa_server import SherpaServer

server = SherpaServer()
server.initialize()
result = server.transcribe_audio('meeting.wav')
print(result['text'])
# "我跟你說啊，這個東西真的很厲害，你不信的話，你自己試試看。"
```

## 為什麼選擇 Sherpa_ASR？

### 優化堆疊

| 優化 | 效果 | 原理 |
|------|------|------|
| **2x 音檔加速** | 處理時間砍半 | 每隔一個 sample 取一個 |
| **Silero VAD** | 跳過靜音段 | 30 秒音檔可能只有 15 秒有語音 |
| **int8 量化模型** | 比 fp32 快一倍、體積小一半 | Paraformer int8 ONNX |
| **動態執行緒** | 善用多核 CPU | `min(cpu_count, 8)` |
| **音頻預處理** | 提升辨識率 | 音量正規化 + 降噪 |
| **規則式標點** | 即時、零依賴 | 來自 [biaodian](https://github.com/Jeffrey0117/biaodian) |
| **熱詞** | 專有名詞辨識率提升 | 自訂辭彙權重加成 |

### 與其他方案的比較

| | Sherpa_ASR | Whisper | 雲端 API |
|---|---|---|---|
| 延遲 | 即時（本地） | 慢（大模型） | 看網路 |
| 隱私 | 完全離線 | 可離線 | 資料上傳雲端 |
| 成本 | 免費 | 免費 | 按量計費 |
| 中文優化 | Paraformer（專為中文） | 通用模型 | 依平台 |
| 部署大小 | ~200MB | ~1.5GB | N/A |
| 標點 | 內建規則式 | 無 | 依平台 |

## 安裝

```bash
pip install -r requirements.txt
```

依賴只有兩個：`sherpa-onnx` 和 `numpy`。

### 下載模型

```bash
# Paraformer 中文模型（~200MB）
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
tar xjf sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2 -C models/

# Silero VAD 模型（~2MB）
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx -P models/
```

## 快速開始

### 啟動伺服器

```bash
python sherpa_server.py
```

可選參數：

```bash
python sherpa_server.py --speed 2        # 音檔加速倍率（預設 2）
python sherpa_server.py --model-dir ./my-model  # 自訂模型路徑
```

### 通訊協議

透過 stdin/stdout JSON lines 溝通。任何語言都能 spawn 這個 process。

**辨識音檔：**
```json
{"action": "transcribe", "audio_path": "/path/to/audio.wav"}
```

回應：
```json
{
  "success": true,
  "text": "辨識結果，帶標點。",
  "raw_text": "辨識結果帶標點",
  "duration": 3.5,
  "rtf": 0.15,
  "speed_factor": 2
}
```

**熱詞管理：**
```json
{"action": "set_hotwords", "config": {"words": ["克 勞 德", "你 的 品 牌", "ClaudeBot"]}}
{"action": "get_hotwords"}
```

**其他指令：**
```json
{"action": "status"}
{"action": "stats"}
{"action": "exit"}
```

## 熱詞系統

在專案根目錄建立 `hotwords.txt`：

```
# 每行一個詞（中文用空格分隔字元）
克 勞 德
雲 管 家
ClaudeBot
CloudPipe
```

熱詞會提升這些詞的辨識權重，適合加入：
- 專案名稱、產品名稱
- 領域術語
- 容易被辨識錯誤的詞

內建已有常見開發術語（異步、組件、部署、編譯等）。

## 整合範例

### Node.js

```javascript
const { spawn } = require('child_process')
const proc = spawn('python', ['sherpa_server.py'])

// 送指令
proc.stdin.write(JSON.stringify({
  action: 'transcribe',
  audio_path: 'test.wav'
}) + '\n')

// 讀結果
proc.stdout.on('data', (data) => {
  const result = JSON.parse(data.toString())
  console.log(result.text)
})
```

### Python（直接呼叫）

```python
from sherpa_server import SherpaServer

server = SherpaServer()
server.initialize()

result = server.transcribe_audio('test.wav')
print(result['text'])  # 帶標點的辨識結果

server.set_hotwords({'words': ['你 的 品 牌']})
```

## 檔案結構

```
Sherpa_ASR/
  sherpa_server.py    — 核心伺服器（辨識 + VAD + 加速 + 熱詞）
  punctuation.py      — 規則式中文標點（來自 biaodian）
  requirements.txt    — Python 依賴
  models/             — 模型檔案（需另外下載）
  hotwords.txt        — 自訂熱詞（選用）
```

## 應用場景

- **Telegram Bot 語音輸入** — 對著手機講話，辨識後送 AI 處理
- **會議逐字稿** — 離線辨識會議錄音，自動加標點
- **字幕生成** — 影片音軌轉文字
- **語音日記** — 語音轉文字後存檔

## 誰在用

| 專案 | 說明 |
|------|------|
| [**ClaudeBot**](https://github.com/Jeffrey0117/ClaudeBot) | Telegram AI 指揮中心，語音辨識後送 Claude/Gemini 處理 |

## 相關專案

| 專案 | 關係 |
|------|------|
| [**biaodian**](https://github.com/Jeffrey0117/biaodian) | 標點模組的上游，`punctuation.py` 源自這裡 |
| [**Hanzi_ASR**](https://github.com/Jeffrey0117/Hanzi_ASR) | 另一套中文 ASR，不同模型架構 |
| [**ClaudeBot**](https://github.com/Jeffrey0117/ClaudeBot) | 主要使用者，自動偵測並載入 Sherpa_ASR |

## 授權

MIT License
