# Sherpa_ASR

Offline Chinese speech recognition server. Zero cloud, zero API keys.

Built on [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) Paraformer with optimizations:

- **2x audio speedup** — halves processing time by dropping every other sample
- **Silero VAD** — skips silence segments, saves compute
- **Audio preprocessing** — volume normalization + denoising
- **Rule-based Chinese punctuation** — zero ML dependency
- **Hotwords** — boost recognition of custom vocabulary
- **int8 quantized model** — fast and lightweight

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download model (Paraformer zh)
# Place in models/sherpa-onnx-paraformer-zh-2023-09-14/
# Also place silero_vad.onnx in models/

# Run server
python sherpa_server.py
```

## Protocol

Communicates via stdin/stdout JSON lines. Any language can spawn this as a subprocess.

### Transcribe

```json
{"action": "transcribe", "audio_path": "/path/to/audio.wav"}
```

Response:
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

### Hotwords

```json
{"action": "set_hotwords", "config": {"words": ["Claude Bot", "你 的 專 案 名"]}}
{"action": "get_hotwords"}
```

### Other

```json
{"action": "status"}
{"action": "stats"}
{"action": "exit"}
```

## Hotwords

Create `hotwords.txt` in the project root:

```
# One word per line (space-separated characters for Chinese)
克 勞 德
你 的 品 牌
English Words
```

## Integration

### Node.js (ClaudeBot style)

```javascript
const { spawn } = require('child_process')
const proc = spawn('python', ['sherpa_server.py'])

proc.stdin.write(JSON.stringify({ action: 'transcribe', audio_path: 'test.wav' }) + '\n')

proc.stdout.on('data', (data) => {
  const result = JSON.parse(data.toString())
  console.log(result.text)
})
```

### Python

```python
from sherpa_server import SherpaServer

server = SherpaServer()
server.initialize()
result = server.transcribe_audio('test.wav')
print(result['text'])
```

## File Structure

```
Sherpa_ASR/
  sherpa_server.py    — ASR server (VAD + 2x speed + hotwords)
  punctuation.py      — Rule-based Chinese punctuation
  requirements.txt    — Python dependencies
  models/             — Model files (download separately)
  hotwords.txt        — Custom vocabulary (optional)
```

## Related

- [ClaudeBot](https://github.com/Jeffrey0117/ClaudeBot) — Telegram command center for Claude Code
- [Hanzi_ASR](https://github.com/Jeffrey0117/Hanzi_ASR) — Chinese ASR system (different model architecture)
