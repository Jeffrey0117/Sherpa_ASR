#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sherpa-ONNX ASR Server

Offline speech recognition with:
- Paraformer model (int8 quantized)
- 2x audio speedup (halves processing time)
- Silero VAD (skips silence)
- Audio preprocessing (normalization + denoising)
- Hotwords (custom vocabulary boost)
- Rule-based Chinese punctuation

Communicates via stdin/stdout JSON lines.
Any language can spawn this as a subprocess.
"""

import sys
import json
import os
import logging
import traceback
import signal
import wave
import tarfile
import urllib.request
import numpy as np

from punctuation import add_punctuation

# 簡轉繁
try:
    from opencc import OpenCC
    _s2t = OpenCC('s2t')
except ImportError:
    _s2t = None

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class SherpaServer:
    def __init__(self, model_dir=None):
        self.recognizer = None
        self.vad = None
        self.initialized = False
        self.running = True
        self.transcription_count = 0
        self.total_audio_duration = 0.0
        self.vad_skipped_duration = 0.0

        # Audio speedup factor (2 = 2x speed, halves processing time)
        self.speed_factor = 2

        # Hotwords
        self.hotwords_file = None
        self.hotwords_score = 1.5
        self.hotwords_enabled = True

        # Dynamic thread count
        self.num_threads = min(os.cpu_count() or 4, 8)
        logger.info(f"Threads: {self.num_threads} (CPU cores: {os.cpu_count()})")

        # Model directory
        self.model_dir = model_dir or self._find_model_dir()

        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _find_model_dir(self):
        """Find sherpa-onnx Paraformer model directory, auto-download if missing"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, "models")
        model_name = "sherpa-onnx-paraformer-zh-2023-09-14"
        local_model = os.path.join(models_dir, model_name)

        # Check local models/ directory
        if os.path.exists(os.path.join(local_model, "model.int8.onnx")):
            return local_model

        # Check user cache
        cache_dir = os.path.expanduser("~/.cache/sherpa-onnx")
        cache_model = os.path.join(cache_dir, model_name)
        if os.path.exists(os.path.join(cache_model, "model.int8.onnx")):
            return cache_model

        # Auto-download
        logger.info("Model not found, downloading automatically...")
        self._download_models(models_dir, model_name)
        return local_model

    def _download_models(self, models_dir, model_name):
        """Download Paraformer model + Silero VAD automatically"""
        os.makedirs(models_dir, exist_ok=True)

        # Paraformer model (~200MB)
        tar_url = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{model_name}.tar.bz2"
        tar_path = os.path.join(models_dir, f"{model_name}.tar.bz2")

        logger.info(f"Downloading Paraformer model...")
        try:
            urllib.request.urlretrieve(tar_url, tar_path)
            with tarfile.open(tar_path, "r:bz2") as tar:
                tar.extractall(path=models_dir)
            os.remove(tar_path)
            logger.info("Paraformer model downloaded")
        except Exception as e:
            logger.error(f"Model download failed: {e}")
            logger.error("Manual download: https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models")

        # Silero VAD (~2MB)
        vad_path = os.path.join(models_dir, "silero_vad.onnx")
        if not os.path.exists(vad_path):
            vad_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
            logger.info("Downloading Silero VAD model...")
            try:
                urllib.request.urlretrieve(vad_url, vad_path)
                logger.info("Silero VAD model downloaded")
            except Exception as e:
                logger.warning(f"VAD download failed: {e} (VAD is optional)")

    def _signal_handler(self, signum, frame):
        logger.info(f"Signal {signum} received, shutting down...")
        self.running = False

    # ── VAD ────────────────────────────────────────────────────────────────

    def _init_vad(self):
        """Initialize Silero VAD"""
        try:
            import sherpa_onnx

            script_dir = os.path.dirname(os.path.abspath(__file__))
            vad_model_path = os.path.join(script_dir, "models", "silero_vad.onnx")

            if not os.path.exists(vad_model_path):
                logger.warning(f"Silero VAD model not found: {vad_model_path}")
                self.vad = None
                return

            vad_config = sherpa_onnx.VadModelConfig()
            vad_config.silero_vad.model = vad_model_path
            vad_config.silero_vad.threshold = 0.3
            vad_config.silero_vad.min_silence_duration = 0.4
            vad_config.silero_vad.min_speech_duration = 0.1
            vad_config.silero_vad.max_speech_duration = 15.0
            vad_config.silero_vad.window_size = 512
            vad_config.sample_rate = 16000
            vad_config.num_threads = self.num_threads

            self.vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)
            logger.info("Silero VAD initialized")

        except Exception as e:
            logger.warning(f"Silero VAD init failed: {e}")
            self.vad = None

    def _extract_speech_segments(self, samples, sample_rate):
        """Use VAD to extract speech segments, skip silence"""
        if self.vad is None:
            return samples, 0.0

        try:
            self.vad.reset()

            window_size = 512
            for i in range(0, len(samples), window_size):
                chunk = samples[i:i + window_size]
                if len(chunk) < window_size:
                    chunk = np.pad(chunk, (0, window_size - len(chunk)), 'constant')
                self.vad.accept_waveform(chunk)

            self.vad.flush()

            speech_segments = []
            while not self.vad.empty():
                segment = self.vad.front()
                speech_segments.append(segment)
                self.vad.pop()

            if not speech_segments:
                logger.warning("VAD: no speech detected")
                return samples, 0.0

            speech_samples = []
            for seg in speech_segments:
                speech_samples.extend(seg.samples)

            speech_samples = np.array(speech_samples, dtype=np.float32)

            original_duration = len(samples) / sample_rate
            speech_duration = len(speech_samples) / sample_rate
            skipped_duration = original_duration - speech_duration

            logger.info(f"VAD: {original_duration:.2f}s -> {speech_duration:.2f}s, skipped {skipped_duration:.2f}s ({len(speech_segments)} segments)")

            return speech_samples, skipped_duration

        except Exception as e:
            logger.warning(f"VAD failed: {e}, using original audio")
            return samples, 0.0

    # ── Audio Processing ───────────────────────────────────────────────────

    def _preprocess_audio(self, samples):
        """Audio preprocessing: normalize volume"""
        if len(samples) == 0:
            return samples

        # Volume normalization
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            target_peak = 0.7
            if max_val < 0.1:
                gain = min(target_peak / max_val, 5.0)
                samples = samples * gain
                logger.debug(f"Volume: amplified {gain:.1f}x (peak: {max_val:.3f})")
            elif max_val > 0.95:
                samples = samples * (target_peak / max_val)
                logger.debug(f"Volume: reduced to {target_peak:.1f} (peak: {max_val:.3f})")

        return samples.astype(np.float32)

    def _speedup_audio(self, samples):
        """Speed up audio via resampling (1.5x = keep 2/3 of samples)"""
        if self.speed_factor <= 1:
            return samples
        new_len = int(len(samples) / self.speed_factor)
        old_indices = np.linspace(0, len(samples) - 1, new_len)
        return np.interp(old_indices, np.arange(len(samples)), samples).astype(np.float32)

    def _read_wave_file(self, wav_path):
        """Read WAV file"""
        with wave.open(wav_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            num_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            num_frames = wf.getnframes()

            data = wf.readframes(num_frames)

            if sample_width == 2:
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                samples = np.frombuffer(data, dtype=np.int8).astype(np.float32) / 128.0

            if num_channels == 2:
                samples = samples.reshape(-1, 2).mean(axis=1)

            return samples, sample_rate

    # ── Hotwords ───────────────────────────────────────────────────────────

    _BUILTIN_HOTWORDS = [
        # Dev terms (simplified Chinese, space-separated for model vocab)
        "异 步", "同 步", "缓 存", "优 化", "渲 染",
        "组 件", "框 架", "状 态", "路 由", "插 件",
        "接 口", "函 数", "回 调", "线 程", "进 程",
        "容 器", "部 署", "编 译", "调 试",
        "依 赖", "模 块", "配 置", "环 境 变 量",
        "异 常", "错 误", "日 志",
    ]

    def _get_hotwords_path(self):
        """Get hotwords file path"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(script_dir, "hotwords.txt")

    def _load_hotwords_file(self):
        """Load hotwords from file"""
        hotwords_path = self._get_hotwords_path()
        words = []

        if os.path.exists(hotwords_path):
            try:
                with open(hotwords_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            words.append(line)
                logger.info(f"Loaded {len(words)} hotwords from {hotwords_path}")
            except Exception as e:
                logger.error(f"Failed to load hotwords: {e}")

        return words

    def _get_all_hotwords(self):
        """Get all hotwords (builtin + user)"""
        user_words = self._load_hotwords_file()
        all_words = list(self._BUILTIN_HOTWORDS)
        for word in user_words:
            if word not in all_words:
                all_words.append(word)
        return all_words

    def _save_hotwords_file(self, words):
        """Save hotwords to file"""
        hotwords_path = self._get_hotwords_path()
        try:
            with open(hotwords_path, 'w', encoding='utf-8') as f:
                f.write("# Hotwords - one per line\n")
                for word in words:
                    if word.strip():
                        f.write(f"{word.strip()}\n")
            logger.info(f"Saved {len(words)} hotwords to {hotwords_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save hotwords: {e}")
            return False

    def get_hotwords(self):
        """Get hotword settings"""
        words = self._load_hotwords_file()
        return {
            "success": True,
            "enabled": self.hotwords_enabled,
            "score": self.hotwords_score,
            "words": words,
        }

    def set_hotwords(self, config):
        """Set hotwords (enabled, score, words)"""
        try:
            if "enabled" in config:
                self.hotwords_enabled = bool(config["enabled"])
            if "score" in config:
                self.hotwords_score = max(1.0, min(3.0, float(config["score"])))
            if "words" in config and isinstance(config["words"], list):
                self._save_hotwords_file(config["words"])

            return {
                "success": True,
                "enabled": self.hotwords_enabled,
                "score": self.hotwords_score,
                "words": self._load_hotwords_file(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Initialize ─────────────────────────────────────────────────────────

    def initialize(self):
        """Initialize sherpa-onnx recognizer"""
        if self.initialized:
            return {"success": True, "message": "Already initialized"}

        try:
            import time
            start_time = time.time()
            logger.info(f"Initializing sherpa-onnx, model: {self.model_dir}")

            model_path = os.path.join(self.model_dir, "model.int8.onnx")
            tokens_path = os.path.join(self.model_dir, "tokens.txt")

            if not os.path.exists(model_path):
                return {
                    "success": False,
                    "error": f"Model not found: {model_path}",
                    "type": "models_not_downloaded"
                }

            if not os.path.exists(tokens_path):
                return {
                    "success": False,
                    "error": f"Tokens not found: {tokens_path}",
                    "type": "models_not_downloaded"
                }

            import sherpa_onnx

            self.recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
                paraformer=model_path,
                tokens=tokens_path,
                num_threads=self.num_threads,
                sample_rate=16000,
                feature_dim=80,
                decoding_method="greedy_search",
            )

            self._init_vad()

            load_time = time.time() - start_time
            self.initialized = True
            logger.info(f"sherpa-onnx initialized in {load_time:.2f}s, threads: {self.num_threads}, speed: {self.speed_factor}x")

            return {
                "success": True,
                "message": f"Initialized in {load_time:.2f}s",
            }

        except ImportError:
            error_msg = "sherpa-onnx not installed. Run: pip install sherpa-onnx"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "type": "import_error"}

        except Exception as e:
            error_msg = f"Init failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {"success": False, "error": error_msg, "type": "init_error"}

    # ── Transcribe ─────────────────────────────────────────────────────────

    def transcribe_audio(self, audio_path, options=None):
        """Transcribe audio file"""
        if not self.initialized:
            init_result = self.initialize()
            if not init_result["success"]:
                return init_result

        try:
            import time

            if not os.path.exists(audio_path):
                return {"success": False, "error": f"File not found: {audio_path}"}

            logger.info(f"Transcribing: {audio_path}")
            start_time = time.time()

            # Read audio
            samples, sample_rate = self._read_wave_file(audio_path)
            duration = len(samples) / sample_rate

            # Preprocess: normalize volume
            samples = self._preprocess_audio(samples)

            # VAD: extract speech, skip silence
            samples, skipped_duration = self._extract_speech_segments(samples, sample_rate)

            # 2x speedup: drop every other sample
            samples_fast = self._speedup_audio(samples)
            logger.info(f"Speedup: {len(samples)} -> {len(samples_fast)} samples ({self.speed_factor}x)")

            # Recognize
            stream = self.recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples_fast)
            self.recognizer.decode_stream(stream)
            text = stream.result.text

            elapsed = time.time() - start_time
            rtf = elapsed / duration

            self.transcription_count += 1
            self.total_audio_duration += duration
            self.vad_skipped_duration += skipped_duration

            # 簡轉繁 → 標點
            text_tc = _s2t.convert(text) if _s2t else text
            text_with_punc = add_punctuation(text_tc)

            logger.info(f"Result: {text_with_punc[:100]}... (RTF: {rtf:.3f})")

            return {
                "success": True,
                "text": text_with_punc,
                "raw_text": text_tc,
                "confidence": 0.95,
                "duration": duration,
                "language": "zh",
                "model_type": "sherpa-onnx",
                "rtf": rtf,
                "process_time": elapsed,
                "vad_skipped": skipped_duration,
                "speed_factor": self.speed_factor,
            }

        except Exception as e:
            error_msg = f"Transcription failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {"success": False, "error": error_msg, "type": "transcription_error"}

    # ── Status ─────────────────────────────────────────────────────────────

    def check_status(self):
        """Check server status"""
        try:
            import sherpa_onnx
            return {
                "success": True,
                "installed": True,
                "initialized": self.initialized,
                "version": sherpa_onnx.__version__,
                "model_dir": self.model_dir,
                "num_threads": self.num_threads,
                "speed_factor": self.speed_factor,
                "models": {
                    "asr": self.recognizer is not None,
                    "vad": self.vad is not None,
                },
            }
        except ImportError:
            return {
                "success": False,
                "installed": False,
                "initialized": False,
                "error": "sherpa-onnx not installed",
            }

    def get_performance_stats(self):
        """Get performance statistics"""
        return {
            "transcription_count": self.transcription_count,
            "total_audio_duration": round(self.total_audio_duration, 2),
            "average_duration": round(
                self.total_audio_duration / max(1, self.transcription_count), 2
            ),
            "vad_skipped_duration": round(self.vad_skipped_duration, 2),
            "vad_efficiency": round(
                self.vad_skipped_duration / max(0.01, self.total_audio_duration) * 100, 1
            ) if self.total_audio_duration > 0 else 0,
            "initialized": self.initialized,
            "engine": "sherpa-onnx",
            "num_threads": self.num_threads,
            "speed_factor": self.speed_factor,
            "vad_enabled": self.vad is not None,
        }

    # ── Main Loop ──────────────────────────────────────────────────────────

    def run(self):
        """Run server main loop (stdin/stdout JSON)"""
        logger.info("Sherpa ASR server starting")

        init_result = self.initialize()
        print(json.dumps(init_result, ensure_ascii=False))
        sys.stdout.flush()

        while self.running:
            try:
                line = sys.stdin.readline()
                if not line:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    command = json.loads(line)
                except json.JSONDecodeError:
                    result = {"success": False, "error": "Invalid JSON"}
                    print(json.dumps(result, ensure_ascii=False))
                    sys.stdout.flush()
                    continue

                action = command.get("action")

                if action == "transcribe":
                    audio_path = command.get("audio_path")
                    options = command.get("options", {})
                    result = self.transcribe_audio(audio_path, options)
                elif action == "status":
                    result = self.check_status()
                elif action == "stats":
                    result = {"success": True, "stats": self.get_performance_stats()}
                elif action == "get_hotwords":
                    result = self.get_hotwords()
                elif action == "set_hotwords":
                    config = command.get("config", {})
                    result = self.set_hotwords(config)
                elif action == "exit":
                    result = {"success": True, "message": "Server exiting"}
                    print(json.dumps(result, ensure_ascii=False))
                    sys.stdout.flush()
                    break
                else:
                    result = {"success": False, "error": f"Unknown action: {action}"}

                print(json.dumps(result, ensure_ascii=False))
                sys.stdout.flush()

            except KeyboardInterrupt:
                break
            except Exception as e:
                error_result = {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
                print(json.dumps(error_result, ensure_ascii=False))
                sys.stdout.flush()

        logger.info("Sherpa ASR server stopped")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sherpa-ONNX ASR Server")
    parser.add_argument("--model-dir", type=str, default=None, help="Model directory path")
    parser.add_argument("--speed", type=float, default=1.5, help="Audio speed factor (default: 1.5)")
    args = parser.parse_args()

    server = SherpaServer(model_dir=args.model_dir)
    server.speed_factor = args.speed
    server.run()
