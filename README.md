# faster-whisper-node

A pure Node.js/Rust module for Whisper speech-to-text transcription. **No Python runtime required.**

Uses [CTranslate2](https://github.com/OpenNMT/CTranslate2) as the inference engine, the same battle-tested backend that powers [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

[![CI](https://github.com/manmal/faster-whisper-node/actions/workflows/ci.yml/badge.svg)](https://github.com/manmal/faster-whisper-node/actions/workflows/ci.yml)
[![npm version](https://img.shields.io/npm/v/faster-whisper-node.svg)](https://www.npmjs.com/package/faster-whisper-node)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Node.js Version](https://img.shields.io/node/v/faster-whisper-node.svg)](https://nodejs.org)

## Features

- **Structured output** — Get segments with timestamps, confidence scores, and metadata
- **Transcription options** — Beam size, temperature, language, and more
- **Multiple input formats** — File paths, Buffers, or raw samples
- **Stereo support** — Automatically converts stereo to mono
- **No Python** — Pure Node.js/Rust, zero Python dependencies

## Quick Start

```javascript
const { Engine, formatTimestamp } = require('faster-whisper-node');

// Load model
const engine = new Engine('./models/tiny');

// Simple transcription (returns text)
const text = engine.transcribe('./audio.wav');
console.log(text); // "Hello world."

// Get segments with timestamps
const result = engine.transcribeSegments('./audio.wav');
for (const segment of result.segments) {
  console.log(`[${formatTimestamp(segment.start)} -> ${formatTimestamp(segment.end)}] ${segment.text}`);
}
// [00:00.000 -> 00:02.500] Hello world.

// With options
const text2 = engine.transcribeWithOptions('./audio.wav', {
  language: 'en',
  beamSize: 5,
  temperature: 0.0
});
```

---

## Why This Exists

[faster-whisper](https://github.com/SYSTRAN/faster-whisper) is an excellent Python library that provides fast Whisper inference via CTranslate2. However, integrating Python into Node.js/Electron applications introduces significant complexity:

| Challenge | Python Approach | This Package |
|-----------|-----------------|--------------|
| **Runtime dependency** | Requires Python interpreter + pip packages | Single `.node` binary + shared library |
| **Distribution** | Bundle Python or require user installation | npm install (prebuilt binaries) |
| **Electron/Node integration** | Child process or Python bridge | Native N-API module |
| **Cold start time** | Python interpreter startup overhead | Native code, instant loading |
| **Cross-platform packaging** | Complex (pyinstaller, conda, etc.) | Standard npm workflow |

**This package provides the same CTranslate2 performance with zero Python dependencies.**

---

## Installation

```bash
npm install faster-whisper-node
```

### Platform Requirements

| Platform | Additional Requirements |
|----------|------------------------|
| **Linux** | None |
| **Windows** | [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| **macOS** | None |

---

## API Reference

### Engine

```typescript
import { Engine, TranscribeOptions, ModelOptions, TranscriptionResult, Segment } from 'faster-whisper-node';
```

#### Constructor

```typescript
// Simple
const engine = new Engine(modelPath: string);

// With options
const engine = Engine.withOptions(modelPath: string, options?: ModelOptions);
```

**ModelOptions:**
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `device` | `"cpu" \| "cuda"` | `"cpu"` | Computation device |
| `computeType` | `string` | `"default"` | `"default"`, `"auto"`, `"int8"`, `"int8_float16"`, `"float16"`, `"float32"` |
| `cpuThreads` | `number` | `0` | Number of CPU threads (0 = auto) |

#### Methods

##### `transcribe(audioPath: string): string`

Simple transcription, returns text only.

```typescript
const text = engine.transcribe('./audio.wav');
// "Hello world."
```

##### `transcribeSegments(audioPath: string, options?: TranscribeOptions): TranscriptionResult`

Full transcription with segments and metadata.

```typescript
const result = engine.transcribeSegments('./audio.wav', { language: 'en' });
console.log(result.text);      // "Hello world."
console.log(result.duration);  // 2.5
console.log(result.segments);  // [{ id: 0, start: 0, end: 2.5, text: "Hello world.", ... }]
```

##### `transcribeWithOptions(audioPath: string, options: TranscribeOptions): string`

Transcription with options, returns text only.

##### `transcribeBuffer(buffer: Buffer, options?: TranscribeOptions): TranscriptionResult`

Transcribe from a Buffer containing WAV data.

```typescript
const buffer = fs.readFileSync('./audio.wav');
const result = engine.transcribeBuffer(buffer);
```

##### `transcribeSamples(samples: number[], options?: TranscribeOptions): TranscriptionResult`

Transcribe from raw audio samples (must be 16kHz mono, normalized to [-1, 1]).

```typescript
const samples = [0.0, 0.1, -0.05, ...]; // Float32 samples
const result = engine.transcribeSamples(samples);
```

##### `samplingRate(): number`

Returns expected sampling rate (16000 Hz).

##### `isMultilingual(): boolean`

Returns true if model supports multiple languages.

##### `numLanguages(): number`

Returns number of supported languages.

#### TranscribeOptions

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `language` | `string` | auto-detect | Source language code (e.g., "en", "de", "fr") |
| `task` | `"transcribe" \| "translate"` | `"transcribe"` | Task to perform |
| `beamSize` | `number` | `5` | Beam size (1 = greedy search) |
| `patience` | `number` | `1.0` | Beam search patience |
| `lengthPenalty` | `number` | `1.0` | Length penalty |
| `repetitionPenalty` | `number` | `1.0` | Repetition penalty (>1 to penalize) |
| `noRepeatNgramSize` | `number` | `0` | Prevent ngram repetitions |
| `temperature` | `number` | `1.0` | Sampling temperature |
| `suppressBlank` | `boolean` | `true` | Suppress blank outputs |
| `maxLength` | `number` | `448` | Maximum generation length |
| `wordTimestamps` | `boolean` | `false` | Include word-level timestamps |

#### TranscriptionResult

```typescript
interface TranscriptionResult {
  segments: Segment[];
  language: string;
  languageProbability: number;
  duration: number;
  text: string;
}
```

#### Segment

```typescript
interface Segment {
  id: number;
  seek: number;
  start: number;  // seconds
  end: number;    // seconds
  text: string;
  tokens: number[];
  temperature: number;
  avgLogprob: number;
  compressionRatio: number;
  noSpeechProb: number;
}
```

### Utility Functions

#### `availableModels(): string[]`

Returns list of supported model size aliases.

```typescript
const models = availableModels();
// ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v1", "large-v2", "large-v3"]
```

#### `formatTimestamp(seconds: number, alwaysIncludeHours?: boolean): string`

Format seconds to timestamp string.

```typescript
formatTimestamp(65.5);       // "01:05.500"
formatTimestamp(65.5, true); // "00:01:05.500"
formatTimestamp(3661.5);     // "01:01:01.500"
```

---

## Models

Download models from Hugging Face:

```bash
mkdir -p models && cd models
git lfs install
git clone --depth 1 https://huggingface.co/Systran/faster-whisper-tiny tiny
```

**Available models:** `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large-v1`, `large-v2`, `large-v3`

---

## Audio Format

Audio must be WAV format:

| Property | Required |
|----------|----------|
| Sample rate | 16000 Hz |
| Channels | Mono or Stereo (auto-converted) |
| Bit depth | 8, 16, 24, or 32-bit PCM |

**Convert with ffmpeg:**

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -acodec pcm_s16le output.wav
```

---

## Platform Support

| Platform | Architecture | Status |
|----------|--------------|--------|
| Linux | x86_64, arm64 | ✅ |
| macOS | arm64 (Apple Silicon), x86_64 | ✅ |
| Windows | x86_64 | ✅ |

---

## Development

```bash
# Clone
git clone --recursive https://github.com/manmal/faster-whisper-node
cd faster-whisper-node

# Install dependencies (downloads prebuilt CTranslate2)
npm install

# Build
npm run build

# Test
npm test
```

---

## Credits

- **[SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)** — Original implementation and model conversions
- **[OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2)** — The inference engine
- **[ct2rs](https://github.com/jkawamoto/ctranslate2-rs)** — Rust bindings for CTranslate2

---

## License

MIT — See [`THIRD_PARTY_LICENSES.md`](./THIRD_PARTY_LICENSES.md) for third-party licenses.
