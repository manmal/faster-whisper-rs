use napi_derive::napi;
use ct2rs::{Whisper, WhisperOptions, Config};
use ct2rs::sys::{Device, ComputeType};
use std::fs::File;
use std::io::Read;

/// Transcription segment with timing and confidence information
#[napi(object)]
#[derive(Clone, Debug)]
pub struct Segment {
    /// Segment ID (0-indexed)
    pub id: u32,
    /// Seek position in audio frames
    pub seek: u32,
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds
    pub end: f64,
    /// Transcribed text
    pub text: String,
    /// Token IDs
    pub tokens: Vec<u32>,
    /// Decoding temperature used
    pub temperature: f64,
    /// Average log probability
    pub avg_logprob: f64,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Probability of no speech
    pub no_speech_prob: f64,
}

/// Transcription options
#[napi(object)]
#[derive(Clone, Debug)]
pub struct TranscribeOptions {
    /// Source language (e.g., "en", "de", "fr"). If not set, language is auto-detected.
    pub language: Option<String>,
    /// Task to perform: "transcribe" or "translate"
    pub task: Option<String>,
    /// Beam size for beam search (default: 5, set to 1 for greedy search)
    pub beam_size: Option<u32>,
    /// Beam search patience factor (default: 1.0)
    pub patience: Option<f64>,
    /// Exponential penalty applied to length during beam search (default: 1.0)
    pub length_penalty: Option<f64>,
    /// Penalty for repetition (default: 1.0, set > 1 to penalize)
    pub repetition_penalty: Option<f64>,
    /// Prevent repetitions of ngrams with this size (default: 0, disabled)
    pub no_repeat_ngram_size: Option<u32>,
    /// Sampling temperature (default: 1.0)
    pub temperature: Option<f64>,
    /// Suppress blank outputs at beginning (default: true)
    pub suppress_blank: Option<bool>,
    /// Maximum generation length (default: 448)
    pub max_length: Option<u32>,
    /// Include timestamps in output (default: false)
    pub word_timestamps: Option<bool>,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            language: None,
            task: None,
            beam_size: None,
            patience: None,
            length_penalty: None,
            repetition_penalty: None,
            no_repeat_ngram_size: None,
            temperature: None,
            suppress_blank: None,
            max_length: None,
            word_timestamps: None,
        }
    }
}

/// Model configuration options
#[napi(object)]
#[derive(Clone, Debug)]
pub struct ModelOptions {
    /// Device to use: "cpu" or "cuda" (default: "cpu")
    pub device: Option<String>,
    /// Compute type: "default", "auto", "int8", "int8_float16", "int16", "float16", "float32"
    pub compute_type: Option<String>,
    /// Number of CPU threads per replica (0 for auto)
    pub cpu_threads: Option<u32>,
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            device: None,
            compute_type: None,
            cpu_threads: None,
        }
    }
}

/// Transcription result containing all segments and metadata
#[napi(object)]
#[derive(Clone, Debug)]
pub struct TranscriptionResult {
    /// All transcribed segments
    pub segments: Vec<Segment>,
    /// Detected or specified language
    pub language: String,
    /// Language detection probability (0 if language was specified)
    pub language_probability: f64,
    /// Total audio duration in seconds
    pub duration: f64,
    /// Full transcribed text (all segments joined)
    pub text: String,
}

#[napi]
pub struct Engine {
    model: Whisper,
    sampling_rate: u32,
}

#[napi]
impl Engine {
    /// Create a new transcription engine from a model path
    #[napi(constructor)]
    pub fn new(model_path: String) -> napi::Result<Self> {
        Self::with_options(model_path, None)
    }

    /// Create a new transcription engine with options
    #[napi(factory)]
    pub fn with_options(model_path: String, options: Option<ModelOptions>) -> napi::Result<Self> {
        let opts = options.unwrap_or_default();
        
        let device = match opts.device.as_deref() {
            Some("cuda") | Some("CUDA") => Device::CUDA,
            _ => Device::CPU,
        };
        
        let compute_type = match opts.compute_type.as_deref() {
            Some("auto") => ComputeType::AUTO,
            Some("int8") => ComputeType::INT8,
            Some("int8_float16") => ComputeType::INT8_FLOAT16,
            Some("int8_float32") => ComputeType::INT8_FLOAT32,
            Some("int16") => ComputeType::INT16,
            Some("float16") => ComputeType::FLOAT16,
            Some("float32") => ComputeType::FLOAT32,
            _ => ComputeType::DEFAULT,
        };
        
        let config = Config {
            device,
            compute_type,
            num_threads_per_replica: opts.cpu_threads.unwrap_or(0) as usize,
            ..Config::default()
        };
        
        let model = Whisper::new(&model_path, config)
            .map_err(|e| napi::Error::from_reason(format!("Failed to load model: {}", e)))?;
        
        let sampling_rate = model.sampling_rate() as u32;
        
        Ok(Self { model, sampling_rate })
    }

    /// Transcribe audio file and return structured segments
    #[napi]
    pub fn transcribe_segments(
        &self,
        audio_path: String,
        options: Option<TranscribeOptions>,
    ) -> napi::Result<TranscriptionResult> {
        let opts = options.unwrap_or_default();
        let samples = read_wav_samples(&audio_path)
            .map_err(|e| napi::Error::from_reason(format!("Failed to read audio: {}", e)))?;
        
        self.transcribe_samples_internal(&samples, &opts)
    }

    /// Simple transcription returning just the text (backward compatible)
    #[napi]
    pub fn transcribe(&self, audio_file: String) -> napi::Result<String> {
        let result = self.transcribe_segments(audio_file, None)?;
        Ok(result.text)
    }

    /// Transcribe with options, returning just the text
    #[napi]
    pub fn transcribe_with_options(
        &self,
        audio_file: String,
        options: TranscribeOptions,
    ) -> napi::Result<String> {
        let result = self.transcribe_segments(audio_file, Some(options))?;
        Ok(result.text)
    }

    /// Transcribe from a Buffer containing WAV audio data
    #[napi]
    pub fn transcribe_buffer(
        &self,
        buffer: napi::bindgen_prelude::Buffer,
        options: Option<TranscribeOptions>,
    ) -> napi::Result<TranscriptionResult> {
        let opts = options.unwrap_or_default();
        let samples = parse_wav_buffer(&buffer)
            .map_err(|e| napi::Error::from_reason(format!("Failed to parse audio buffer: {}", e)))?;
        
        self.transcribe_samples_internal(&samples, &opts)
    }

    /// Transcribe from raw Float32Array samples (must be 16kHz mono, normalized to [-1, 1])
    #[napi]
    pub fn transcribe_samples(
        &self,
        samples: Vec<f64>,
        options: Option<TranscribeOptions>,
    ) -> napi::Result<TranscriptionResult> {
        let opts = options.unwrap_or_default();
        // Convert f64 to f32
        let samples_f32: Vec<f32> = samples.iter().map(|&x| x as f32).collect();
        
        self.transcribe_samples_internal(&samples_f32, &opts)
    }

    /// Get the expected sampling rate (16000 Hz for Whisper)
    #[napi]
    pub fn sampling_rate(&self) -> u32 {
        self.sampling_rate
    }

    /// Check if the model is multilingual
    #[napi]
    pub fn is_multilingual(&self) -> bool {
        self.model.is_multilingual()
    }

    /// Get the number of supported languages
    #[napi]
    pub fn num_languages(&self) -> u32 {
        self.model.num_languages() as u32
    }

    // Internal transcription implementation
    fn transcribe_samples_internal(
        &self,
        samples: &[f32],
        opts: &TranscribeOptions,
    ) -> napi::Result<TranscriptionResult> {
        // Calculate duration
        let duration = samples.len() as f64 / self.sampling_rate as f64;
        
        // Build whisper options
        let whisper_opts = self.build_whisper_options(opts);
        
        // Determine if we want timestamps
        let timestamp = opts.word_timestamps.unwrap_or(false);
        
        // Perform transcription
        let results = self.model.generate(
            samples,
            opts.language.as_deref(),
            timestamp,
            &whisper_opts,
        ).map_err(|e| napi::Error::from_reason(format!("Transcription failed: {}", e)))?;
        
        // Build segments from results
        let mut segments = Vec::new();
        let mut full_text = String::new();
        let samples_per_segment = self.model.n_samples();
        
        for (idx, result) in results.iter().enumerate() {
            let text = result.trim().to_string();
            if !text.is_empty() {
                let segment_start = (idx * samples_per_segment) as f64 / self.sampling_rate as f64;
                let segment_end = ((idx + 1) * samples_per_segment) as f64 / self.sampling_rate as f64;
                let segment_end = segment_end.min(duration);
                
                if !full_text.is_empty() {
                    full_text.push(' ');
                }
                full_text.push_str(&text);
                
                segments.push(Segment {
                    id: idx as u32,
                    seek: (idx * samples_per_segment) as u32,
                    start: segment_start,
                    end: segment_end,
                    text,
                    tokens: vec![], // ct2rs doesn't expose token IDs directly in generate()
                    temperature: opts.temperature.unwrap_or(1.0),
                    avg_logprob: 0.0, // Not available from ct2rs high-level API
                    compression_ratio: 0.0,
                    no_speech_prob: 0.0,
                });
            }
        }
        
        Ok(TranscriptionResult {
            segments,
            language: opts.language.clone().unwrap_or_else(|| "auto".to_string()),
            language_probability: 0.0, // Would need low-level API for this
            duration,
            text: full_text,
        })
    }

    // Helper: build WhisperOptions from TranscribeOptions
    fn build_whisper_options(&self, opts: &TranscribeOptions) -> WhisperOptions {
        let mut whisper_opts = WhisperOptions::default();
        
        if let Some(beam_size) = opts.beam_size {
            whisper_opts.beam_size = beam_size as usize;
        }
        if let Some(patience) = opts.patience {
            whisper_opts.patience = patience as f32;
        }
        if let Some(length_penalty) = opts.length_penalty {
            whisper_opts.length_penalty = length_penalty as f32;
        }
        if let Some(repetition_penalty) = opts.repetition_penalty {
            whisper_opts.repetition_penalty = repetition_penalty as f32;
        }
        if let Some(no_repeat_ngram_size) = opts.no_repeat_ngram_size {
            whisper_opts.no_repeat_ngram_size = no_repeat_ngram_size as usize;
        }
        if let Some(temperature) = opts.temperature {
            whisper_opts.sampling_temperature = temperature as f32;
        }
        if let Some(suppress_blank) = opts.suppress_blank {
            whisper_opts.suppress_blank = suppress_blank;
        }
        if let Some(max_length) = opts.max_length {
            whisper_opts.max_length = max_length as usize;
        }
        
        whisper_opts
    }
}

/// Read WAV file and return normalized f32 samples in range [-1, 1]
fn read_wav_samples(path: &str) -> anyhow::Result<Vec<f32>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    parse_wav_buffer(&buffer)
}

/// Parse WAV buffer and return normalized f32 samples
fn parse_wav_buffer(buffer: &[u8]) -> anyhow::Result<Vec<f32>> {
    if buffer.len() < 44 {
        anyhow::bail!("File too small to be a valid WAV");
    }
    
    // Check for RIFF header
    if &buffer[0..4] != b"RIFF" || &buffer[8..12] != b"WAVE" {
        anyhow::bail!("Not a valid WAV file");
    }
    
    // Parse fmt chunk
    let mut pos = 12;
    let mut audio_format = 0u16;
    let mut num_channels = 0u16;
    let mut sample_rate = 0u32;
    let mut bits_per_sample = 0u16;
    
    while pos + 8 <= buffer.len() {
        let chunk_id = &buffer[pos..pos+4];
        let chunk_size = u32::from_le_bytes([
            buffer[pos+4], buffer[pos+5], buffer[pos+6], buffer[pos+7]
        ]) as usize;
        
        if chunk_id == b"fmt " {
            if pos + 8 + 16 > buffer.len() {
                anyhow::bail!("Invalid fmt chunk");
            }
            audio_format = u16::from_le_bytes([buffer[pos+8], buffer[pos+9]]);
            num_channels = u16::from_le_bytes([buffer[pos+10], buffer[pos+11]]);
            sample_rate = u32::from_le_bytes([
                buffer[pos+12], buffer[pos+13], buffer[pos+14], buffer[pos+15]
            ]);
            bits_per_sample = u16::from_le_bytes([buffer[pos+22], buffer[pos+23]]);
        } else if chunk_id == b"data" {
            // Validate format
            if audio_format != 1 {
                anyhow::bail!("Only PCM format is supported (got format {})", audio_format);
            }
            if sample_rate != 16000 {
                anyhow::bail!(
                    "Sample rate must be 16000 Hz (got {} Hz). Please resample your audio.",
                    sample_rate
                );
            }
            
            let data_start = pos + 8;
            let data_end = (data_start + chunk_size).min(buffer.len());
            let data = &buffer[data_start..data_end];
            
            return parse_pcm_samples(data, num_channels, bits_per_sample);
        }
        
        pos += 8 + chunk_size;
        // Align to 2 bytes
        if chunk_size % 2 != 0 {
            pos += 1;
        }
    }
    
    anyhow::bail!("No data chunk found in WAV file")
}

/// Parse PCM samples from raw bytes
fn parse_pcm_samples(data: &[u8], channels: u16, bits_per_sample: u16) -> anyhow::Result<Vec<f32>> {
    let bytes_per_sample = (bits_per_sample / 8) as usize;
    let frame_size = bytes_per_sample * channels as usize;
    
    let samples: Vec<f32> = match bits_per_sample {
        16 => {
            data.chunks(frame_size)
                .filter_map(|frame| {
                    if frame.len() >= bytes_per_sample {
                        // Take first channel (mono) or average for stereo
                        if channels == 1 {
                            let sample = i16::from_le_bytes([frame[0], frame[1]]);
                            Some(sample as f32 / 32768.0)
                        } else if channels == 2 && frame.len() >= 4 {
                            // Average left and right channels
                            let left = i16::from_le_bytes([frame[0], frame[1]]) as f32;
                            let right = i16::from_le_bytes([frame[2], frame[3]]) as f32;
                            Some((left + right) / 2.0 / 32768.0)
                        } else {
                            // Take first channel for multi-channel
                            let sample = i16::from_le_bytes([frame[0], frame[1]]);
                            Some(sample as f32 / 32768.0)
                        }
                    } else {
                        None
                    }
                })
                .collect()
        }
        24 => {
            data.chunks(frame_size)
                .filter_map(|frame| {
                    if frame.len() >= 3 {
                        // 24-bit samples need special handling
                        let sample = ((frame[2] as i32) << 16 | (frame[1] as i32) << 8 | (frame[0] as i32)) as i32;
                        // Sign extend from 24-bit
                        let sample = if sample & 0x800000 != 0 {
                            sample | !0xFFFFFF
                        } else {
                            sample
                        };
                        Some(sample as f32 / 8388608.0)
                    } else {
                        None
                    }
                })
                .collect()
        }
        32 => {
            data.chunks(frame_size)
                .filter_map(|frame| {
                    if frame.len() >= 4 {
                        // Could be 32-bit int or float - assume float for now
                        let sample = f32::from_le_bytes([frame[0], frame[1], frame[2], frame[3]]);
                        Some(sample)
                    } else {
                        None
                    }
                })
                .collect()
        }
        8 => {
            data.chunks(frame_size)
                .filter_map(|frame| {
                    if !frame.is_empty() {
                        // 8-bit is unsigned
                        let sample = (frame[0] as i16 - 128) as f32 / 128.0;
                        Some(sample)
                    } else {
                        None
                    }
                })
                .collect()
        }
        _ => {
            anyhow::bail!("Unsupported bit depth: {} bits", bits_per_sample);
        }
    };
    
    Ok(samples)
}

/// Get list of supported model size aliases
#[napi]
pub fn available_models() -> Vec<String> {
    vec![
        "tiny".to_string(),
        "tiny.en".to_string(),
        "base".to_string(),
        "base.en".to_string(),
        "small".to_string(),
        "small.en".to_string(),
        "medium".to_string(),
        "medium.en".to_string(),
        "large-v1".to_string(),
        "large-v2".to_string(),
        "large-v3".to_string(),
    ]
}

/// Format seconds to timestamp string (HH:MM:SS.mmm or MM:SS.mmm)
#[napi]
pub fn format_timestamp(seconds: f64, always_include_hours: Option<bool>) -> String {
    let include_hours = always_include_hours.unwrap_or(false);
    let hours = (seconds / 3600.0).floor() as u32;
    let minutes = ((seconds % 3600.0) / 60.0).floor() as u32;
    let secs = seconds % 60.0;
    
    if include_hours || hours > 0 {
        format!("{:02}:{:02}:{:06.3}", hours, minutes, secs)
    } else {
        format!("{:02}:{:06.3}", minutes, secs)
    }
}
