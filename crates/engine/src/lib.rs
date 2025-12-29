use napi_derive::napi;
use ct2rs::{Whisper, WhisperOptions, Config};
use std::fs::File;
use std::io::Read;

#[napi]
pub struct Engine {
    model: Whisper,
}

#[napi]
impl Engine {
    #[napi(constructor)]
    pub fn new(model_path: String) -> napi::Result<Self> {
        // Load the model from disk (CTranslate2 format)
        let config = Config::default();
        let model = Whisper::new(&model_path, config)
            .map_err(|e| napi::Error::from_reason(format!("Load failed: {}", e)))?;
        Ok(Self { model })
    }

    #[napi]
    pub fn transcribe(&self, audio_file: String) -> napi::Result<String> {
        // Read audio file and convert to samples
        // Note: In production, you'd use a proper audio library to read WAV/MP3
        // For now, we'll read raw PCM samples from a WAV file
        let samples = read_wav_samples(&audio_file)
            .map_err(|e| napi::Error::from_reason(format!("Failed to read audio: {}", e)))?;
        
        // Perform transcription
        let options = WhisperOptions::default();
        let result = self.model.generate(&samples, None, false, &options)
            .map_err(|e| napi::Error::from_reason(format!("Inference failed: {}", e)))?;
        
        // Combine all segments into one string
        let full_text = result.join(" ");
        Ok(full_text)
    }

    #[napi]
    pub fn sampling_rate(&self) -> u32 {
        self.model.sampling_rate() as u32
    }
}

/// Read WAV file and return normalized f32 samples in range [-1, 1]
fn read_wav_samples(path: &str) -> anyhow::Result<Vec<f32>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    
    // Simple WAV parser - assumes 16-bit PCM mono
    // Skip the first 44 bytes (standard WAV header)
    if buffer.len() < 44 {
        anyhow::bail!("File too small to be a valid WAV");
    }
    
    // Check for RIFF header
    if &buffer[0..4] != b"RIFF" || &buffer[8..12] != b"WAVE" {
        anyhow::bail!("Not a valid WAV file");
    }
    
    // Find the data chunk
    let mut pos = 12;
    let mut data_start = 0usize;
    let mut data_size = 0usize;
    
    while pos + 8 <= buffer.len() {
        let chunk_id = &buffer[pos..pos+4];
        let chunk_size = u32::from_le_bytes([
            buffer[pos+4], buffer[pos+5], buffer[pos+6], buffer[pos+7]
        ]) as usize;
        
        if chunk_id == b"data" {
            data_start = pos + 8;
            data_size = chunk_size;
            break;
        }
        pos += 8 + chunk_size;
    }
    
    if data_start == 0 {
        anyhow::bail!("No data chunk found in WAV file");
    }
    
    // Parse 16-bit PCM samples
    let samples: Vec<f32> = buffer[data_start..data_start + data_size]
        .chunks(2)
        .filter_map(|chunk| {
            if chunk.len() == 2 {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                Some(sample as f32 / 32768.0)
            } else {
                None
            }
        })
        .collect();
    
    Ok(samples)
}
