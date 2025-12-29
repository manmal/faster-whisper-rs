//! Model downloading from HuggingFace Hub
//!
//! Downloads Whisper models in GGML format from ggerganov/whisper.cpp repo.

use anyhow::{Context, Result};
use futures::StreamExt;
use reqwest::Client;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;

/// Model sizes and their GGML filenames on HuggingFace
/// Format: (alias, ggml_filename)
const MODEL_FILES: &[(&str, &str)] = &[
    ("tiny", "ggml-tiny.bin"),
    ("tiny.en", "ggml-tiny.en.bin"),
    ("base", "ggml-base.bin"),
    ("base.en", "ggml-base.en.bin"),
    ("small", "ggml-small.bin"),
    ("small.en", "ggml-small.en.bin"),
    ("medium", "ggml-medium.bin"),
    ("medium.en", "ggml-medium.en.bin"),
    ("large-v1", "ggml-large-v1.bin"),
    ("large-v2", "ggml-large-v2.bin"),
    ("large-v3", "ggml-large-v3.bin"),
    ("large-v3-turbo", "ggml-large-v3-turbo.bin"),
];

/// HuggingFace repository for GGML models
const GGML_REPO: &str = "ggerganov/whisper.cpp";

/// Get the default cache directory for models
pub fn default_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("whisper-node")
        .join("models")
}

/// Get the path for a specific model size
pub fn model_path_for_size(size: &str, cache_dir: Option<&Path>) -> PathBuf {
    let cache = cache_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(default_cache_dir);
    
    // Return path to the .bin file directly
    if let Some((_, filename)) = MODEL_FILES.iter().find(|(s, _)| *s == size) {
        cache.join(filename)
    } else {
        // Assume it's a custom path/filename
        cache.join(format!("ggml-{}.bin", size))
    }
}

/// Check if a model is already downloaded
pub fn is_model_downloaded(size: &str, cache_dir: Option<&Path>) -> bool {
    let model_path = model_path_for_size(size, cache_dir);
    model_path.exists()
}

/// Resolve a model path or size to an actual path
/// If a path is given, returns it directly
/// If a size alias is given, returns the cached model path
pub fn resolve_model_path(path_or_size: &str) -> String {
    // If it looks like a path (contains / or \ or ends with .bin), return as-is
    if path_or_size.contains('/') || path_or_size.contains('\\') 
        || path_or_size.ends_with(".bin")
        || Path::new(path_or_size).exists() {
        return path_or_size.to_string();
    }
    
    // It's a size alias, return the cache path
    model_path_for_size(path_or_size, None)
        .to_string_lossy()
        .into_owned()
}

/// Get the GGML filename for a model size
pub fn get_repo_for_size(size: &str) -> Option<&'static str> {
    MODEL_FILES.iter()
        .find(|(s, _)| *s == size)
        .map(|(_, filename)| *filename)
}

/// Download a model from HuggingFace Hub
pub async fn download_model<F>(
    size: &str,
    cache_dir: Option<&Path>,
    progress_callback: Option<F>,
) -> Result<PathBuf>
where
    F: Fn(f64) + Send + Sync,
{
    let filename = MODEL_FILES.iter()
        .find(|(s, _)| *s == size)
        .map(|(_, f)| *f)
        .context(format!(
            "Unknown model size: {}. Available: {:?}", 
            size, 
            MODEL_FILES.iter().map(|(s, _)| *s).collect::<Vec<_>>()
        ))?;
    
    let cache = cache_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(default_cache_dir);
    
    // Create cache directory
    fs::create_dir_all(&cache).await
        .context("Failed to create cache directory")?;
    
    let model_path = cache.join(filename);
    
    // Check if already exists
    if model_path.exists() {
        if let Some(ref cb) = progress_callback {
            cb(100.0);
        }
        return Ok(model_path);
    }
    
    let client = Client::builder()
        .user_agent("whisper-node/0.1")
        .build()
        .context("Failed to create HTTP client")?;
    
    // Download URL from HuggingFace
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        GGML_REPO, filename
    );
    
    // First, get the file size with a HEAD request
    let head_response = client.head(&url)
        .send()
        .await
        .context("Failed to get file info")?;
    
    let total_size = head_response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    
    // Download the file
    let response = client.get(&url)
        .send()
        .await
        .context("Failed to start download")?;
    
    if !response.status().is_success() {
        anyhow::bail!("Download failed: {} for URL {}", response.status(), url);
    }
    
    // Write to a temporary file first
    let temp_path = cache.join(format!("{}.tmp", filename));
    let mut file = fs::File::create(&temp_path).await
        .context("Failed to create temp file")?;
    
    let mut downloaded: u64 = 0;
    let mut stream = response.bytes_stream();
    
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Failed to read chunk")?;
        file.write_all(&chunk).await
            .context("Failed to write chunk")?;
        
        downloaded += chunk.len() as u64;
        
        if total_size > 0 {
            if let Some(ref cb) = progress_callback {
                cb((downloaded as f64 / total_size as f64) * 100.0);
            }
        }
    }
    
    file.flush().await?;
    drop(file);
    
    // Rename temp file to final path
    fs::rename(&temp_path, &model_path).await
        .context("Failed to rename temp file")?;
    
    if let Some(ref cb) = progress_callback {
        cb(100.0);
    }
    
    Ok(model_path)
}

/// List available model sizes
pub fn available_model_sizes() -> Vec<&'static str> {
    MODEL_FILES.iter().map(|(s, _)| *s).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_cache_dir() {
        let dir = default_cache_dir();
        assert!(dir.to_string_lossy().contains("whisper-node"));
    }
    
    #[test]
    fn test_resolve_model_path_existing() {
        let resolved = resolve_model_path("/some/path/model.bin");
        assert_eq!(resolved, "/some/path/model.bin");
    }
    
    #[test]
    fn test_resolve_model_path_size() {
        let resolved = resolve_model_path("tiny");
        assert!(resolved.contains("ggml-tiny.bin"));
    }
    
    #[test]
    fn test_get_repo() {
        assert_eq!(get_repo_for_size("tiny"), Some("ggml-tiny.bin"));
        assert_eq!(get_repo_for_size("large-v3"), Some("ggml-large-v3.bin"));
        assert_eq!(get_repo_for_size("invalid"), None);
    }
    
    #[test]
    fn test_available_sizes() {
        let sizes = available_model_sizes();
        assert!(sizes.contains(&"tiny"));
        assert!(sizes.contains(&"base"));
        assert!(sizes.contains(&"large-v3"));
    }
}
