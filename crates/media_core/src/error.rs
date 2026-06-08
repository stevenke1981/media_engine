use crate::backend::BackendKind;
use crate::pixel_format::PixelFormat;

/// Unified error type for the media engine.
#[derive(thiserror::Error, Debug)]
pub enum MediaError {
    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error("WGPU error: {0}")]
    Wgpu(String),

    #[error("Image error: {0}")]
    Image(String),

    #[error("Video error: {0}")]
    Video(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Unsupported backend: {0:?}")]
    UnsupportedBackend(BackendKind),

    #[error("Unsupported pixel format: {0:?}")]
    UnsupportedPixelFormat(PixelFormat),

    #[error("Frame transfer failed: {from:?} -> {to:?}")]
    TransferFailed { from: BackendKind, to: BackendKind },

    #[error("Invalid frame size: {width}x{height}")]
    InvalidFrameSize { width: u32, height: u32 },

    #[error("Invalid stride: expected {expected}, got {actual}")]
    InvalidStride { expected: usize, actual: usize },

    #[error("Buffer too small: needed {needed}, got {actual}")]
    BufferTooSmall { needed: usize, actual: usize },

    #[error("Allocation failed: {0}")]
    Allocation(String),

    #[error("Kernel error: {name}: {message}")]
    Kernel { name: String, message: String },

    #[error("{0}")]
    Other(String),
}

/// Convenience alias.
pub type MediaResult<T> = Result<T, MediaError>;

impl From<String> for MediaError {
    fn from(s: String) -> Self {
        MediaError::Other(s)
    }
}

impl From<&str> for MediaError {
    fn from(s: &str) -> Self {
        MediaError::Other(s.to_string())
    }
}
