//! # Media Core
//!
//! Core data types for the Rust Media Engine.
//! This crate has zero dependencies on backend libraries (CUDA, WGPU, FFmpeg, etc.).

pub mod backend;
pub mod error;
pub mod frame;
pub mod pixel_format;

pub use backend::*;
pub use error::*;
pub use frame::*;
pub use pixel_format::*;

/// Supported output image encoding format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EncoderFormat {
    Png,
    Jpeg { quality: u8 },
    WebP { quality: f32 },
}
