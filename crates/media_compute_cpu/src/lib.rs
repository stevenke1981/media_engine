//! # Media Compute CPU
//!
//! CPU compute backend for the Rust Media Engine.
//! Implements the `ComputeBackend` trait with scalar Rust loops.

pub mod backend;
pub mod buffer;
pub mod effects;
pub mod simd;

pub use backend::CpuBackend;
pub use buffer::CpuBuffer;
pub use effects::{
    BlurEffect, BrightnessEffect, ContrastEffect, GrayscaleEffect, InvertEffect, SharpenEffect,
};
