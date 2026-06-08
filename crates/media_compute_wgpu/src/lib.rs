//! WGPU compute backend for the Rust Media Engine.
//!
//! Provides `WgpuBackend` implementing `ComputeBackend` using
//! wgpu (WebGPU) compute shaders implemented in WGSL.

mod backend;
mod buffer;
mod effects;

pub use backend::WgpuBackend;
pub use buffer::WgpuBuffer;
pub use effects::WgpuEffect;
