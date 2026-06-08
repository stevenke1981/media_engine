//! # Media Compute
//!
//! Compute backend abstraction layer for the Rust Media Engine.
//! Defines the `ComputeBackend` and `ComputeEffect` traits that
//! CPU, CUDA, and WGPU backends implement.

pub mod buffer;
pub mod codec;
pub mod effect;
pub mod kernel;

pub use buffer::ComputeBuffer;
pub use codec::ImageCodec;
pub use effect::{block_for_bpp, grid_for_size, ComputeBackend, ComputeEffect, DEFAULT_BLOCK_SIZE};
pub use kernel::{
    Args, EffectDesc, EffectKind, EffectParam, Kernel, KernelArg, KernelDesc, LaunchConfig,
};
