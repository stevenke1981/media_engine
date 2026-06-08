//! # media_compute_cuda — CUDA compute backend for the Rust Media Engine.
//!
//! This crate implements `ComputeBackend` using the CUDA Driver API
//! (`nvcuda.dll` / `libcuda.so`).  PTX kernels are compiled at build
//! time via `nvcc` and embedded in the binary.
//!
//! ## Usage
//! ```rust,ignore
//! use media_compute_cuda::CudaBackend;
//! use media_compute::ComputeBackend;
//!
//! let backend = CudaBackend::new()?;
//! assert!(backend.has_effect(&EffectKind::Grayscale));
//! ```

pub mod backend;
pub mod buffer;
pub mod device;
pub mod effects;
pub mod ffi;
pub mod module;

pub use backend::CudaBackend;
pub use buffer::CudaBuffer;
pub use device::{CudaContext, CudaDevice};
pub use module::{CudaFunction, CudaModule};
