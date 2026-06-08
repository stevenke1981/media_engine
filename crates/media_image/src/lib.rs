//! # Media Image
//!
//! Image decode / encode pipeline for the Rust Media Engine.
//! Wraps the `image` crate to decode PNG/JPEG/WebP into `Frame` and encode back.

pub mod convert;
pub mod decoder;
pub mod encoder;

pub use decoder::{
    decode_from_bytes, decode_from_path, probe_from_bytes, probe_from_path, ProbeResult,
};
pub use encoder::{encode_to_bytes, encode_to_writer};
pub use media_core::EncoderFormat;
