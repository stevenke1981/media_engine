//! Trait for image decode/encode operations decoupled from any specific backend.
//!
//! Backends such as `CpuBackend` implement `ImageCodec` alongside `ComputeBackend`.

use media_core::{EncoderFormat, Frame, MediaError};

/// Backend-agnostic image codec trait.
///
/// Handles decoding image bytes into [`Frame`]s and encoding [`Frame`]s back into
/// image bytes in a specified format.  The backend may use SIMD or GPU kernels
/// during pixel format conversion (e.g. BGRA→RGBA before handing to the encoder).
pub trait ImageCodec {
    /// Decode encoded image bytes into a [`Frame`].
    ///
    /// `format_hint` can be `None` to let the decoder sniff the format from
    /// magic bytes, or `Some` to force a specific decoder.
    fn decode_from_bytes(
        &self,
        data: &[u8],
        format_hint: Option<media_core::PixelFormat>,
    ) -> Result<Frame, MediaError>;

    /// Encode a [`Frame`] into the specified [`EncoderFormat`].
    fn encode_to_bytes(&self, frame: &Frame, format: EncoderFormat) -> Result<Vec<u8>, MediaError>;
}
