//! CPU compute backend implementation.

use crate::buffer::CpuBuffer;
use crate::effects::{
    BlurEffect, BrightnessEffect, ContrastEffect, GrayscaleEffect, InvertEffect, SharpenEffect,
};
use media_compute::{
    ComputeBackend, ComputeBuffer, ComputeEffect, EffectDesc, EffectKind, ImageCodec,
};
use media_core::{EncoderFormat, Frame, MediaError, MediaResult, PixelFormat};

/// The CPU compute backend.
///
/// This backend runs all effects using Rust scalar loops (with rayon
/// parallelism available via the `apply_parallel` feature).
pub struct CpuBackend {
    effects: Vec<Box<dyn ComputeEffect>>,
}

impl CpuBackend {
    /// Create a new CPU backend with the standard set of effects.
    pub fn new() -> Self {
        let effects: Vec<Box<dyn ComputeEffect>> = vec![
            Box::new(BrightnessEffect),
            Box::new(ContrastEffect),
            Box::new(GrayscaleEffect),
            Box::new(InvertEffect),
            Box::new(BlurEffect),
            Box::new(SharpenEffect),
        ];

        CpuBackend { effects }
    }

    /// Create a CPU backend with a custom list of effects.
    pub fn with_effects(effects: Vec<Box<dyn ComputeEffect>>) -> Self {
        CpuBackend { effects }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn alloc_buffer(&self, size: usize) -> MediaResult<Box<dyn ComputeBuffer>> {
        Ok(Box::new(CpuBuffer::new(size)))
    }

    fn register_buffer(&self, data: &[u8]) -> MediaResult<Box<dyn ComputeBuffer>> {
        Ok(Box::new(CpuBuffer::from_slice(data)))
    }

    fn available_effects(&self) -> Vec<EffectDesc> {
        self.effects
            .iter()
            .map(|e| {
                let kind = match e.name() {
                    "brightness" => EffectKind::Brightness,
                    "contrast" => EffectKind::Contrast,
                    "grayscale" => EffectKind::Grayscale,
                    "invert" => EffectKind::Invert,
                    "blur" => EffectKind::Blur,
                    "sharpen" => EffectKind::Sharpen,
                    name => EffectKind::Custom(name.to_string()),
                };
                EffectDesc::new(kind)
            })
            .collect()
    }

    fn has_effect(&self, kind: &EffectKind) -> bool {
        self.effects
            .iter()
            .any(|e| effect_kind_matches(e.name(), kind))
    }

    fn get_effect(&self, kind: &EffectKind) -> Option<&dyn ComputeEffect> {
        self.effects
            .iter()
            .find(|e| effect_kind_matches(e.name(), kind))
            .map(|e| e.as_ref())
    }
}

impl ImageCodec for CpuBackend {
    /// Decode image bytes into a CPU frame.
    ///
    /// Delegates to `media_image`, optionally converting a BGRA input into RGBA
    /// for consistency within the pipeline.  The `format_hint` parameter is
    /// currently unused (auto-detect by magic bytes).
    fn decode_from_bytes(
        &self,
        data: &[u8],
        _format_hint: Option<PixelFormat>,
    ) -> Result<Frame, MediaError> {
        media_image::decode_from_bytes(data)
    }

    /// Encode a CPU frame into the specified output format.
    ///
    /// Uses SIMD-accelerated pixel conversions where applicable
    /// (RGBA → RGB stripping for JPEG output).
    fn encode_to_bytes(&self, frame: &Frame, format: EncoderFormat) -> Result<Vec<u8>, MediaError> {
        // For JPEG + RGBA8 source: use SIMD rgba→rgb before encoding.
        let frame_to_encode =
            if frame.format == PixelFormat::Rgba8 && matches!(format, EncoderFormat::Jpeg { .. }) {
                let cpu = frame
                    .as_cpu()
                    .ok_or_else(|| MediaError::Other("Frame is not CPU-backed".into()))?;
                let mut rgb_data = cpu.data.clone();
                let rgb_len = rgb_data.len() * 3 / 4;
                crate::simd::rgba_to_rgb(&mut rgb_data);
                rgb_data.truncate(rgb_len);

                let mut rgb_frame = Frame::new_cpu(frame.width, frame.height, PixelFormat::Rgb8)
                    .map_err(|e| MediaError::Other(e.to_string()))?;
                if let Some(cpu_rgb) = rgb_frame.as_cpu_mut() {
                    cpu_rgb.data = rgb_data;
                }
                rgb_frame
            } else {
                frame.clone()
            };

        media_image::encode_to_bytes(&frame_to_encode, format)
    }
}

fn effect_kind_matches(name: &str, kind: &EffectKind) -> bool {
    match kind {
        EffectKind::Brightness => name == "brightness",
        EffectKind::Contrast => name == "contrast",
        EffectKind::Grayscale => name == "grayscale",
        EffectKind::Invert => name == "invert",
        EffectKind::Blur => name == "blur",
        EffectKind::Sharpen => name == "sharpen",
        EffectKind::Custom(s) => name == s.as_str(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_name() {
        let backend = CpuBackend::new();
        assert_eq!(backend.name(), "cpu");
    }

    #[test]
    fn test_alloc_buffer() {
        let backend = CpuBackend::new();
        let buf = backend.alloc_buffer(1024).unwrap();
        assert_eq!(buf.size(), 1024);
    }

    #[test]
    fn test_has_effect() {
        let backend = CpuBackend::new();
        assert!(backend.has_effect(&EffectKind::Brightness));
        assert!(backend.has_effect(&EffectKind::Contrast));
        assert!(backend.has_effect(&EffectKind::Grayscale));
        assert!(backend.has_effect(&EffectKind::Invert));
        assert!(backend.has_effect(&EffectKind::Blur));
        assert!(backend.has_effect(&EffectKind::Sharpen));
    }

    #[test]
    fn test_available_effects() {
        let backend = CpuBackend::new();
        let effects = backend.available_effects();
        assert!(effects.len() >= 6);
    }

    #[test]
    fn test_get_effect() {
        let backend = CpuBackend::new();
        let effect = backend.get_effect(&EffectKind::Grayscale);
        assert!(effect.is_some());
        assert_eq!(effect.unwrap().name(), "grayscale");
    }

    #[test]
    fn test_register_buffer() {
        let backend = CpuBackend::new();
        let buf = backend.register_buffer(&[1, 2, 3, 4]).unwrap();
        assert_eq!(buf.size(), 4);
    }
}
