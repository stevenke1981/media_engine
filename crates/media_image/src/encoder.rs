//! Image encoder (Frame → file bytes).

use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::PngEncoder;
use image::codecs::webp::WebPEncoder;
use image::{ColorType, ExtendedColorType, ImageEncoder};
use media_core::{EncoderFormat, Frame, FrameStorage, MediaError, MediaResult, PixelFormat};

/// Write a CPU-backed Frame into the given writer.
pub fn encode_to_writer(
    frame: &Frame,
    format: EncoderFormat,
    writer: &mut impl std::io::Write,
) -> MediaResult<()> {
    let (width, height, raw) = match &frame.storage {
        FrameStorage::Cpu(cpu) => {
            let width = frame.width;
            let height = frame.height;
            (width, height, cpu.data.as_slice())
        }
        _ => {
            return Err(MediaError::Other(
                "Only CPU frames can be encoded".to_string(),
            ))
        }
    };

    match format {
        EncoderFormat::Png => {
            let color_type: ExtendedColorType = rgba_format_to_color_type(frame.format)?.into();
            let encoder = PngEncoder::new(writer);
            encoder
                .write_image(raw, width, height, color_type)
                .map_err(|e| MediaError::Image(e.to_string()))?;
        }
        EncoderFormat::Jpeg { quality } => {
            // JPEG does not support alpha. Convert RGBA8 → RGB8 inline.
            let (data, color_type) = if frame.format == PixelFormat::Rgba8 {
                let rgb: Vec<u8> = raw.chunks(4).flat_map(|p| p[..3].to_vec()).collect();
                (rgb, ColorType::Rgb8)
            } else {
                (raw.to_vec(), ColorType::Rgb8)
            };
            let mut encoder = JpegEncoder::new_with_quality(writer, quality);
            encoder
                .encode(&data, width, height, color_type.into())
                .map_err(|e| MediaError::Image(e.to_string()))?;
        }
        EncoderFormat::WebP { quality: _quality } => {
            let color_type: ExtendedColorType = rgba_format_to_color_type(frame.format)?.into();
            let encoder = WebPEncoder::new_lossless(writer);
            encoder
                .write_image(raw, width, height, color_type)
                .map_err(|e| MediaError::Image(e.to_string()))?;
            // NOTE: For lossy WebP, use `image::codecs::webp::WebPEncoder::new_lossy`,
            // or the `libwebp` crate.
        }
    }

    Ok(())
}

/// Encode to bytes in memory.
pub fn encode_to_bytes(frame: &Frame, format: EncoderFormat) -> MediaResult<Vec<u8>> {
    let mut buf = Vec::new();
    encode_to_writer(frame, format, &mut buf)?;
    Ok(buf)
}

fn rgba_format_to_color_type(format: PixelFormat) -> MediaResult<ColorType> {
    match format {
        PixelFormat::Rgba8 => Ok(ColorType::Rgba8),
        PixelFormat::Rgb8 => Ok(ColorType::Rgb8),
        _ => Err(MediaError::UnsupportedPixelFormat(format)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use media_core::Frame;

    #[test]
    fn test_encode_png_roundtrip() {
        // Create a 4x4 RGBA8 frame with a red pixel.
        let mut frame = Frame::new_cpu(4, 4, PixelFormat::Rgba8).unwrap();
        if let Some(cpu) = frame.as_cpu_mut() {
            // Row 0 pixel 0 = solid red.
            cpu.data[0] = 255;
            cpu.data[1] = 0;
            cpu.data[2] = 0;
            cpu.data[3] = 255;
        }

        let bytes = encode_to_bytes(&frame, EncoderFormat::Png).unwrap();
        assert!(!bytes.is_empty());

        // Decode back and verify.
        let decoded = super::super::decoder::decode_from_bytes(&bytes).unwrap();
        assert!(decoded.width > 0);
        assert!(decoded.height > 0);
    }

    #[test]
    fn test_encode_jpeg() {
        let frame = Frame::new_cpu(8, 8, PixelFormat::Rgba8).unwrap();
        let bytes = encode_to_bytes(&frame, EncoderFormat::Jpeg { quality: 85 }).unwrap();
        assert!(!bytes.is_empty());
    }
}
