//! Image decoder (file bytes → Frame).

use image::ImageReader;
use media_core::{CpuFrame, Frame, FrameStorage, MediaError, MediaResult, PixelFormat};

/// Decode an image from a byte slice, returning a CPU-backed Frame.
///
/// The image is converted to RGBA8 before returning.
pub fn decode_from_bytes(data: &[u8]) -> MediaResult<Frame> {
    let img = ImageReader::new(std::io::Cursor::new(data))
        .with_guessed_format()
        .map_err(|e| MediaError::Image(e.to_string()))?
        .decode()
        .map_err(|e| MediaError::Image(e.to_string()))?;

    let width = img.width();
    let height = img.height();

    // Convert to RGBA8.
    let rgba = img.to_rgba8();
    let stride = width as usize * 4;

    let cpu = CpuFrame {
        data: rgba.into_raw(),
        stride,
    };

    Ok(Frame {
        width,
        height,
        format: PixelFormat::Rgba8,
        timestamp: None,
        color_space: None,
        storage: FrameStorage::Cpu(cpu),
    })
}

/// Decode an image from a file path.
pub fn decode_from_path<P: AsRef<std::path::Path>>(path: P) -> MediaResult<Frame> {
    let data = std::fs::read(path.as_ref())?;
    decode_from_bytes(&data)
}

/// Query image dimensions and format without full decode.
pub fn probe_from_bytes(data: &[u8]) -> MediaResult<ProbeResult> {
    let reader = ImageReader::new(std::io::Cursor::new(data))
        .with_guessed_format()
        .map_err(|e| MediaError::Image(e.to_string()))?;

    let format = reader
        .format()
        .ok_or_else(|| MediaError::Image("Unknown image format".into()))?;

    // Read just the header.
    let (width, height) = reader
        .into_dimensions()
        .map_err(|e| MediaError::Image(e.to_string()))?;

    Ok(ProbeResult {
        width,
        height,
        format_name: format!("{:?}", format),
    })
}

/// Probe an image from a file path.
pub fn probe_from_path<P: AsRef<std::path::Path>>(path: P) -> MediaResult<ProbeResult> {
    let data = std::fs::read(path.as_ref())?;
    probe_from_bytes(&data)
}

/// Result from probing an image.
#[derive(Debug, Clone)]
pub struct ProbeResult {
    pub width: u32,
    pub height: u32,
    pub format_name: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::codecs::png::PngEncoder;
    use image::{ExtendedColorType, ImageEncoder};

    /// Generate a minimal valid 2x2 RGBA PNG in memory.
    fn generate_test_png() -> Vec<u8> {
        let mut buf = Vec::new();
        let encoder = PngEncoder::new(&mut buf);
        let pixels: Vec<u8> = vec![
            255, 0, 0, 255, // red
            0, 255, 0, 255, // green
            0, 0, 255, 255, // blue
            255, 255, 0, 255, // yellow
        ];
        encoder
            .write_image(&pixels, 2, 2, ExtendedColorType::Rgba8)
            .unwrap();
        buf
    }

    #[test]
    fn test_decode_small_png() {
        let png_bytes = generate_test_png();
        let frame = decode_from_bytes(&png_bytes).unwrap();
        assert_eq!(frame.width, 2);
        assert_eq!(frame.height, 2);
        assert_eq!(frame.format, PixelFormat::Rgba8);
    }

    #[test]
    fn test_probe_from_bytes() {
        let png_bytes = generate_test_png();
        let result = probe_from_bytes(&png_bytes).unwrap();
        assert_eq!(result.width, 2);
        assert_eq!(result.height, 2);
        assert_eq!(result.format_name, "Png");
    }
}
