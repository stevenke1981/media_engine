//! Format conversion utilities for CPU frames.

use media_core::{CpuFrame, Frame, FrameStorage, MediaError, MediaResult, PixelFormat};

/// Convert a CPU Frame from one packed format to another.
/// Only supports packed → packed conversions (e.g. RGBA8 → RGB8, RGB8 → RGBA8).
pub fn convert_cpu_format(frame: &Frame, target_format: PixelFormat) -> MediaResult<Frame> {
    let src = frame
        .as_cpu()
        .ok_or(MediaError::Other("Source frame is not CPU-backed".into()))?;

    let dst_bpp = target_format
        .bytes_per_pixel()
        .ok_or(MediaError::UnsupportedPixelFormat(target_format))?;

    let pixel_count = (frame.width as usize) * (frame.height as usize);
    let mut dst_data = vec![0u8; pixel_count * dst_bpp];
    let dst_stride = frame.width as usize * dst_bpp;

    match (frame.format, target_format) {
        // RGBA8 → RGB8: drop alpha
        (PixelFormat::Rgba8, PixelFormat::Rgb8) => {
            for y in 0..frame.height as usize {
                let src_row = &src.data[y * src.stride..];
                let dst_row = &mut dst_data[y * dst_stride..];
                for x in 0..frame.width as usize {
                    let si = x * 4;
                    let di = x * 3;
                    dst_row[di..di + 3].copy_from_slice(&src_row[si..si + 3]);
                }
            }
        }
        // RGB8 → RGBA8: add 255 alpha
        (PixelFormat::Rgb8, PixelFormat::Rgba8) => {
            for y in 0..frame.height as usize {
                let src_row = &src.data[y * src.stride..];
                let dst_row = &mut dst_data[y * dst_stride..];
                for x in 0..frame.width as usize {
                    let si = x * 3;
                    let di = x * 4;
                    dst_row[di..di + 3].copy_from_slice(&src_row[si..si + 3]);
                    dst_row[di + 3] = 255;
                }
            }
        }
        // RGBA8 → BGRA8: swap R and B
        (PixelFormat::Rgba8, PixelFormat::Bgra8) => {
            for y in 0..frame.height as usize {
                let src_row = &src.data[y * src.stride..];
                let dst_row = &mut dst_data[y * dst_stride..];
                for x in 0..frame.width as usize {
                    let i = x * 4;
                    dst_row[i] = src_row[i + 2]; // B
                    dst_row[i + 1] = src_row[i + 1]; // G
                    dst_row[i + 2] = src_row[i]; // R
                    dst_row[i + 3] = src_row[i + 3]; // A
                }
            }
        }
        _ => {
            return Err(MediaError::Other(format!(
                "Unsupported format conversion: {:?} -> {:?}",
                frame.format, target_format
            )))
        }
    }

    let cpu = CpuFrame {
        data: dst_data,
        stride: dst_stride,
    };

    Ok(Frame {
        width: frame.width,
        height: frame.height,
        format: target_format,
        timestamp: frame.timestamp,
        color_space: frame.color_space,
        storage: FrameStorage::Cpu(cpu),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use media_core::Frame;

    #[test]
    fn test_rgba8_to_rgb8() {
        let mut src = Frame::new_cpu(2, 2, PixelFormat::Rgba8).unwrap();
        // Set pixel 0 to red with alpha.
        if let Some(cpu) = src.as_cpu_mut() {
            cpu.data[0] = 255;
            cpu.data[1] = 0;
            cpu.data[2] = 0;
            cpu.data[3] = 255;
        }

        let dst = convert_cpu_format(&src, PixelFormat::Rgb8).unwrap();
        assert_eq!(dst.format, PixelFormat::Rgb8);

        let dst_cpu = dst.as_cpu().unwrap();
        // First pixel should have 3 bytes: R, G, B
        assert_eq!(dst_cpu.data[0], 255);
        assert_eq!(dst_cpu.data[1], 0);
        assert_eq!(dst_cpu.data[2], 0);
    }

    #[test]
    fn test_rgb8_to_rgba8() {
        let src = Frame::new_cpu(2, 2, PixelFormat::Rgb8).unwrap();
        let dst = convert_cpu_format(&src, PixelFormat::Rgba8).unwrap();
        assert_eq!(dst.format, PixelFormat::Rgba8);
        let dst_cpu = dst.as_cpu().unwrap();
        // Alpha should be 255.
        assert_eq!(dst_cpu.data[3], 255);
    }

    #[test]
    fn test_rgba8_to_bgra8() {
        let mut src = Frame::new_cpu(1, 1, PixelFormat::Rgba8).unwrap();
        if let Some(cpu) = src.as_cpu_mut() {
            cpu.data[0] = 255; // R
            cpu.data[1] = 128; // G
            cpu.data[2] = 64; // B
            cpu.data[3] = 255; // A
        }

        let dst = convert_cpu_format(&src, PixelFormat::Bgra8).unwrap();
        let dst_cpu = dst.as_cpu().unwrap();
        assert_eq!(dst_cpu.data[0], 64); // B
        assert_eq!(dst_cpu.data[1], 128); // G
        assert_eq!(dst_cpu.data[2], 255); // R
        assert_eq!(dst_cpu.data[3], 255); // A
    }
}
