//! CPU-based effect implementations.

use media_compute::{ComputeEffect, EffectDesc, EffectParam};
use media_core::{CpuFrame, Frame, FrameStorage, MediaError, MediaResult};

/// Helper: get a mutable RGBA8 pixel slice from a CPU frame.
fn rgba_mut<'a>(cpu: &'a mut CpuFrame, x: u32, y: u32, _width: u32) -> &'a mut [u8] {
    let idx = (y as usize) * cpu.stride + (x as usize) * 4;
    &mut cpu.data[idx..idx + 4]
}

/// Helper: get an RGBA8 pixel reference from a CPU frame.
#[allow(dead_code)]
fn rgba_ref(cpu: &CpuFrame, x: u32, y: u32, _width: u32) -> [u8; 4] {
    let idx = (y as usize) * cpu.stride + (x as usize) * 4;
    let slice = &cpu.data[idx..idx + 4];
    [slice[0], slice[1], slice[2], slice[3]]
}

// ----------------------------------------------------------------
// Brightness
// ----------------------------------------------------------------
#[derive(Default)]
pub struct BrightnessEffect;

impl ComputeEffect for BrightnessEffect {
    fn name(&self) -> &str {
        "brightness"
    }

    fn supports_in_place(&self) -> bool {
        true
    }

    fn apply(&self, src: &Frame, desc: &EffectDesc) -> MediaResult<Frame> {
        let mut dst = clone_frame(src)?;
        self.apply_in_place(&mut dst, desc)?;
        Ok(dst)
    }

    fn apply_in_place(&self, frame: &mut Frame, desc: &EffectDesc) -> MediaResult<()> {
        let factor = desc
            .params
            .first()
            .and_then(|p| match p {
                EffectParam::Float(f) => Some(*f),
                EffectParam::Int(i) => Some(*i as f32),
                _ => None,
            })
            .unwrap_or(0.0);

        let (w, h) = (frame.width, frame.height);
        let cpu = frame
            .as_cpu_mut()
            .ok_or_else(|| MediaError::Other("Not a CPU frame".into()))?;

        for y in 0..h {
            for x in 0..w {
                let pixel = rgba_mut(cpu, x, y, w);
                for c in 0..3 {
                    let val = (pixel[c] as f32 + factor * 255.0).clamp(0.0, 255.0) as u8;
                    pixel[c] = val;
                }
            }
        }
        Ok(())
    }
}

// ----------------------------------------------------------------
// Contrast
// ----------------------------------------------------------------
#[derive(Default)]
pub struct ContrastEffect;

impl ComputeEffect for ContrastEffect {
    fn name(&self) -> &str {
        "contrast"
    }

    fn supports_in_place(&self) -> bool {
        true
    }

    fn apply(&self, src: &Frame, desc: &EffectDesc) -> MediaResult<Frame> {
        let mut dst = clone_frame(src)?;
        self.apply_in_place(&mut dst, desc)?;
        Ok(dst)
    }

    fn apply_in_place(&self, frame: &mut Frame, desc: &EffectDesc) -> MediaResult<()> {
        let factor = desc
            .params
            .first()
            .and_then(|p| match p {
                EffectParam::Float(f) => Some(*f),
                EffectParam::Int(i) => Some(*i as f32),
                _ => None,
            })
            .unwrap_or(1.0);

        // factor < 1 → less contrast, > 1 → more contrast
        let (w, h) = (frame.width, frame.height);
        let cpu = frame
            .as_cpu_mut()
            .ok_or_else(|| MediaError::Other("Not a CPU frame".into()))?;

        for y in 0..h {
            for x in 0..w {
                let pixel = rgba_mut(cpu, x, y, w);
                for c in 0..3 {
                    let centered = pixel[c] as f32 - 128.0;
                    let val = (centered * factor + 128.0).clamp(0.0, 255.0) as u8;
                    pixel[c] = val;
                }
            }
        }
        Ok(())
    }
}

// ----------------------------------------------------------------
// Grayscale (luminosity method)
// ----------------------------------------------------------------
#[derive(Default)]
pub struct GrayscaleEffect;

impl ComputeEffect for GrayscaleEffect {
    fn name(&self) -> &str {
        "grayscale"
    }

    fn supports_in_place(&self) -> bool {
        true
    }

    fn apply(&self, src: &Frame, desc: &EffectDesc) -> MediaResult<Frame> {
        let mut dst = clone_frame(src)?;
        self.apply_in_place(&mut dst, desc)?;
        Ok(dst)
    }

    fn apply_in_place(&self, frame: &mut Frame, _desc: &EffectDesc) -> MediaResult<()> {
        let (w, h) = (frame.width, frame.height);
        let cpu = frame
            .as_cpu_mut()
            .ok_or_else(|| MediaError::Other("Not a CPU frame".into()))?;

        for y in 0..h {
            for x in 0..w {
                let pixel = rgba_mut(cpu, x, y, w);
                // Luminosity weights: 0.2126 R + 0.7152 G + 0.0722 B
                let gray = (pixel[0] as f32 * 0.2126
                    + pixel[1] as f32 * 0.7152
                    + pixel[2] as f32 * 0.0722) as u8;
                pixel[0] = gray;
                pixel[1] = gray;
                pixel[2] = gray;
                // Alpha unchanged.
            }
        }
        Ok(())
    }
}

// ----------------------------------------------------------------
// Invert
// ----------------------------------------------------------------
#[derive(Default)]
pub struct InvertEffect;

impl ComputeEffect for InvertEffect {
    fn name(&self) -> &str {
        "invert"
    }

    fn supports_in_place(&self) -> bool {
        true
    }

    fn apply(&self, src: &Frame, desc: &EffectDesc) -> MediaResult<Frame> {
        let mut dst = clone_frame(src)?;
        self.apply_in_place(&mut dst, desc)?;
        Ok(dst)
    }

    fn apply_in_place(&self, frame: &mut Frame, _desc: &EffectDesc) -> MediaResult<()> {
        let (w, h) = (frame.width, frame.height);
        let cpu = frame
            .as_cpu_mut()
            .ok_or_else(|| MediaError::Other("Not a CPU frame".into()))?;

        for y in 0..h {
            for x in 0..w {
                let pixel = rgba_mut(cpu, x, y, w);
                for c in 0..3 {
                    pixel[c] = 255 - pixel[c];
                }
                // Alpha unchanged.
            }
        }
        Ok(())
    }
}

// ----------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------

/// Clone a CPU-backed Frame (deep copy of pixel data).
fn clone_frame(src: &Frame) -> MediaResult<Frame> {
    let cpu_src = src
        .as_cpu()
        .ok_or_else(|| MediaError::Other("Source is not CPU-backed".into()))?;

    let cpu_dst = CpuFrame {
        data: cpu_src.data.clone(),
        stride: cpu_src.stride,
    };

    Ok(Frame {
        width: src.width,
        height: src.height,
        format: src.format,
        timestamp: src.timestamp,
        color_space: src.color_space,
        storage: FrameStorage::Cpu(cpu_dst),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use media_compute::EffectKind;
    use media_core::PixelFormat;

    fn make_test_frame() -> Frame {
        let mut frame = Frame::new_cpu(2, 2, PixelFormat::Rgba8).unwrap();
        // Set specific pixel values
        let cpu = frame.as_cpu_mut().unwrap();
        // pixel (0,0): R=100, G=150, B=200, A=255
        cpu.data[0..4].copy_from_slice(&[100, 150, 200, 255]);
        // pixel (1,0)
        cpu.data[4..8].copy_from_slice(&[10, 20, 30, 255]);
        // pixel (0,1)
        cpu.data[8..12].copy_from_slice(&[200, 100, 50, 255]);
        // pixel (1,1)
        cpu.data[12..16].copy_from_slice(&[255, 255, 255, 255]);
        frame
    }

    #[test]
    fn test_brightness() {
        let frame = make_test_frame();
        let effect = BrightnessEffect;
        let desc = EffectDesc::new(EffectKind::Brightness).with(0.5f32);
        let result = effect.apply(&frame, &desc).unwrap();
        let cpu = result.as_cpu().unwrap();
        // pixel (0,0): R=100 + 0.5 * 255.0 = 227.5 → 227
        assert_eq!(cpu.data[0], 227);
    }

    #[test]
    fn test_grayscale() {
        let frame = make_test_frame();
        let effect = GrayscaleEffect;
        let desc = EffectDesc::new(EffectKind::Grayscale);
        let result = effect.apply(&frame, &desc).unwrap();
        let cpu = result.as_cpu().unwrap();
        // All three channels should be equal.
        assert_eq!(cpu.data[0], cpu.data[1]);
        assert_eq!(cpu.data[1], cpu.data[2]);
    }

    #[test]
    fn test_invert() {
        let frame = make_test_frame();
        let effect = InvertEffect;
        let desc = EffectDesc::new(EffectKind::Invert);
        let result = effect.apply(&frame, &desc).unwrap();
        let cpu = result.as_cpu().unwrap();
        // pixel (0,0): R=100 → 155
        assert_eq!(cpu.data[0], 155);
        assert_eq!(cpu.data[1], 105);
        assert_eq!(cpu.data[2], 55);
        // Alpha unchanged
        assert_eq!(cpu.data[3], 255);
    }

    #[test]
    fn test_contrast() {
        let frame = make_test_frame();
        let effect = ContrastEffect;
        let desc = EffectDesc::new(EffectKind::Contrast).with(2.0f32);
        let result = effect.apply(&frame, &desc).unwrap();
        let cpu = result.as_cpu().unwrap();
        // pixel (0,0): R = (100-128)*2 + 128 = 72
        assert_eq!(cpu.data[0], 72);
        // pixel (1,1): R = (255-128)*2 + 128 = 255
        assert_eq!(cpu.data[12], 255);
    }

    #[test]
    fn test_supports_in_place() {
        let effect = BrightnessEffect;
        assert!(effect.supports_in_place());
    }

    #[test]
    fn test_effect_names() {
        assert_eq!(BrightnessEffect.name(), "brightness");
        assert_eq!(GrayscaleEffect.name(), "grayscale");
        assert_eq!(InvertEffect.name(), "invert");
        assert_eq!(ContrastEffect.name(), "contrast");
    }
}
