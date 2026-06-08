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
// Blur (3×3 box blur)
// ----------------------------------------------------------------
#[derive(Default)]
pub struct BlurEffect;

impl ComputeEffect for BlurEffect {
    fn name(&self) -> &str {
        "blur"
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

        // Clone source so we read clean values while writing back.
        let src_data = cpu.data.clone();
        let stride = cpu.stride;

        for y in 0..h {
            for x in 0..w {
                let pixel = rgba_mut(cpu, x, y, w);
                let (r, g, b) = box3_blur_pixel(&src_data, stride, x, y, w, h);
                pixel[0] = r;
                pixel[1] = g;
                pixel[2] = b;
                // Alpha unchanged.
            }
        }
        Ok(())
    }
}

// ----------------------------------------------------------------
// Sharpen (unsharp mask)
// ----------------------------------------------------------------
#[derive(Default)]
pub struct SharpenEffect;

impl ComputeEffect for SharpenEffect {
    fn name(&self) -> &str {
        "sharpen"
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
        let strength = desc
            .params
            .first()
            .and_then(|p| match p {
                EffectParam::Float(f) => Some(*f),
                EffectParam::Int(i) => Some(*i as f32),
                _ => None,
            })
            .unwrap_or(1.0);

        let (w, h) = (frame.width, frame.height);
        let cpu = frame
            .as_cpu_mut()
            .ok_or_else(|| MediaError::Other("Not a CPU frame".into()))?;

        // Clone source for computing the blur.
        let src_data = cpu.data.clone();
        let stride = cpu.stride;

        for y in 0..h {
            for x in 0..w {
                let pixel = rgba_mut(cpu, x, y, w);
                let (blur_r, blur_g, blur_b) = box3_blur_pixel(&src_data, stride, x, y, w, h);

                let src_idx = y as usize * stride + x as usize * 4;
                let orig_r = src_data[src_idx] as f32;
                let orig_g = src_data[src_idx + 1] as f32;
                let orig_b = src_data[src_idx + 2] as f32;

                pixel[0] = (orig_r + strength * (orig_r - blur_r as f32)).clamp(0.0, 255.0) as u8;
                pixel[1] = (orig_g + strength * (orig_g - blur_g as f32)).clamp(0.0, 255.0) as u8;
                pixel[2] = (orig_b + strength * (orig_b - blur_b as f32)).clamp(0.0, 255.0) as u8;
                // Alpha unchanged.
            }
        }
        Ok(())
    }
}

// ----------------------------------------------------------------
// Sobel Edge Detection
// ----------------------------------------------------------------
#[derive(Default)]
pub struct SobelEdgeDetectEffect;

impl ComputeEffect for SobelEdgeDetectEffect {
    fn name(&self) -> &str {
        "edge_detect"
    }

    fn supports_in_place(&self) -> bool {
        false
    }

    fn apply(&self, src: &Frame, _desc: &EffectDesc) -> MediaResult<Frame> {
        let cpu_src = src
            .as_cpu()
            .ok_or_else(|| MediaError::Other("Not a CPU frame".into()))?;

        let (w, h) = (src.width, src.height);
        let stride = cpu_src.stride;
        let src_data = &cpu_src.data;

        // Grayscale the source first.
        let mut gray = vec![0u8; (h as usize) * stride];
        for y in 0..h {
            for x in 0..w {
                let idx = y as usize * stride + x as usize * 4;
                let r = src_data[idx] as f32;
                let g = src_data[idx + 1] as f32;
                let b = src_data[idx + 2] as f32;
                let luma = (r * 0.2126 + g * 0.7152 + b * 0.0722) as u8;
                gray[y as usize * stride + x as usize] = luma;
            }
        }

        // Sobel X and Y 3×3 kernels.
        let sobel_x: [[i32; 3]; 3] = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
        let sobel_y: [[i32; 3]; 3] = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

        let mut dst = clone_frame(src)?;
        let cpu_dst = dst.as_cpu_mut().unwrap();

        for y in 0..h {
            for x in 0..w {
                let mut gx = 0i32;
                let mut gy = 0i32;

                for ky in 0..3 {
                    for kx in 0..3 {
                        let nx = x as i32 + (kx as i32 - 1);
                        let ny = y as i32 + (ky as i32 - 1);
                        if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                            let pval = gray[ny as usize * stride + nx as usize] as i32;
                            gx += pval * sobel_x[ky][kx];
                            gy += pval * sobel_y[ky][kx];
                        }
                    }
                }

                let mag = ((gx * gx + gy * gy) as f32).sqrt().clamp(0.0, 255.0) as u8;
                let pixel = rgba_mut(cpu_dst, x, y, w);
                pixel[0] = mag;
                pixel[1] = mag;
                pixel[2] = mag;
                // Alpha unchanged.
            }
        }
        Ok(dst)
    }
}

// ----------------------------------------------------------------
// Sepia
// ----------------------------------------------------------------
#[derive(Default)]
pub struct SepiaEffect;

impl ComputeEffect for SepiaEffect {
    fn name(&self) -> &str {
        "sepia"
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
                let r = pixel[0] as f32;
                let g = pixel[1] as f32;
                let b = pixel[2] as f32;

                let out_r = (r * 0.393 + g * 0.769 + b * 0.189).clamp(0.0, 255.0) as u8;
                let out_g = (r * 0.349 + g * 0.686 + b * 0.168).clamp(0.0, 255.0) as u8;
                let out_b = (r * 0.272 + g * 0.534 + b * 0.131).clamp(0.0, 255.0) as u8;
                pixel[0] = out_r;
                pixel[1] = out_g;
                pixel[2] = out_b;
                // Alpha unchanged.
            }
        }
        Ok(())
    }
}

// ----------------------------------------------------------------
// Threshold
// ----------------------------------------------------------------
#[derive(Default)]
pub struct ThresholdEffect;

impl ComputeEffect for ThresholdEffect {
    fn name(&self) -> &str {
        "threshold"
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
        let threshold = desc
            .params
            .first()
            .and_then(|p| match p {
                EffectParam::Float(f) => Some((*f * 255.0) as u8),
                EffectParam::Int(i) => Some((*i).clamp(0, 255) as u8),
                EffectParam::U32(u) => Some((*u).clamp(0, 255) as u8),
            })
            .unwrap_or(128);

        let (w, h) = (frame.width, frame.height);
        let cpu = frame
            .as_cpu_mut()
            .ok_or_else(|| MediaError::Other("Not a CPU frame".into()))?;

        for y in 0..h {
            for x in 0..w {
                let pixel = rgba_mut(cpu, x, y, w);
                let luma = (pixel[0] as u16 + pixel[1] as u16 + pixel[2] as u16) / 3;
                let val: u8 = if luma > threshold as u16 { 255 } else { 0 };
                pixel[0] = val;
                pixel[1] = val;
                pixel[2] = val;
                // Alpha unchanged.
            }
        }
        Ok(())
    }
}

// ----------------------------------------------------------------
// Box Blur (separable O(1), configurable radius)
// ----------------------------------------------------------------
pub struct BoxBlurEffect {
    /// Default radius when no parameter is given.
    pub default_radius: u32,
}

impl Default for BoxBlurEffect {
    fn default() -> Self {
        BoxBlurEffect { default_radius: 3 }
    }
}

impl ComputeEffect for BoxBlurEffect {
    fn name(&self) -> &str {
        "box_blur"
    }

    fn supports_in_place(&self) -> bool {
        false
    }

    fn apply(&self, src: &Frame, desc: &EffectDesc) -> MediaResult<Frame> {
        let radius = desc
            .params
            .first()
            .and_then(|p| match p {
                EffectParam::U32(u) => Some(*u),
                EffectParam::Int(i) => Some((*i).clamp(0, 255) as u32),
                _ => None,
            })
            .unwrap_or(self.default_radius)
            .max(1);

        let cpu_src = src
            .as_cpu()
            .ok_or_else(|| MediaError::Other("Not a CPU frame".into()))?;

        let (w, h) = (src.width, src.height);
        let stride = cpu_src.stride;

        // Horizontal pass: for each row, compute prefix-sum per channel over 4-byte RGBA.
        let horiz = cpu_src.data.clone();
        // We'll store horizontal blur result in a temporary buffer.
        let mut tmp = vec![0u8; cpu_src.data.len()];

        for y in 0..h {
            let row_start = y as usize * stride;

            // Build prefix sums for R, G, B channels across the row.
            let mut ps_r = vec![0u32; w as usize + 1];
            let mut ps_g = vec![0u32; w as usize + 1];
            let mut ps_b = vec![0u32; w as usize + 1];

            for x in 0..w as usize {
                let idx = row_start + x * 4;
                ps_r[x + 1] = ps_r[x] + horiz[idx] as u32;
                ps_g[x + 1] = ps_g[x] + horiz[idx + 1] as u32;
                ps_b[x + 1] = ps_b[x] + horiz[idx + 2] as u32;
            }

            // Apply horizontal blur using prefix sums for O(1) per pixel.
            let _diameter = (2 * radius + 1) as usize;
            for x in 0..w as usize {
                let x0 = if x >= radius as usize {
                    x - radius as usize
                } else {
                    0
                };
                let x1 = (x + radius as usize).min(w as usize - 1);
                let count = (x1 - x0 + 1) as u32;

                let sum_r = ps_r[x1 + 1] - ps_r[x0];
                let sum_g = ps_g[x1 + 1] - ps_g[x0];
                let sum_b = ps_b[x1 + 1] - ps_b[x0];

                let idx = row_start + x * 4;
                tmp[idx] = (sum_r / count) as u8;
                tmp[idx + 1] = (sum_g / count) as u8;
                tmp[idx + 2] = (sum_b / count) as u8;
                tmp[idx + 3] = horiz[idx + 3]; // copy alpha
            }
        }

        // Build output frame.
        let mut dst = clone_frame(src)?;
        let cpu_dst = dst.as_cpu_mut().unwrap();

        // Vertical pass: column prefix sums from tmp.
        for x in 0..w {
            // Build column prefix sums for each channel.
            let mut ps_r = vec![0u32; h as usize + 1];
            let mut ps_g = vec![0u32; h as usize + 1];
            let mut ps_b = vec![0u32; h as usize + 1];

            for y in 0..h as usize {
                let idx = y * stride + x as usize * 4;
                ps_r[y + 1] = ps_r[y] + tmp[idx] as u32;
                ps_g[y + 1] = ps_g[y] + tmp[idx + 1] as u32;
                ps_b[y + 1] = ps_b[y] + tmp[idx + 2] as u32;
            }

            let _diameter = (2 * radius + 1) as usize;
            for y in 0..h as usize {
                let y0 = if y >= radius as usize {
                    y - radius as usize
                } else {
                    0
                };
                let y1 = (y + radius as usize).min(h as usize - 1);
                let count = (y1 - y0 + 1) as u32;

                let sum_r = ps_r[y1 + 1] - ps_r[y0];
                let sum_g = ps_g[y1 + 1] - ps_g[y0];
                let sum_b = ps_b[y1 + 1] - ps_b[y0];

                let idx = y * stride + x as usize * 4;
                cpu_dst.data[idx] = (sum_r / count) as u8;
                cpu_dst.data[idx + 1] = (sum_g / count) as u8;
                cpu_dst.data[idx + 2] = (sum_b / count) as u8;
                // Alpha unchanged from source.
                let src_idx = y * stride + x as usize * 4;
                cpu_dst.data[idx + 3] = tmp[src_idx + 3];
            }
        }

        Ok(dst)
    }
}

// ----------------------------------------------------------------
// Emboss
// ----------------------------------------------------------------
#[derive(Default)]
pub struct EmbossEffect;

impl ComputeEffect for EmbossEffect {
    fn name(&self) -> &str {
        "emboss"
    }

    fn supports_in_place(&self) -> bool {
        false
    }

    fn apply(&self, src: &Frame, _desc: &EffectDesc) -> MediaResult<Frame> {
        let cpu_src = src
            .as_cpu()
            .ok_or_else(|| MediaError::Other("Not a CPU frame".into()))?;

        let (w, h) = (src.width, src.height);
        let stride = cpu_src.stride;
        let src_data = &cpu_src.data;

        // Emboss kernel:
        // [[-2, -1, 0],
        //  [-1,  1, 1],
        //  [ 0,  1, 2]]
        let kernel: [[i32; 3]; 3] = [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]];

        // First grayscale.
        let mut gray = vec![0u8; (h as usize) * stride];
        for y in 0..h {
            for x in 0..w {
                let idx = y as usize * stride + x as usize * 4;
                let r = src_data[idx] as f32;
                let g = src_data[idx + 1] as f32;
                let b = src_data[idx + 2] as f32;
                let luma = (r * 0.2126 + g * 0.7152 + b * 0.0722) as u8;
                gray[y as usize * stride + x as usize] = luma;
            }
        }

        let mut dst = clone_frame(src)?;
        let cpu_dst = dst.as_cpu_mut().unwrap();

        for y in 0..h {
            for x in 0..w {
                let mut acc = 0i32;
                for ky in 0..3 {
                    for kx in 0..3 {
                        let nx = x as i32 + (kx as i32 - 1);
                        let ny = y as i32 + (ky as i32 - 1);
                        if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                            let pval = gray[ny as usize * stride + nx as usize] as i32;
                            acc += pval * kernel[ky][kx];
                        }
                    }
                }
                // Bias by 128 for a raised effect.
                let val = (acc + 128).clamp(0, 255) as u8;
                let pixel = rgba_mut(cpu_dst, x, y, w);
                pixel[0] = val;
                pixel[1] = val;
                pixel[2] = val;
                // Alpha unchanged.
            }
        }
        Ok(dst)
    }
}

// ----------------------------------------------------------------
// Pixelate (mosaic)
// ----------------------------------------------------------------
#[derive(Default)]
pub struct PixelateEffect;

impl ComputeEffect for PixelateEffect {
    fn name(&self) -> &str {
        "pixelate"
    }

    fn supports_in_place(&self) -> bool {
        false
    }

    fn apply(&self, src: &Frame, desc: &EffectDesc) -> MediaResult<Frame> {
        let block_size = desc
            .params
            .first()
            .and_then(|p| match p {
                EffectParam::U32(u) => Some(*u),
                EffectParam::Int(i) => Some((*i).clamp(1, 256) as u32),
                _ => None,
            })
            .unwrap_or(8)
            .max(1);

        let cpu_src = src
            .as_cpu()
            .ok_or_else(|| MediaError::Other("Not a CPU frame".into()))?;

        let (w, h) = (src.width, src.height);
        let stride = cpu_src.stride;
        let src_data = &cpu_src.data;

        let mut dst = clone_frame(src)?;
        let cpu_dst = dst.as_cpu_mut().unwrap();

        for by in 0..((h + block_size - 1) / block_size) {
            for bx in 0..((w + block_size - 1) / block_size) {
                // Compute average color of the block.
                let start_x = bx * block_size;
                let start_y = by * block_size;
                let end_x = (start_x + block_size).min(w);
                let end_y = (start_y + block_size).min(h);

                let mut sum_r = 0u64;
                let mut sum_g = 0u64;
                let mut sum_b = 0u64;
                let mut count = 0u64;

                for py in start_y..end_y {
                    for px in start_x..end_x {
                        let idx = py as usize * stride + px as usize * 4;
                        sum_r += src_data[idx] as u64;
                        sum_g += src_data[idx + 1] as u64;
                        sum_b += src_data[idx + 2] as u64;
                        count += 1;
                    }
                }

                let avg_r = (sum_r / count) as u8;
                let avg_g = (sum_g / count) as u8;
                let avg_b = (sum_b / count) as u8;

                // Fill the block.
                for py in start_y..end_y {
                    for px in start_x..end_x {
                        let idx = py as usize * stride + px as usize * 4;
                        cpu_dst.data[idx] = avg_r;
                        cpu_dst.data[idx + 1] = avg_g;
                        cpu_dst.data[idx + 2] = avg_b;
                        cpu_dst.data[idx + 3] = src_data[idx + 3]; // alpha
                    }
                }
            }
        }
        Ok(dst)
    }
}

// ----------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------

/// Compute the 3×3 box-blurred RGB for a pixel, reading from `data`.
fn box3_blur_pixel(data: &[u8], stride: usize, x: u32, y: u32, w: u32, h: u32) -> (u8, u8, u8) {
    let mut sum_r = 0u32;
    let mut sum_g = 0u32;
    let mut sum_b = 0u32;
    let mut count = 0u32;

    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                let idx = ny as usize * stride + nx as usize * 4;
                sum_r += data[idx] as u32;
                sum_g += data[idx + 1] as u32;
                sum_b += data[idx + 2] as u32;
                count += 1;
            }
        }
    }

    (
        (sum_r / count) as u8,
        (sum_g / count) as u8,
        (sum_b / count) as u8,
    )
}

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
        assert_eq!(BlurEffect.name(), "blur");
        assert_eq!(SharpenEffect.name(), "sharpen");
    }

    #[test]
    fn test_blur_changes_pixels() {
        let frame = make_test_frame();
        let effect = BlurEffect;
        let desc = EffectDesc::new(EffectKind::Blur);
        let result = effect.apply(&frame, &desc).unwrap();
        let cpu = result.as_cpu().unwrap();

        // With a 2×2 image every pixel sees all 4 neighbours.
        // pixel (0,0): R = (100+10+200+255)/4 = 141
        assert_eq!(cpu.data[0], 141);
    }

    #[test]
    fn test_blur_uniform_unchanged() {
        let mut frame = make_test_frame();
        let cpu = frame.as_cpu_mut().unwrap();
        cpu.data.fill(128);

        let effect = BlurEffect;
        let desc = EffectDesc::new(EffectKind::Blur);
        let result = effect.apply(&frame, &desc).unwrap();
        let out = result.as_cpu().unwrap();
        assert!(
            out.data.iter().all(|&v| v == 128),
            "Blur of uniform field should stay uniform"
        );
    }

    #[test]
    fn test_sharpen_changes_pixels() {
        let frame = make_test_frame();
        let effect = SharpenEffect;
        let desc = EffectDesc::new(EffectKind::Sharpen).with(EffectParam::Float(1.0));
        let result = effect.apply(&frame, &desc).unwrap();
        let cpu = result.as_cpu().unwrap();

        // Blur of pixel (0,0): R = 141
        // sharpen 1.0: 100 + 1.0*(100-141) = 59
        assert_eq!(cpu.data[0], 59);
    }

    #[test]
    fn test_sharpen_zero_strength_identity() {
        let frame = make_test_frame();
        let effect = SharpenEffect;
        let desc = EffectDesc::new(EffectKind::Sharpen).with(EffectParam::Float(0.0));
        let result = effect.apply(&frame, &desc).unwrap();
        let out = result.as_cpu().unwrap();
        let orig = frame.as_cpu().unwrap();
        assert_eq!(out.data, orig.data, "Sharpen strength=0 should be identity");
    }

    #[test]
    fn test_sharpen_uniform_unchanged() {
        let mut frame = make_test_frame();
        let cpu = frame.as_cpu_mut().unwrap();
        cpu.data.fill(128);

        let effect = SharpenEffect;
        let desc = EffectDesc::new(EffectKind::Sharpen).with(EffectParam::Float(1.0));
        let result = effect.apply(&frame, &desc).unwrap();
        let out = result.as_cpu().unwrap();
        assert!(
            out.data.iter().all(|&v| v == 128),
            "Sharpen of uniform field should stay unchanged"
        );
    }

    // ----------------------------------------------------------------
    // New effect tests
    // ----------------------------------------------------------------

    #[test]
    fn test_edge_detect() {
        let frame = make_test_frame();
        let effect = SobelEdgeDetectEffect;
        let desc = EffectDesc::new(EffectKind::EdgeDetect);
        let result = effect.apply(&frame, &desc).unwrap();
        let cpu = result.as_cpu().unwrap();
        // With a 2×2 image each pixel sees all its neighbours — non-zero edges.
        // Just verify the output is valid RGBA.
        assert_eq!(cpu.data.len(), 16);
        // Alpha should be preserved (255).
        assert_eq!(cpu.data[3], 255);
    }

    #[test]
    fn test_edge_detect_uniform_interior_zero() {
        // 4×4 uniform image → center pixels have full 3×3 neighborhood → zero gradient.
        let mut frame = Frame::new_cpu(4, 4, PixelFormat::Rgba8).unwrap();
        frame.as_cpu_mut().unwrap().data.fill(128);

        let effect = SobelEdgeDetectEffect;
        let desc = EffectDesc::new(EffectKind::EdgeDetect);
        let result = effect.apply(&frame, &desc).unwrap();
        let cpu = result.as_cpu().unwrap();
        let stride = cpu.stride;
        // Pixel (1,1) interior → zero.
        let idx = 1 * stride + 1 * 4;
        assert_eq!(cpu.data[idx], 0);
        // Border pixels may be non-zero (partial Sobel window) — that's expected.
    }

    #[test]
    fn test_sepia() {
        let frame = make_test_frame();
        let effect = SepiaEffect;
        let desc = EffectDesc::new(EffectKind::Sepia);
        let result = effect.apply(&frame, &desc).unwrap();
        let cpu = result.as_cpu().unwrap();
        // pixel (0,0): R=100, G=150, B=200
        // R = 100*0.393 + 150*0.769 + 200*0.189 = 39.3 + 115.35 + 37.8 = 192.45
        assert_eq!(cpu.data[0], 192);
        assert_eq!(cpu.data[3], 255); // alpha preserved
    }

    #[test]
    fn test_threshold_above() {
        let frame = make_test_frame();
        let effect = ThresholdEffect;
        let desc = EffectDesc::new(EffectKind::Threshold).with(EffectParam::Float(0.2)); // threshold = 51
        let result = effect.apply(&frame, &desc).unwrap();
        let cpu = result.as_cpu().unwrap();
        // pixel (0,0): luma ≈ 100*0.2126 + 150*0.7152 + 200*0.0722 = 147.7 > 51 → 255
        assert_eq!(cpu.data[0], 255);
        assert_eq!(cpu.data[1], 255);
        assert_eq!(cpu.data[2], 255);
        assert_eq!(cpu.data[3], 255);
    }

    #[test]
    fn test_threshold_below() {
        let mut frame = make_test_frame();
        let cpu = frame.as_cpu_mut().unwrap();
        // Set all to very dark.
        cpu.data.fill(10);
        cpu.data[3] = 255; // alpha

        let effect = ThresholdEffect;
        let desc = EffectDesc::new(EffectKind::Threshold).with(EffectParam::Float(0.5)); // threshold = 127
        let result = effect.apply(&frame, &desc).unwrap();
        let out = result.as_cpu().unwrap();
        // All luma below threshold → 0
        assert_eq!(out.data[0], 0);
        assert_eq!(out.data[1], 0);
        assert_eq!(out.data[2], 0);
    }

    #[test]
    fn test_box_blur_separable() {
        let frame = make_test_frame();
        let effect = BoxBlurEffect::default();
        let desc = EffectDesc::new(EffectKind::BoxBlur).with(EffectParam::U32(1));
        let result = effect.apply(&frame, &desc).unwrap();
        let cpu = result.as_cpu().unwrap();
        // With radius 1 and a 2×2 image, each pixel averages all 4 neighbours.
        // pixel (0,0): R = (100+10+200+255)/4 = 141
        assert_eq!(cpu.data[0], 141);
        assert_eq!(cpu.data[3], 255); // alpha preserved
    }

    #[test]
    fn test_box_blur_uniform_unchanged() {
        let mut frame = make_test_frame();
        let cpu = frame.as_cpu_mut().unwrap();
        cpu.data.fill(128);

        let effect = BoxBlurEffect::default();
        let desc = EffectDesc::new(EffectKind::BoxBlur).with(EffectParam::U32(5));
        let result = effect.apply(&frame, &desc).unwrap();
        let out = result.as_cpu().unwrap();
        assert!(
            out.data.iter().filter(|&&v| v != 128).count() == 0,
            "Blur of uniform field should stay unchanged"
        );
    }

    #[test]
    fn test_emboss() {
        let frame = make_test_frame();
        let effect = EmbossEffect;
        let desc = EffectDesc::new(EffectKind::Emboss);
        let result = effect.apply(&frame, &desc).unwrap();
        let cpu = result.as_cpu().unwrap();
        // Just verify output is valid with alpha preserved.
        assert_eq!(cpu.data.len(), 16);
        assert_eq!(cpu.data[3], 255);
    }

    #[test]
    fn test_pixelate() {
        let frame = make_test_frame();
        let effect = PixelateEffect;
        let desc = EffectDesc::new(EffectKind::Pixelate).with(EffectParam::U32(2));
        let result = effect.apply(&frame, &desc).unwrap();
        let cpu = result.as_cpu().unwrap();
        // With block_size=2 on a 2×2 image, it's one block.
        // All pixels should have the same color.
        assert_eq!(cpu.data[0..4], cpu.data[4..8]);
        assert_eq!(cpu.data[0..4], cpu.data[8..12]);
        assert_eq!(cpu.data[0..4], cpu.data[12..16]);
    }

    #[test]
    fn test_new_effect_names() {
        assert_eq!(SobelEdgeDetectEffect.name(), "edge_detect");
        assert_eq!(SepiaEffect.name(), "sepia");
        assert_eq!(ThresholdEffect.name(), "threshold");
        assert_eq!(BoxBlurEffect::default().name(), "box_blur");
        assert_eq!(EmbossEffect.name(), "emboss");
        assert_eq!(PixelateEffect.name(), "pixelate");
    }
}
