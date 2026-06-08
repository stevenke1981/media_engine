//! CUDA effect implementations (brightness, contrast, grayscale, invert).

use crate::buffer::CudaBuffer;
use crate::module::CudaFunction;
use media_compute::{ComputeBuffer, ComputeEffect, EffectDesc, EffectKind};
use media_core::error::{MediaError, MediaResult};
use media_core::Frame;

/// Wrapper for a launchable CUDA kernel function acting as an effect.
pub struct CudaKernelEffect {
    name: String,
    kind: EffectKind,
    func: CudaFunction,
}

impl CudaKernelEffect {
    pub fn new(name: String, kind: EffectKind, func: CudaFunction) -> Self {
        CudaKernelEffect { name, kind, func }
    }
}

impl ComputeEffect for CudaKernelEffect {
    fn name(&self) -> &str {
        &self.name
    }

    fn apply(&self, src: &Frame, desc: &EffectDesc) -> MediaResult<Frame> {
        let mut dst = src.clone();

        let width = src.width;
        let height = src.height;
        let cpu = src
            .as_cpu()
            .ok_or_else(|| MediaError::Other("CUDA effect requires a CPU-backed frame".into()))?;
        let stride = cpu.stride as u32;
        let data_len = cpu.data.len();

        // ── Allocate device buffer and copy data ──────────────
        let mut dev_buf = CudaBuffer::new(data_len)?;
        dev_buf.host.copy_from_slice(&cpu.data);
        dev_buf.copy_to_device()?;

        // ── Launch kernel ─────────────────────────────────────
        let grid_x = (width + 15) / 16;
        let grid_y = (height + 15) / 16;

        match &self.kind {
            EffectKind::Brightness => {
                let factor = extract_float(&desc.params, 0, 0.1);
                launch_brightness(
                    &self.func, &dev_buf, width, height, stride, grid_x, grid_y, factor,
                )?;
            }
            EffectKind::Contrast => {
                let factor = extract_float(&desc.params, 0, 1.5);
                launch_contrast(
                    &self.func, &dev_buf, width, height, stride, grid_x, grid_y, factor,
                )?;
            }
            EffectKind::Grayscale => {
                launch_grayscale(&self.func, &dev_buf, width, height, stride, grid_x, grid_y)?;
            }
            EffectKind::Invert => {
                launch_invert(&self.func, &dev_buf, width, height, stride, grid_x, grid_y)?;
            }
            EffectKind::Blur => {
                launch_blur(&self.func, &dev_buf, width, height, stride, grid_x, grid_y)?;
            }
            EffectKind::Sharpen => {
                let strength = extract_float(&desc.params, 0, 0.5);
                launch_sharpen(
                    &self.func, &dev_buf, width, height, stride, grid_x, grid_y, strength,
                )?;
            }
            EffectKind::Custom(_) => {
                return Err(MediaError::Other(format!(
                    "Custom effect '{:?}' not implemented in CUDA backend",
                    desc.kind,
                )));
            }
            _ => {
                return Err(MediaError::Other(format!(
                    "Effect '{:?}' not implemented in CUDA backend",
                    desc.kind,
                )));
            }
        }

        // ── Read back ─────────────────────────────────────────
        let result = dev_buf.read_into_vec()?;
        if let Some(cpu) = dst.as_cpu_mut() {
            cpu.data = result;
        }
        Ok(dst)
    }

    fn supports_in_place(&self) -> bool {
        true
    }

    fn apply_in_place(&self, frame: &mut Frame, desc: &EffectDesc) -> MediaResult<()> {
        let width = frame.width;
        let height = frame.height;
        let cpu = frame
            .as_cpu()
            .ok_or_else(|| MediaError::Other("CUDA effect requires a CPU-backed frame".into()))?;
        let stride = cpu.stride as u32;
        let data_len = cpu.data.len();

        let mut dev_buf = CudaBuffer::new(data_len)?;
        dev_buf.host.copy_from_slice(&cpu.data);
        dev_buf.copy_to_device()?;

        let grid_x = (width + 15) / 16;
        let grid_y = (height + 15) / 16;

        match &self.kind {
            EffectKind::Brightness => {
                let factor = extract_float(&desc.params, 0, 0.1);
                launch_brightness(
                    &self.func, &dev_buf, width, height, stride, grid_x, grid_y, factor,
                )?;
            }
            EffectKind::Contrast => {
                let factor = extract_float(&desc.params, 0, 1.5);
                launch_contrast(
                    &self.func, &dev_buf, width, height, stride, grid_x, grid_y, factor,
                )?;
            }
            EffectKind::Grayscale => {
                launch_grayscale(&self.func, &dev_buf, width, height, stride, grid_x, grid_y)?;
            }
            EffectKind::Invert => {
                launch_invert(&self.func, &dev_buf, width, height, stride, grid_x, grid_y)?;
            }
            EffectKind::Blur => {
                launch_blur(&self.func, &dev_buf, width, height, stride, grid_x, grid_y)?;
            }
            EffectKind::Sharpen => {
                let strength = extract_float(&desc.params, 0, 0.5);
                launch_sharpen(
                    &self.func, &dev_buf, width, height, stride, grid_x, grid_y, strength,
                )?;
            }
            EffectKind::Custom(_) => {
                return Err(MediaError::Other(format!(
                    "Custom effect '{:?}' not implemented in CUDA backend",
                    desc.kind,
                )));
            }
            _ => {
                return Err(MediaError::Other(format!(
                    "Effect '{:?}' not implemented in CUDA backend",
                    desc.kind,
                )));
            }
        }

        let result = dev_buf.read_into_vec()?;
        if let Some(cpu) = frame.as_cpu_mut() {
            cpu.data = result;
        }
        Ok(())
    }
}

// ── Helper launch functions ────────────────────────────────────

fn extract_float(params: &[media_compute::EffectParam], index: usize, default: f32) -> f32 {
    params
        .get(index)
        .map(|p| match p {
            media_compute::EffectParam::Float(v) => *v,
            media_compute::EffectParam::Int(v) => *v as f32,
            media_compute::EffectParam::U32(v) => *v as f32,
        })
        .unwrap_or(default)
}

fn launch_brightness(
    func: &CudaFunction,
    buf: &CudaBuffer,
    width: u32,
    height: u32,
    stride: u32,
    grid_x: u32,
    grid_y: u32,
    factor: f32,
) -> MediaResult<()> {
    let args: [*mut std::ffi::c_void; 5] = [
        &buf.device_ptr() as *const u64 as *mut _,
        &(width as i32) as *const i32 as *mut _,
        &(height as i32) as *const i32 as *mut _,
        &(stride as i32) as *const i32 as *mut _,
        &factor as *const f32 as *mut _,
    ];
    func.launch(grid_x, grid_y, 1, 16, 16, 1, &args)
}

fn launch_contrast(
    func: &CudaFunction,
    buf: &CudaBuffer,
    width: u32,
    height: u32,
    stride: u32,
    grid_x: u32,
    grid_y: u32,
    factor: f32,
) -> MediaResult<()> {
    let args: [*mut std::ffi::c_void; 5] = [
        &buf.device_ptr() as *const u64 as *mut _,
        &(width as i32) as *const i32 as *mut _,
        &(height as i32) as *const i32 as *mut _,
        &(stride as i32) as *const i32 as *mut _,
        &factor as *const f32 as *mut _,
    ];
    func.launch(grid_x, grid_y, 1, 16, 16, 1, &args)
}

fn launch_grayscale(
    func: &CudaFunction,
    buf: &CudaBuffer,
    width: u32,
    height: u32,
    stride: u32,
    grid_x: u32,
    grid_y: u32,
) -> MediaResult<()> {
    let args: [*mut std::ffi::c_void; 4] = [
        &buf.device_ptr() as *const u64 as *mut _,
        &(width as i32) as *const i32 as *mut _,
        &(height as i32) as *const i32 as *mut _,
        &(stride as i32) as *const i32 as *mut _,
    ];
    func.launch(grid_x, grid_y, 1, 16, 16, 1, &args)
}

fn launch_blur(
    func: &CudaFunction,
    buf: &CudaBuffer,
    width: u32,
    height: u32,
    stride: u32,
    grid_x: u32,
    grid_y: u32,
) -> MediaResult<()> {
    let args: [*mut std::ffi::c_void; 4] = [
        &buf.device_ptr() as *const u64 as *mut _,
        &(width as i32) as *const i32 as *mut _,
        &(height as i32) as *const i32 as *mut _,
        &(stride as i32) as *const i32 as *mut _,
    ];
    func.launch(grid_x, grid_y, 1, 16, 16, 1, &args)
}

fn launch_sharpen(
    func: &CudaFunction,
    buf: &CudaBuffer,
    width: u32,
    height: u32,
    stride: u32,
    grid_x: u32,
    grid_y: u32,
    strength: f32,
) -> MediaResult<()> {
    let args: [*mut std::ffi::c_void; 5] = [
        &buf.device_ptr() as *const u64 as *mut _,
        &(width as i32) as *const i32 as *mut _,
        &(height as i32) as *const i32 as *mut _,
        &(stride as i32) as *const i32 as *mut _,
        &strength as *const f32 as *mut _,
    ];
    func.launch(grid_x, grid_y, 1, 16, 16, 1, &args)
}

fn launch_invert(
    func: &CudaFunction,
    buf: &CudaBuffer,
    width: u32,
    height: u32,
    stride: u32,
    grid_x: u32,
    grid_y: u32,
) -> MediaResult<()> {
    let args: [*mut std::ffi::c_void; 4] = [
        &buf.device_ptr() as *const u64 as *mut _,
        &(width as i32) as *const i32 as *mut _,
        &(height as i32) as *const i32 as *mut _,
        &(stride as i32) as *const i32 as *mut _,
    ];
    func.launch(grid_x, grid_y, 1, 16, 16, 1, &args)
}
