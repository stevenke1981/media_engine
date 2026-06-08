use crate::backend::BackendKind;
use crate::error::{MediaError, MediaResult};
use crate::pixel_format::PixelFormat;

/// CPU-side frame data.
#[derive(Debug, Clone)]
pub struct CpuFrame {
    /// Raw pixel data (packed RGBA8, etc.).
    pub data: Vec<u8>,
    /// Row stride in bytes.
    pub stride: usize,
}

impl CpuFrame {
    /// Create a new CPU frame with given dimensions and format.
    /// Allocates buffer and sets default stride (width × bpp).
    pub fn new(width: u32, height: u32, format: PixelFormat) -> MediaResult<Self> {
        let bpp = format
            .bytes_per_pixel()
            .ok_or_else(|| MediaError::UnsupportedPixelFormat(format))?;
        let stride = width as usize * bpp;
        let size = stride * height as usize;
        Ok(CpuFrame {
            data: vec![0u8; size],
            stride,
        })
    }

    /// Returns the number of bytes per row.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Returns a mutable byte slice for a given row.
    pub fn row_mut(&mut self, row: usize) -> &mut [u8] {
        let start = row * self.stride;
        &mut self.data[start..start + self.stride]
    }

    /// Returns a byte slice for a given row.
    pub fn row(&self, row: usize) -> &[u8] {
        let start = row * self.stride;
        &self.data[start..start + self.stride]
    }
}

/// Opaque handle type for device-side textures (WGPU).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub u64);

/// Device-side CUDA frame reference.
#[derive(Debug, Clone)]
pub struct CudaFrame {
    pub device_id: u32,
    /// Device pointer (raw CUDA pointer, wrapped safely).
    pub ptr: u64,
    /// Pitch / stride in bytes.
    pub pitch: usize,
    /// Allocated size in bytes.
    pub size: usize,
}

/// WGPU-side frame reference.
#[derive(Debug, Clone)]
pub struct WgpuFrame {
    pub texture_id: TextureHandle,
    pub width: u32,
    pub height: u32,
}

/// Externally-owned frame (from plugins, interop, etc.).
#[derive(Debug, Clone)]
pub struct ExternalFrame {
    pub backend: BackendKind,
    pub opaque: u64,
    pub size: usize,
}

/// Enum over all possible frame storage locations.
#[derive(Debug, Clone)]
pub enum FrameStorage {
    Cpu(CpuFrame),
    Cuda(CudaFrame),
    Wgpu(WgpuFrame),
    External(ExternalFrame),
}

impl FrameStorage {
    /// Which backend owns this storage.
    pub fn backend_kind(&self) -> BackendKind {
        match self {
            FrameStorage::Cpu(_) => BackendKind::Cpu,
            FrameStorage::Cuda(_) => BackendKind::Cuda,
            FrameStorage::Wgpu(_) => BackendKind::Wgpu,
            FrameStorage::External(ext) => ext.backend,
        }
    }
}

/// Optional color space metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    Srgb,
    Rec709,
    Rec2020,
    Linear,
}

/// The central media frame type.
#[derive(Debug, Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub timestamp: Option<f64>,
    pub color_space: Option<ColorSpace>,
    pub storage: FrameStorage,
}

impl Frame {
    /// Create a new CPU-backed frame with zeroed data.
    pub fn new_cpu(width: u32, height: u32, format: PixelFormat) -> MediaResult<Self> {
        let cpu = CpuFrame::new(width, height, format)?;
        Ok(Frame {
            width,
            height,
            format,
            timestamp: None,
            color_space: None,
            storage: FrameStorage::Cpu(cpu),
        })
    }

    /// Returns a reference to the CPU frame data if this is a CPU frame.
    pub fn as_cpu(&self) -> Option<&CpuFrame> {
        match &self.storage {
            FrameStorage::Cpu(cpu) => Some(cpu),
            _ => None,
        }
    }

    /// Returns a mutable reference to the CPU frame data if this is a CPU frame.
    pub fn as_cpu_mut(&mut self) -> Option<&mut CpuFrame> {
        match &mut self.storage {
            FrameStorage::Cpu(cpu) => Some(cpu),
            _ => None,
        }
    }

    /// Pixel count.
    pub fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Validate frame consistency.
    pub fn validate(&self) -> MediaResult<()> {
        if self.width == 0 || self.height == 0 {
            return Err(MediaError::InvalidFrameSize {
                width: self.width,
                height: self.height,
            });
        }

        if let FrameStorage::Cpu(cpu) = &self.storage {
            let expected_min = self.width as usize
                * self.format.bytes_per_pixel().unwrap_or(4)
                * self.height as usize;
            if cpu.data.len() < expected_min {
                return Err(MediaError::BufferTooSmall {
                    needed: expected_min,
                    actual: cpu.data.len(),
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_cpu_frame() {
        let frame = Frame::new_cpu(1920, 1080, PixelFormat::Rgba8).unwrap();
        assert_eq!(frame.width, 1920);
        assert_eq!(frame.height, 1080);
        assert_eq!(frame.pixel_count(), 1920 * 1080);
        assert!(frame.as_cpu().is_some());
    }

    #[test]
    fn test_cpu_frame_stride() {
        let frame = Frame::new_cpu(640, 480, PixelFormat::Rgba8).unwrap();
        let cpu = frame.as_cpu().unwrap();
        assert_eq!(cpu.stride, 640 * 4);
    }

    #[test]
    fn test_validate_zero_size() {
        let frame = Frame::new_cpu(0, 100, PixelFormat::Rgba8).unwrap();
        assert!(frame.validate().is_err());
    }

    #[test]
    fn test_validate_ok() {
        let frame = Frame::new_cpu(100, 100, PixelFormat::Rgba8).unwrap();
        assert!(frame.validate().is_ok());
    }

    #[test]
    fn test_backend_kind() {
        let frame = Frame::new_cpu(100, 100, PixelFormat::Rgba8).unwrap();
        assert_eq!(frame.storage.backend_kind(), BackendKind::Cpu);
    }

    #[test]
    fn test_row_access() {
        let mut cpu = CpuFrame::new(10, 10, PixelFormat::Rgba8).unwrap();
        let row = cpu.row_mut(5);
        row[0] = 255;
        assert_eq!(cpu.row(5)[0], 255);
    }
}
