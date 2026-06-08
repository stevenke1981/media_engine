/// Supported compute/render backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Cpu,
    Cuda,
    Wgpu,
    Vulkan,
    External,
}

impl BackendKind {
    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            BackendKind::Cpu => "CPU",
            BackendKind::Cuda => "CUDA",
            BackendKind::Wgpu => "WGPU",
            BackendKind::Vulkan => "Vulkan",
            BackendKind::External => "External",
        }
    }
}

/// Descriptor for a flat buffer allocation.
#[derive(Debug, Clone)]
pub struct BufferDesc {
    pub size: usize,
    pub name: Option<String>,
}

/// Descriptor for a texture/image allocation.
#[derive(Debug, Clone)]
pub struct TextureDesc {
    pub width: u32,
    pub height: u32,
    pub format: crate::pixel_format::PixelFormat,
    pub name: Option<String>,
}

/// Descriptor for a frame allocation.
#[derive(Debug, Clone)]
pub struct FrameDesc {
    pub width: u32,
    pub height: u32,
    pub format: crate::pixel_format::PixelFormat,
}

/// Descriptor for a compute kernel invocation.
#[derive(Debug, Clone)]
pub struct KernelDesc {
    pub name: String,
    pub ptx_name: Option<String>,
    pub shared_mem_bytes: u32,
}

impl BufferDesc {
    pub fn new(size: usize) -> Self {
        BufferDesc { size, name: None }
    }

    pub fn named(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }
}

impl TextureDesc {
    pub fn new(width: u32, height: u32, format: crate::pixel_format::PixelFormat) -> Self {
        TextureDesc {
            width,
            height,
            format,
            name: None,
        }
    }

    pub fn named(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }
}

/// Grid size for kernel launch.
#[derive(Debug, Clone, Copy)]
pub struct GridSize {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl GridSize {
    pub fn xy(x: u32, y: u32) -> Self {
        GridSize { x, y, z: 1 }
    }

    pub fn flat(x: u32) -> Self {
        GridSize { x, y: 1, z: 1 }
    }
}

/// Block / thread group size for kernel launch.
#[derive(Debug, Clone, Copy)]
pub struct BlockSize {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl BlockSize {
    pub fn xy(x: u32, y: u32) -> Self {
        BlockSize { x, y, z: 1 }
    }

    pub fn flat(x: u32) -> Self {
        BlockSize { x, y: 1, z: 1 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_name() {
        assert_eq!(BackendKind::Cpu.name(), "CPU");
        assert_eq!(BackendKind::Cuda.name(), "CUDA");
        assert_eq!(BackendKind::Wgpu.name(), "WGPU");
    }

    #[test]
    fn test_buffer_desc() {
        let desc = BufferDesc::new(1024).named("test");
        assert_eq!(desc.size, 1024);
        assert_eq!(desc.name.as_deref(), Some("test"));
    }

    #[test]
    fn test_grid_size() {
        let g = GridSize::xy(16, 16);
        assert_eq!(g.x, 16);
        assert_eq!(g.y, 16);
        assert_eq!(g.z, 1);
    }
}
