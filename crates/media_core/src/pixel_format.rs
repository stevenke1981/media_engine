/// Supported pixel formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// 8-bit RGB, 3 channels.
    Rgb8,
    /// 8-bit RGBA, 4 channels.
    Rgba8,
    /// 8-bit BGRA, 4 channels.
    Bgra8,
    /// 16-bit float RGBA, 4 channels.
    Rgba16Float,
    /// 32-bit float RGBA, 4 channels.
    Rgba32Float,
    /// Planar YUV 4:2:0, 3 planes.
    Yuv420p,
    /// Semi-planar NV12, 2 planes.
    Nv12,
    /// 10-bit planar, 2 planes.
    P010,
}

impl PixelFormat {
    /// Number of bytes per pixel (for packed formats).
    /// Returns `None` for planar formats.
    pub fn bytes_per_pixel(&self) -> Option<usize> {
        match self {
            PixelFormat::Rgb8 => Some(3),
            PixelFormat::Rgba8 => Some(4),
            PixelFormat::Bgra8 => Some(4),
            PixelFormat::Rgba16Float => Some(8),
            PixelFormat::Rgba32Float => Some(16),
            PixelFormat::Yuv420p => None,
            PixelFormat::Nv12 => None,
            PixelFormat::P010 => None,
        }
    }

    /// Total number of planes.
    pub fn planes(&self) -> usize {
        match self {
            PixelFormat::Yuv420p => 3,
            PixelFormat::Nv12 => 2,
            PixelFormat::P010 => 2,
            _ => 1,
        }
    }

    /// Whether this format is planar (YUV-style).
    pub fn is_planar(&self) -> bool {
        self.planes() > 1
    }

    /// Whether this format has an alpha channel.
    pub fn has_alpha(&self) -> bool {
        matches!(
            self,
            PixelFormat::Rgba8
                | PixelFormat::Bgra8
                | PixelFormat::Rgba16Float
                | PixelFormat::Rgba32Float
        )
    }

    /// Returns the minimum size in bytes for a buffer of given dimensions.
    pub fn buffer_size(&self, width: u32, height: u32) -> Option<usize> {
        match self {
            PixelFormat::Rgb8 => Some(width as usize * height as usize * 3),
            PixelFormat::Rgba8 => Some(width as usize * height as usize * 4),
            PixelFormat::Bgra8 => Some(width as usize * height as usize * 4),
            PixelFormat::Rgba16Float => Some(width as usize * height as usize * 8),
            PixelFormat::Rgba32Float => Some(width as usize * height as usize * 16),
            PixelFormat::Yuv420p => {
                let w = width as usize;
                let h = height as usize;
                Some(w * h + (w / 2) * (h / 2) * 2)
            }
            PixelFormat::Nv12 => {
                let w = width as usize;
                let h = height as usize;
                Some(w * h + (w / 2) * (h / 2) * 2)
            }
            PixelFormat::P010 => {
                let w = width as usize;
                let h = height as usize;
                Some(w * h * 2 + (w / 2) * (h / 2) * 4)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_per_pixel() {
        assert_eq!(PixelFormat::Rgba8.bytes_per_pixel(), Some(4));
        assert_eq!(PixelFormat::Rgb8.bytes_per_pixel(), Some(3));
        assert_eq!(PixelFormat::Yuv420p.bytes_per_pixel(), None);
    }

    #[test]
    fn test_buffer_size_rgba8() {
        let size = PixelFormat::Rgba8.buffer_size(1920, 1080);
        assert_eq!(size, Some(1920 * 1080 * 4));
    }

    #[test]
    fn test_is_planar() {
        assert!(!PixelFormat::Rgba8.is_planar());
        assert!(PixelFormat::Yuv420p.is_planar());
        assert!(PixelFormat::Nv12.is_planar());
    }

    #[test]
    fn test_has_alpha() {
        assert!(PixelFormat::Rgba8.has_alpha());
        assert!(!PixelFormat::Rgb8.has_alpha());
        assert!(!PixelFormat::Yuv420p.has_alpha());
    }
}
