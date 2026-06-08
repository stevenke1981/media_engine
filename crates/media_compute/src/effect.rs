//! Compute effect trait — the core image processing abstraction.

use crate::buffer::ComputeBuffer;
use crate::kernel::{EffectDesc, LaunchConfig};
use media_core::{Frame, MediaResult};

/// Trait for objects that can apply compute effects to frames.
pub trait ComputeEffect: Send + Sync {
    /// The name of this effect (e.g. "brightness").
    fn name(&self) -> &str;

    /// Apply the effect to a source frame, producing a new output frame.
    fn apply(&self, src: &Frame, desc: &EffectDesc) -> MediaResult<Frame>;

    /// Whether this effect prefers an in-place (mutable) interface.
    fn supports_in_place(&self) -> bool {
        false
    }

    /// Apply the effect in-place, mutating the frame.
    fn apply_in_place(&self, _frame: &mut Frame, _desc: &EffectDesc) -> MediaResult<()> {
        Err(media_core::MediaError::Other(
            "In-place not supported for this effect".into(),
        ))
    }
}

/// Trait for compute backends (CPU, CUDA, WGPU, etc.)
pub trait ComputeBackend: Send + Sync {
    /// A human-readable name for this backend.
    fn name(&self) -> &str;

    /// Allocate a buffer on this backend.
    fn alloc_buffer(&self, size: usize) -> MediaResult<Box<dyn ComputeBuffer>>;

    /// Register an existing CPU buffer with this backend.
    fn register_buffer(&self, _data: &[u8]) -> MediaResult<Box<dyn ComputeBuffer>> {
        Err(media_core::MediaError::Other(
            "Buffer registration not supported on this backend".into(),
        ))
    }

    /// List available effects.
    fn available_effects(&self) -> Vec<EffectDesc>;

    /// Check whether a specific effect is available.
    fn has_effect(&self, kind: &crate::kernel::EffectKind) -> bool;

    /// Get the launcher for a specific effect.
    fn get_effect(&self, kind: &crate::kernel::EffectKind) -> Option<&dyn ComputeEffect>;
}

/// Default launch config: 16x16 thread blocks.
pub const DEFAULT_BLOCK_SIZE: (u32, u32) = (16, 16);

/// Compute the grid size needed to cover an image.
pub fn grid_for_size(width: u32, height: u32, block_w: u32, block_h: u32) -> LaunchConfig {
    LaunchConfig::xy(width, height, block_w, block_h)
}

/// Convert bytes per pixel to a block dimension heuristic.
pub fn block_for_bpp(bpp: u32) -> (u32, u32) {
    if bpp <= 4 {
        (32, 8)
    } else {
        (16, 16)
    }
}
