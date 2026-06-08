//! Buffer abstractions for compute backends.

use media_core::MediaResult;
use std::ops::Range;

/// Trait for compute-accessible buffer objects.
pub trait ComputeBuffer: Send + Sync {
    /// Returns the buffer size in bytes.
    fn size(&self) -> usize;

    /// Returns an opaque identifier (e.g. device pointer or handle).
    fn id(&self) -> u64;

    /// Whether this buffer is mapped (CPU-visible).
    fn is_mapped(&self) -> bool;

    /// Map buffer for CPU read/write.
    fn map(&mut self) -> MediaResult<()>;

    /// Unmap buffer.
    fn unmap(&mut self) -> MediaResult<()>;

    /// Read buffer contents into a CPU Vec.
    fn read_into_vec(&self) -> MediaResult<Vec<u8>>;

    /// Write CPU data into the buffer (at a given offset).
    fn write_from_slice(&mut self, offset: usize, data: &[u8]) -> MediaResult<()>;

    /// Copy data between two buffers (both on the same backend).
    fn copy_to(&self, dst: &mut dyn ComputeBuffer, range: Range<usize>) -> MediaResult<()>;
}
