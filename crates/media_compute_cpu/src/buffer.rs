//! CPU-side `ComputeBuffer` implementation.

use media_compute::ComputeBuffer;
use media_core::MediaResult;
use std::ops::Range;

/// A simple CPU-allocated buffer.
#[derive(Debug, Clone)]
pub struct CpuBuffer {
    data: Vec<u8>,
    mapped: bool,
}

impl CpuBuffer {
    pub fn new(size: usize) -> Self {
        CpuBuffer {
            data: vec![0u8; size],
            mapped: false,
        }
    }

    pub fn from_slice(data: &[u8]) -> Self {
        CpuBuffer {
            data: data.to_vec(),
            mapped: false,
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

impl ComputeBuffer for CpuBuffer {
    fn size(&self) -> usize {
        self.data.len()
    }

    fn id(&self) -> u64 {
        // Use the raw pointer as an ID.
        self.data.as_ptr() as u64
    }

    fn is_mapped(&self) -> bool {
        self.mapped
    }

    fn map(&mut self) -> MediaResult<()> {
        self.mapped = true;
        Ok(())
    }

    fn unmap(&mut self) -> MediaResult<()> {
        self.mapped = false;
        Ok(())
    }

    fn read_into_vec(&self) -> MediaResult<Vec<u8>> {
        Ok(self.data.clone())
    }

    fn write_from_slice(&mut self, offset: usize, data: &[u8]) -> MediaResult<()> {
        let end = offset + data.len();
        if end > self.data.len() {
            return Err(media_core::MediaError::BufferTooSmall {
                needed: end,
                actual: self.data.len(),
            });
        }
        self.data[offset..end].copy_from_slice(data);
        Ok(())
    }

    fn copy_to(&self, dst: &mut dyn ComputeBuffer, range: Range<usize>) -> MediaResult<()> {
        dst.write_from_slice(range.start, &self.data[range.clone()])?;
        Ok(())
    }
}
