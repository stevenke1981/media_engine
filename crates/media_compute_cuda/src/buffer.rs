//! CUDA device buffer implementing `ComputeBuffer`.

use crate::ffi;
use media_compute::ComputeBuffer;
use media_core::error::{MediaError, MediaResult};
use std::ops::Range;

/// A GPU-resident buffer backed by `cuMemAlloc` / `cuMemFree`.
///
/// A host-side `Vec<u8>` mirror is kept so that `map()` / `unmap()`
/// act as explicit synchronisation points:
///   - `map()`   → `cuMemcpyDtoH` (GPU → host)
///   - `unmap()` → `cuMemcpyHtoD` (host → GPU)
pub struct CudaBuffer {
    /// Host-side data mirror (pub(crate) so effects.rs can access it).
    pub(crate) host: Vec<u8>,
    /// CUDA device pointer (CUdeviceptr, stored as u64).
    device: ffi::CUdeviceptr,
    /// Allocation size in bytes.
    alloc_size: usize,
    /// Whether the host mirror is in sync with the device.
    mapped: bool,
}

impl CudaBuffer {
    /// Allocate device memory of the given size and zero the host mirror.
    pub fn new(size: usize) -> MediaResult<Self> {
        let mut device: ffi::CUdeviceptr = 0;
        ffi::cu_check(unsafe { ffi::cuMemAlloc(&mut device, size) })
            .map_err(|e| MediaError::Allocation(format!("cuMemAlloc({size}) failed: {e}")))?;
        Ok(CudaBuffer {
            host: vec![0u8; size],
            device,
            alloc_size: size,
            mapped: false,
        })
    }

    /// Create a CUDA buffer initialised from a byte slice (host → device).
    pub fn from_slice(data: &[u8]) -> MediaResult<Self> {
        let mut buf = Self::new(data.len())?;
        buf.host.copy_from_slice(data);
        buf.copy_to_device()?;
        Ok(buf)
    }

    /// The device pointer (CUdeviceptr).
    pub fn device_ptr(&self) -> ffi::CUdeviceptr {
        self.device
    }

    /// Copy host mirror → device.
    pub fn copy_to_device(&self) -> MediaResult<()> {
        ffi::cu_check(unsafe {
            ffi::cuMemcpyHtoD(self.device, self.host.as_ptr(), self.host.len())
        })
        .map_err(|e| MediaError::Cuda(format!("cuMemcpyHtoD failed: {e}")))
    }

    /// Copy device → host mirror.
    pub fn sync_host(&mut self) -> MediaResult<()> {
        ffi::cu_check(unsafe {
            ffi::cuMemcpyDtoH(self.host.as_mut_ptr(), self.device, self.host.len())
        })
        .map_err(|e| MediaError::Cuda(format!("cuMemcpyDtoH failed: {e}")))
    }

    /// Copy device → a fresh `Vec<u8>` (without touching host mirror).
    pub fn read_into_vec_raw(&self) -> MediaResult<Vec<u8>> {
        let mut dst = vec![0u8; self.alloc_size];
        ffi::cu_check(unsafe { ffi::cuMemcpyDtoH(dst.as_mut_ptr(), self.device, dst.len()) })
            .map_err(|e| MediaError::Cuda(format!("cuMemcpyDtoH failed: {e}")))?;
        Ok(dst)
    }

    /// Write data into the device buffer at the given offset.
    pub fn write_at_offset(&self, offset: usize, data: &[u8]) -> MediaResult<()> {
        if offset + data.len() > self.alloc_size {
            return Err(MediaError::BufferTooSmall {
                needed: offset + data.len(),
                actual: self.alloc_size,
            });
        }
        ffi::cu_check(unsafe {
            ffi::cuMemcpyHtoD(self.device + offset as u64, data.as_ptr(), data.len())
        })
        .map_err(|e| MediaError::Cuda(format!("cuMemcpyHtoD at offset failed: {e}")))
    }
}

impl ComputeBuffer for CudaBuffer {
    fn size(&self) -> usize {
        self.alloc_size
    }

    fn id(&self) -> u64 {
        self.device
    }

    fn is_mapped(&self) -> bool {
        self.mapped
    }

    fn map(&mut self) -> MediaResult<()> {
        if !self.mapped {
            self.sync_host()?;
            self.mapped = true;
        }
        Ok(())
    }

    fn unmap(&mut self) -> MediaResult<()> {
        if self.mapped {
            self.copy_to_device()?;
            self.mapped = false;
        }
        Ok(())
    }

    fn read_into_vec(&self) -> MediaResult<Vec<u8>> {
        if self.mapped {
            Ok(self.host.clone())
        } else {
            self.read_into_vec_raw()
        }
    }

    fn write_from_slice(&mut self, offset: usize, data: &[u8]) -> MediaResult<()> {
        // Write into host mirror
        if offset + data.len() > self.alloc_size {
            return Err(MediaError::BufferTooSmall {
                needed: offset + data.len(),
                actual: self.alloc_size,
            });
        }

        if self.mapped {
            // Host mirror is valid — write there and sync to device.
            self.host[offset..offset + data.len()].copy_from_slice(data);
            self.copy_to_device()?;
        } else {
            // Write directly to device memory.
            self.write_at_offset(offset, data)?;
        }
        Ok(())
    }

    fn copy_to(&self, dst: &mut dyn ComputeBuffer, range: Range<usize>) -> MediaResult<()> {
        // Generic fallback: read from this buffer → write into destination.
        let data = self.read_into_vec()?;
        dst.write_from_slice(range.start, &data[range])?;
        Ok(())
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if self.device != 0 {
            unsafe {
                ffi::cuMemFree(self.device);
            }
        }
    }
}

// SAFETY: the caller must ensure the CUDA context is current on the
//         thread where driver calls are made (handled by CudaBackend).
unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}
