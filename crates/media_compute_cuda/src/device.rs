//! CUDA device enumeration and context management.

use crate::ffi;
use media_core::error::{MediaError, MediaResult};

/// A handle to a CUDA-capable device.
#[derive(Debug, Clone)]
pub struct CudaDevice {
    /// Device ordinal (0-based index).
    pub ordinal: i32,
    /// Device name string (e.g. "NVIDIA GeForce RTX 4090").
    pub name: String,
    /// Total global memory in bytes.
    pub total_mem: usize,
    /// Number of multiprocessors.
    pub num_sms: i32,
    /// CUDA compute-capability major version.
    pub cc_major: i32,
    /// CUDA compute-capability minor version.
    pub cc_minor: i32,
}

/// A CUDA context wrapper that ensures thread-safe usage.
#[derive(Debug)]
pub struct CudaContext {
    /// Opaque context handle (CUcontext).
    handle: usize,
}

// SAFETY: CUDA contexts can be shared across threads if we set the
//         current context before every operation using cuCtxSetCurrent.
unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaDevice {
    /// Enumerate all CUDA-capable devices in the system.
    pub fn enumerate() -> MediaResult<Vec<CudaDevice>> {
        ffi::cu_check(unsafe { ffi::cuInit(0) })
            .map_err(|e| MediaError::Cuda(format!("cuInit failed: {e}")))?;

        let mut count: i32 = 0;
        ffi::cu_check(unsafe { ffi::cuDeviceGetCount(&mut count) })
            .map_err(|e| MediaError::Cuda(format!("cuDeviceGetCount failed: {e}")))?;

        let mut devices = Vec::with_capacity(count as usize);
        for ord in 0..count {
            devices.push(Self::get(ord)?);
        }
        Ok(devices)
    }

    /// Query properties for a specific device by ordinal.
    pub fn get(ordinal: i32) -> MediaResult<CudaDevice> {
        let mut dev: ffi::CUdevice = 0;
        ffi::cu_check(unsafe { ffi::cuDeviceGet(&mut dev, ordinal) })
            .map_err(|e| MediaError::Cuda(format!("cuDeviceGet({ordinal}) failed: {e}")))?;

        // ── Device name ──────────────────────────────────────
        let mut name_buf = [0i8; 256];
        ffi::cu_check(unsafe {
            ffi::cuDeviceGetName(name_buf.as_mut_ptr(), name_buf.len() as i32, dev)
        })
        .map_err(|e| MediaError::Cuda(format!("cuDeviceGetName failed: {e}")))?;
        let name = ptr_to_str(&name_buf);

        // ── Total memory ─────────────────────────────────────
        let mut total_mem: usize = 0;
        ffi::cu_check(unsafe { ffi::cuDeviceTotalMem(&mut total_mem, dev) })
            .map_err(|e| MediaError::Cuda(format!("cuDeviceTotalMem failed: {e}")))?;

        // ── Number of multiprocessors ─────────────────────────
        let num_sms = get_attrib(dev, ffi::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;

        // ── Compute capability ────────────────────────────────
        let cc_major = get_attrib(dev, ffi::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let cc_minor = get_attrib(dev, ffi::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;

        Ok(CudaDevice {
            ordinal,
            name,
            total_mem,
            num_sms,
            cc_major,
            cc_minor,
        })
    }
}

impl CudaContext {
    /// Create a new CUDA context for the specified device.
    pub fn create(device: &CudaDevice) -> MediaResult<Self> {
        let mut dev: ffi::CUdevice = 0;
        unsafe {
            ffi::cuDeviceGet(&mut dev, device.ordinal);
        }

        let mut handle: usize = 0;
        ffi::cu_check(unsafe { ffi::cuCtxCreate(&mut handle, ffi::CU_CTX_SCHED_AUTO, dev) })
            .map_err(|e| MediaError::Cuda(format!("cuCtxCreate failed: {e}")))?;

        Ok(CudaContext { handle })
    }

    /// Make this context current on the calling thread.
    pub fn set_current(&self) -> MediaResult<()> {
        ffi::cu_check(unsafe { ffi::cuCtxSetCurrent(self.handle) })
            .map_err(|e| MediaError::Cuda(format!("cuCtxSetCurrent failed: {e}")))
    }

    /// The raw handle (for FFI calls).
    pub fn handle(&self) -> usize {
        self.handle
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        let _ = unsafe { ffi::cuCtxDestroy(self.handle) };
    }
}

// ── Helpers ───────────────────────────────────────────────────

fn get_attrib(dev: ffi::CUdevice, attrib: i32) -> MediaResult<i32> {
    let mut val: i32 = 0;
    ffi::cu_check(unsafe { ffi::cuDeviceGetAttribute(&mut val, attrib, dev) })
        .map_err(|e| MediaError::Cuda(format!("cuDeviceGetAttribute({attrib}) failed: {e}")))?;
    Ok(val)
}

fn ptr_to_str(buf: &[i8]) -> String {
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}
