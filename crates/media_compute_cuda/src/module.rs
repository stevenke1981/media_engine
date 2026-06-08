//! CUDA module / PTX loading and kernel function management.

use crate::ffi;
use media_core::error::{MediaError, MediaResult};

/// A compiled CUDA module (loaded from PTX).
#[derive(Debug)]
pub struct CudaModule {
    /// Opaque module handle (CUmodule).
    handle: usize,
}

impl CudaModule {
    /// Load a PTX module from raw bytes.
    ///
    /// The context must be current on the calling thread.
    pub fn load(ptx_data: &[u8]) -> MediaResult<Self> {
        let mut handle: usize = 0;
        ffi::cu_check(unsafe { ffi::cuModuleLoadData(&mut handle, ptx_data.as_ptr()) })
            .map_err(|e| MediaError::Cuda(format!("cuModuleLoadData failed: {e}")))?;
        Ok(CudaModule { handle })
    }

    /// Get a kernel function handle by name.
    pub fn get_function(&self, name: &str) -> MediaResult<CudaFunction> {
        let c_name = std::ffi::CString::new(name)
            .map_err(|_| MediaError::Other("Invalid kernel name (contains null byte)".into()))?;

        let mut func: usize = 0;
        ffi::cu_check(unsafe { ffi::cuModuleGetFunction(&mut func, self.handle, c_name.as_ptr()) })
            .map_err(|e| MediaError::Kernel {
                name: name.to_string(),
                message: format!("cuModuleGetFunction failed: {e}"),
            })?;

        Ok(CudaFunction { handle: func })
    }

    /// The raw module handle.
    pub fn handle(&self) -> usize {
        self.handle
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        let _ = unsafe { ffi::cuModuleUnload(self.handle) };
    }
}

/// A handle to a kernel function inside a loaded CUDA module.
#[derive(Debug, Clone, Copy)]
pub struct CudaFunction {
    pub(crate) handle: usize,
}

impl CudaFunction {
    /// Launch this kernel with the given configuration and arguments.
    ///
    /// `kernel_args` is a slice of pointers, each pointing to the raw
    /// argument bytes (matching the kernel's parameter layout).
    pub fn launch(
        &self,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        kernel_args: &[*mut std::ffi::c_void],
    ) -> MediaResult<()> {
        // We need a `*mut *mut c_void`.  The array elements already
        // satisfy `*mut c_void`, so just take the pointer of the array.
        let params = if kernel_args.is_empty() {
            std::ptr::null_mut()
        } else {
            kernel_args.as_ptr() as *mut *mut std::ffi::c_void
        };

        ffi::cu_check(unsafe {
            ffi::cuLaunchKernel(
                self.handle,
                grid_x,
                grid_y,
                grid_z,
                block_x,
                block_y,
                block_z,
                0, // sharedMemBytes
                0, // hStream (default)
                params,
                std::ptr::null_mut(), // extra
            )
        })
        .map_err(|e| MediaError::Kernel {
            name: String::from("unknown"),
            message: format!("cuLaunchKernel failed: {e}"),
        })
    }
}
