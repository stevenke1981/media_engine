//! Raw CUDA Driver API FFI declarations.
//!
//! These are direct bindings to the exports of `nvcuda.dll`.
//! Link configuration (rustc-link-lib=nvcuda + search path) is set in `build.rs`.

#![allow(non_camel_case_types, dead_code)]

// ── Core types ────────────────────────────────────────────────
pub type CUdevice = i32;
pub type CUresult = i32;
pub type CUdeviceptr = u64;

pub const CUDA_SUCCESS: CUresult = 0;

// ── Device attributes (CUdevice_attribute) ──────────────────
pub const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: i32 = 1;
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 16;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;

// ── Context flags ──────────────────────────────────────────
pub const CU_CTX_SCHED_AUTO: u32 = 0;

// ── Initialization ─────────────────────────────────────────
extern "system" {
    pub fn cuInit(flags: u32) -> CUresult;
    pub fn cuDriverGetVersion(version: *mut i32) -> CUresult;
}

// ── Device management ──────────────────────────────────────
extern "system" {
    pub fn cuDeviceGetCount(count: *mut i32) -> CUresult;
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: i32) -> CUresult;
    pub fn cuDeviceGetName(name: *mut i8, len: i32, dev: CUdevice) -> CUresult;
    pub fn cuDeviceTotalMem(bytes: *mut usize, dev: CUdevice) -> CUresult;
    pub fn cuDeviceGetAttribute(pi: *mut i32, attrib: i32, dev: CUdevice) -> CUresult;
}

// ── Context management ─────────────────────────────────────
extern "system" {
    pub fn cuCtxCreate(pctx: *mut usize, flags: u32, dev: CUdevice) -> CUresult;
    pub fn cuCtxSetCurrent(ctx: usize) -> CUresult;
    pub fn cuCtxGetCurrent(ctx: *mut usize) -> CUresult;
    pub fn cuCtxDestroy(ctx: usize) -> CUresult;
    pub fn cuCtxSynchronize() -> CUresult;
}

// ── Memory management ──────────────────────────────────────
extern "system" {
    pub fn cuMemAlloc(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    pub fn cuMemFree(dptr: CUdeviceptr) -> CUresult;
    pub fn cuMemcpyHtoD(dst: CUdeviceptr, src: *const u8, count: usize) -> CUresult;
    pub fn cuMemcpyDtoH(dst: *mut u8, src: CUdeviceptr, count: usize) -> CUresult;
}

// ── Module (PTX) loading ───────────────────────────────────
extern "system" {
    pub fn cuModuleLoadData(module: *mut usize, image: *const u8) -> CUresult;
    pub fn cuModuleGetFunction(func: *mut usize, module: usize, name: *const i8) -> CUresult;
    pub fn cuModuleUnload(module: usize) -> CUresult;
}

// ── Kernel execution ───────────────────────────────────────
extern "system" {
    pub fn cuLaunchKernel(
        func: usize,
        gridDimX: u32,
        gridDimY: u32,
        gridDimZ: u32,
        blockDimX: u32,
        blockDimY: u32,
        blockDimZ: u32,
        sharedMemBytes: u32,
        hStream: usize,
        kernelParams: *mut *mut std::ffi::c_void,
        extra: *mut *mut std::ffi::c_void,
    ) -> CUresult;
}

// ── Utility: check CUresult ──────────────────────────────────
/// Check a CUDA result and convert `CUDA_SUCCESS` → `()`, everything else → `Err`.
pub fn cu_check(result: CUresult) -> Result<(), String> {
    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("CUDA error {}", result))
    }
}
