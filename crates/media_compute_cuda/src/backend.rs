//! CUDA compute backend — implements `ComputeBackend`.

use crate::buffer::CudaBuffer;
use crate::device::{CudaContext, CudaDevice};
use crate::effects::CudaKernelEffect;
use crate::module::CudaModule;
use media_compute::kernel::{EffectDesc, EffectKind};
use media_compute::{ComputeBackend, ComputeBuffer, ComputeEffect};
use media_core::error::{MediaError, MediaResult};

/// The CUDA compute backend.
pub struct CudaBackend {
    device: CudaDevice,
    context: CudaContext,
    _module: CudaModule,
    effects: Vec<Box<dyn ComputeEffect>>,
}

impl CudaBackend {
    /// Initialise CUDA and create a backend on the first available device.
    pub fn new() -> MediaResult<Self> {
        let devices = CudaDevice::enumerate()?;
        if devices.is_empty() {
            return Err(MediaError::Cuda("No CUDA-capable devices found".into()));
        }
        Self::on_device(&devices[0])
    }

    /// Initialise CUDA on a specific device (by ordinal).
    pub fn on_device(device: &CudaDevice) -> MediaResult<Self> {
        let context = CudaContext::create(device)?;
        context.set_current()?;

        // ── Load all PTX kernels ─────────────────────────────
        let module = load_ptx_module("kernels_ptx")?;
        let brightness_func = module.get_function("brightness_kernel")?;
        let contrast_func = module.get_function("contrast_kernel")?;
        let grayscale_func = module.get_function("grayscale_kernel")?;
        let invert_func = module.get_function("invert_kernel")?;
        let blur_func = module.get_function("blur_kernel")?;
        let sharpen_func = module.get_function("sharpen_kernel")?;
        let sepia_func = module.get_function("sepia_kernel")?;
        let edge_detect_func = module.get_function("edge_detect_kernel")?;
        let threshold_func = module.get_function("threshold_kernel")?;
        let box_blur_func = module.get_function("box_blur_kernel")?;
        let emboss_func = module.get_function("emboss_kernel")?;
        let pixelate_func = module.get_function("pixelate_kernel")?;

        let effects: Vec<Box<dyn ComputeEffect>> = vec![
            Box::new(CudaKernelEffect::new(
                "brightness".into(),
                EffectKind::Brightness,
                brightness_func,
            )),
            Box::new(CudaKernelEffect::new(
                "contrast".into(),
                EffectKind::Contrast,
                contrast_func,
            )),
            Box::new(CudaKernelEffect::new(
                "grayscale".into(),
                EffectKind::Grayscale,
                grayscale_func,
            )),
            Box::new(CudaKernelEffect::new(
                "invert".into(),
                EffectKind::Invert,
                invert_func,
            )),
            Box::new(CudaKernelEffect::new(
                "blur".into(),
                EffectKind::Blur,
                blur_func,
            )),
            Box::new(CudaKernelEffect::new(
                "sharpen".into(),
                EffectKind::Sharpen,
                sharpen_func,
            )),
            Box::new(CudaKernelEffect::new(
                "sepia".into(),
                EffectKind::Sepia,
                sepia_func,
            )),
            Box::new(CudaKernelEffect::new(
                "edge_detect".into(),
                EffectKind::EdgeDetect,
                edge_detect_func,
            )),
            Box::new(CudaKernelEffect::new(
                "threshold".into(),
                EffectKind::Threshold,
                threshold_func,
            )),
            Box::new(CudaKernelEffect::new(
                "box_blur".into(),
                EffectKind::BoxBlur,
                box_blur_func,
            )),
            Box::new(CudaKernelEffect::new(
                "emboss".into(),
                EffectKind::Emboss,
                emboss_func,
            )),
            Box::new(CudaKernelEffect::new(
                "pixelate".into(),
                EffectKind::Pixelate,
                pixelate_func,
            )),
        ];

        Ok(CudaBackend {
            device: device.clone(),
            context,
            _module: module,
            effects,
        })
    }

    /// Ensure the CUDA context is current on this thread.
    fn ensure_context(&self) -> MediaResult<()> {
        self.context.set_current()
    }

    /// Device info (read-only).
    pub fn device(&self) -> &CudaDevice {
        &self.device
    }
}

impl ComputeBackend for CudaBackend {
    fn name(&self) -> &str {
        "cuda"
    }

    fn alloc_buffer(&self, size: usize) -> MediaResult<Box<dyn ComputeBuffer>> {
        self.ensure_context()?;
        let buf = CudaBuffer::new(size)?;
        Ok(Box::new(buf))
    }

    fn register_buffer(&self, data: &[u8]) -> MediaResult<Box<dyn ComputeBuffer>> {
        self.ensure_context()?;
        let buf = CudaBuffer::from_slice(data)?;
        Ok(Box::new(buf))
    }

    fn available_effects(&self) -> Vec<EffectDesc> {
        self.effects
            .iter()
            .map(|e| {
                EffectDesc::new(match e.name() {
                    "brightness" => EffectKind::Brightness,
                    "contrast" => EffectKind::Contrast,
                    "grayscale" => EffectKind::Grayscale,
                    "invert" => EffectKind::Invert,
                    "blur" => EffectKind::Blur,
                    "sharpen" => EffectKind::Sharpen,
                    "sepia" => EffectKind::Sepia,
                    "edge_detect" => EffectKind::EdgeDetect,
                    "threshold" => EffectKind::Threshold,
                    "box_blur" => EffectKind::BoxBlur,
                    "emboss" => EffectKind::Emboss,
                    "pixelate" => EffectKind::Pixelate,
                    _ => EffectKind::Custom(e.name().into()),
                })
            })
            .collect()
    }

    fn has_effect(&self, kind: &EffectKind) -> bool {
        self.effects.iter().any(|e| match kind {
            EffectKind::Brightness => e.name() == "brightness",
            EffectKind::Contrast => e.name() == "contrast",
            EffectKind::Grayscale => e.name() == "grayscale",
            EffectKind::Invert => e.name() == "invert",
            EffectKind::Blur => e.name() == "blur",
            EffectKind::Sharpen => e.name() == "sharpen",
            EffectKind::Sepia => e.name() == "sepia",
            EffectKind::EdgeDetect => e.name() == "edge_detect",
            EffectKind::Threshold => e.name() == "threshold",
            EffectKind::BoxBlur => e.name() == "box_blur",
            EffectKind::Emboss => e.name() == "emboss",
            EffectKind::Pixelate => e.name() == "pixelate",
            EffectKind::Custom(name) => e.name() == name,
            _ => false,
        })
    }

    fn get_effect(&self, kind: &EffectKind) -> Option<&dyn ComputeEffect> {
        self.effects
            .iter()
            .find(|e| match kind {
                EffectKind::Brightness => e.name() == "brightness",
                EffectKind::Contrast => e.name() == "contrast",
                EffectKind::Grayscale => e.name() == "grayscale",
                EffectKind::Invert => e.name() == "invert",
                EffectKind::Blur => e.name() == "blur",
                EffectKind::Sharpen => e.name() == "sharpen",
                EffectKind::Sepia => e.name() == "sepia",
                EffectKind::EdgeDetect => e.name() == "edge_detect",
                EffectKind::Threshold => e.name() == "threshold",
                EffectKind::BoxBlur => e.name() == "box_blur",
                EffectKind::Emboss => e.name() == "emboss",
                EffectKind::Pixelate => e.name() == "pixelate",
                EffectKind::Custom(name) => e.name() == name,
                _ => false,
            })
            .map(|b| b.as_ref())
    }
}

// ── PTX loading helper ─────────────────────────────────────────

// Load the bundled PTX module.
// The embed module is generated by `build.rs` and exposes
// static byte arrays named like `BRIGHTNESS_PTX`, etc.
include!(concat!(env!("OUT_DIR"), "/kernels_embedded.rs"));

fn load_ptx_module(_name: &str) -> MediaResult<CudaModule> {
    // For the simplest integration, we concatenate all PTX
    // into a single module.  CUDA supports multi-kernel PTX
    // files, so we just use the first PTX entry (or all
    // concatenated).

    // Actually, CUDA's cuModuleLoadData supports linking multiple
    // PTX strings. The simplest approach: load the concatenation
    // of all kernels.  But cuModuleLoadData takes a null-terminated
    // string, and PTX is text.  Let's concatenate them.

    // Collect all available PTX blobs.
    let blobs: &[&[u8]] = &[
        BRIGHTNESS_PTX,
        CONTRAST_PTX,
        GRAYSCALE_PTX,
        INVERT_PTX,
        BLUR_PTX,
        SHARPEN_PTX,
        SEPIA_PTX,
        EDGE_DETECT_PTX,
        THRESHOLD_PTX,
        BOX_BLUR_PTX,
        EMBOSS_PTX,
        PIXELATE_PTX,
    ];

    // Concatenate into one big PTX string (each is already a
    // complete .module with its own kernels).
    let mut full = Vec::new();
    for blob in blobs {
        full.extend_from_slice(blob);
        full.push(b'\n');
    }

    CudaModule::load(&full)
}
