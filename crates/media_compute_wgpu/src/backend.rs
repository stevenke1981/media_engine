//! WGPU compute backend — implements `ComputeBackend`.

use crate::buffer::WgpuBuffer;
use crate::effects::WgpuEffect;
use media_compute::kernel::EffectKind;
use media_compute::{ComputeBackend, ComputeBuffer, ComputeEffect};
use media_core::error::{MediaError, MediaResult};
use std::sync::Arc;

/// The WGPU compute backend.
pub struct WgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    effects: Vec<Box<dyn ComputeEffect>>,
}

impl WgpuBackend {
    /// Initialise WGPU and create a backend on the first available adapter.
    ///
    /// If `adapter_index` is `None` the first suitable adapter is chosen.
    pub fn new(adapter_index: Option<usize>) -> MediaResult<Self> {
        let instance = wgpu::Instance::default();

        // Enumerate adapters.
        let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all());
        if adapters.is_empty() {
            return Err(MediaError::Wgpu("No WGPU-capable adapters found".into()));
        }

        let idx = adapter_index.unwrap_or(0);
        let adapter = adapters
            .into_iter()
            .nth(idx)
            .ok_or_else(|| MediaError::Wgpu(format!("Adapter index {idx} out of range")))?;

        let features = adapter.features();
        let limits = adapter.limits();

        log::info!(
            "WGPU adapter: {:?} (features={:?}, limits={:?})",
            adapter.get_info(),
            features,
            limits,
        );

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("WgpuBackend device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None, // trace_path
        ))
        .map_err(|e| MediaError::Wgpu(format!("Device request failed: {e}")))?;

        let device: Arc<wgpu::Device> = Arc::new(device);
        let queue: Arc<wgpu::Queue> = Arc::new(queue);

        // ── WGSL shader sources ────────────────────────────────
        //
        // All shaders share the same uniform struct:
        //   struct Uniforms { width: u32, height: u32, param0: f32, param1: f32 };
        //
        // binding 0: read-write storage buffer (BGRA8 u32 pixels)
        // binding 1: uniform buffer (width, height, effect parameters)

        let brightness_shader = r#"
struct Uniforms { width: u32, height: u32, param0: f32, param1: f32 };
@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= uniforms.width || id.y >= uniforms.height) { return; }
    let idx = id.y * uniforms.width + id.x;
    let pixel = buf[idx];
    let b = min(u32(f32(pixel & 0xFFu) * uniforms.param0), 255u);
    let g = min(u32(f32((pixel >> 8u) & 0xFFu) * uniforms.param0), 255u);
    let r = min(u32(f32((pixel >> 16u) & 0xFFu) * uniforms.param0), 255u);
    let a = (pixel >> 24u) & 0xFFu;
    buf[idx] = b | (g << 8u) | (r << 16u) | (a << 24u);
}
"#;

        let contrast_shader = r#"
struct Uniforms { width: u32, height: u32, param0: f32, param1: f32 };
@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= uniforms.width || id.y >= uniforms.height) { return; }
    let idx = id.y * uniforms.width + id.x;
    let pixel = buf[idx];
    let b = f32(pixel & 0xFFu);
    let g = f32((pixel >> 8u) & 0xFFu);
    let r = f32((pixel >> 16u) & 0xFFu);
    let a = (pixel >> 24u) & 0xFFu;
    let nb = clamp(((b / 255.0 - 0.5) * uniforms.param0 + 0.5) * 255.0, 0.0, 255.0);
    let ng = clamp(((g / 255.0 - 0.5) * uniforms.param0 + 0.5) * 255.0, 0.0, 255.0);
    let nr = clamp(((r / 255.0 - 0.5) * uniforms.param0 + 0.5) * 255.0, 0.0, 255.0);
    buf[idx] = u32(nb) | (u32(ng) << 8u) | (u32(nr) << 16u) | (a << 24u);
}
"#;

        let grayscale_shader = r#"
struct Uniforms { width: u32, height: u32, param0: f32, param1: f32 };
@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= uniforms.width || id.y >= uniforms.height) { return; }
    let idx = id.y * uniforms.width + id.x;
    let pixel = buf[idx];
    let b = f32(pixel & 0xFFu);
    let g = f32((pixel >> 8u) & 0xFFu);
    let r = f32((pixel >> 16u) & 0xFFu);
    let a = (pixel >> 24u) & 0xFFu;
    let gray = u32(0.299 * r + 0.587 * g + 0.114 * b);
    let gray8 = gray & 0xFFu;
    buf[idx] = gray8 | (gray8 << 8u) | (gray8 << 16u) | (a << 24u);
}
"#;

        let invert_shader = r#"
struct Uniforms { width: u32, height: u32, param0: f32, param1: f32 };
@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= uniforms.width || id.y >= uniforms.height) { return; }
    let idx = id.y * uniforms.width + id.x;
    let pixel = buf[idx];
    let b = 255u - (pixel & 0xFFu);
    let g = 255u - ((pixel >> 8u) & 0xFFu);
    let r = 255u - ((pixel >> 16u) & 0xFFu);
    let a = (pixel >> 24u) & 0xFFu;
    buf[idx] = b | (g << 8u) | (r << 16u) | (a << 24u);
}
"#;

        let blur_shader = r#"
struct Uniforms { width: u32, height: u32, param0: f32, param1: f32 };
@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= uniforms.width || id.y >= uniforms.height) { return; }
    let idx = id.y * uniforms.width + id.x;
    let pixel = buf[idx];

    var sum_r: u32 = 0u;
    var sum_g: u32 = 0u;
    var sum_b: u32 = 0u;
    var count: u32 = 0u;

    // 3×3 box blur — unrolled neighborhood with bounds checking
    for (var dy = -1i; dy <= 1i; dy++) {
        for (var dx = -1i; dx <= 1i; dx++) {
            let nx = i32(id.x) + dx;
            let ny = i32(id.y) + dy;
            if (nx >= 0i && nx < i32(uniforms.width) && ny >= 0i && ny < i32(uniforms.height)) {
                let ni = u32(ny) * uniforms.width + u32(nx);
                let np = buf[ni];
                sum_r += (np >> 16u) & 0xFFu;
                sum_g += (np >> 8u) & 0xFFu;
                sum_b += np & 0xFFu;
                count++;
            }
        }
    }

    let b = (sum_b / count) & 0xFFu;
    let g = (sum_g / count) & 0xFFu;
    let r = (sum_r / count) & 0xFFu;
    let a = (pixel >> 24u) & 0xFFu;
    buf[idx] = b | (g << 8u) | (r << 16u) | (a << 24u);
}
"#;

        let sharpen_shader = r#"
struct Uniforms { width: u32, height: u32, param0: f32, param1: f32 };
@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= uniforms.width || id.y >= uniforms.height) { return; }
    let idx = id.y * uniforms.width + id.x;
    let pixel = buf[idx];
    let center_r = f32((pixel >> 16u) & 0xFFu);
    let center_g = f32((pixel >> 8u) & 0xFFu);
    let center_b = f32(pixel & 0xFFu);
    let a = (pixel >> 24u) & 0xFFu;

    // 4-neighbour average (up / down / left / right)
    var blur_r: u32 = 0u;
    var blur_g: u32 = 0u;
    var blur_b: u32 = 0u;
    var count: u32 = 0u;

    if (id.x > 0u) {
        let ni = id.y * uniforms.width + (id.x - 1u);
        let np = buf[ni];
        blur_r += (np >> 16u) & 0xFFu;
        blur_g += (np >> 8u) & 0xFFu;
        blur_b += np & 0xFFu;
        count++;
    }
    if (id.x + 1u < uniforms.width) {
        let ni = id.y * uniforms.width + (id.x + 1u);
        let np = buf[ni];
        blur_r += (np >> 16u) & 0xFFu;
        blur_g += (np >> 8u) & 0xFFu;
        blur_b += np & 0xFFu;
        count++;
    }
    if (id.y > 0u) {
        let ni = (id.y - 1u) * uniforms.width + id.x;
        let np = buf[ni];
        blur_r += (np >> 16u) & 0xFFu;
        blur_g += (np >> 8u) & 0xFFu;
        blur_b += np & 0xFFu;
        count++;
    }
    if (id.y + 1u < uniforms.height) {
        let ni = (id.y + 1u) * uniforms.width + id.x;
        let np = buf[ni];
        blur_r += (np >> 16u) & 0xFFu;
        blur_g += (np >> 8u) & 0xFFu;
        blur_b += np & 0xFFu;
        count++;
    }

    let avg_r = f32(blur_r) / f32(max(count, 1u));
    let avg_g = f32(blur_g) / f32(max(count, 1u));
    let avg_b = f32(blur_b) / f32(max(count, 1u));

    // Unsharp-mask: out = center + (center - blur) * strength
    let strength = uniforms.param0;
    let sharp_r = clamp(center_r + (center_r - avg_r) * strength, 0.0, 255.0);
    let sharp_g = clamp(center_g + (center_g - avg_g) * strength, 0.0, 255.0);
    let sharp_b = clamp(center_b + (center_b - avg_b) * strength, 0.0, 255.0);

    buf[idx] = u32(sharp_b) | (u32(sharp_g) << 8u) | (u32(sharp_r) << 16u) | (a << 24u);
}
"#;

        let edge_detect_shader = r#"
struct Uniforms { width: u32, height: u32, param0: f32, param1: f32 };
@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= uniforms.width || id.y >= uniforms.height) { return; }
    let idx = id.y * uniforms.width + id.x;
    let pixel = buf[idx];
    let a = (pixel >> 24u) & 0xFFu;

    var gx: f32 = 0.0;
    var gy: f32 = 0.0;

    for (var ky = -1i; ky <= 1i; ky++) {
        for (var kx = -1i; kx <= 1i; kx++) {
            let nx = i32(id.x) + kx;
            let ny = i32(id.y) + ky;
            if (nx >= 0i && nx < i32(uniforms.width) && ny >= 0i && ny < i32(uniforms.height)) {
                let ni = u32(ny) * uniforms.width + u32(nx);
                let np = buf[ni];
                let nb = f32(np & 0xFFu);
                let ng = f32((np >> 8u) & 0xFFu);
                let nr = f32((np >> 16u) & 0xFFu);
                let lum = nr * 0.2126 + ng * 0.7152 + nb * 0.0722;

                let skx = f32(kx);
                let sky = f32(ky);
                gx += lum * skx * f32(select(1, 2, ky == 0));
                gy += lum * sky * f32(select(1, 2, kx == 0));
            }
        }
    }

    let mag = u32(min(sqrt(gx * gx + gy * gy), 255.0));
    buf[idx] = mag | (mag << 8u) | (mag << 16u) | (a << 24u);
}
"#;

        let sepia_shader = r#"
struct Uniforms { width: u32, height: u32, param0: f32, param1: f32 };
@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= uniforms.width || id.y >= uniforms.height) { return; }
    let idx = id.y * uniforms.width + id.x;
    let pixel = buf[idx];
    let b = f32(pixel & 0xFFu);
    let g = f32((pixel >> 8u) & 0xFFu);
    let r = f32((pixel >> 16u) & 0xFFu);
    let a = (pixel >> 24u) & 0xFFu;

    let sr = min(r * 0.393 + g * 0.769 + b * 0.189, 255.0);
    let sg = min(r * 0.349 + g * 0.686 + b * 0.168, 255.0);
    let sb = min(r * 0.272 + g * 0.534 + b * 0.131, 255.0);

    buf[idx] = u32(sb) | (u32(sg) << 8u) | (u32(sr) << 16u) | (a << 24u);
}
"#;

        let threshold_shader = r#"
struct Uniforms { width: u32, height: u32, param0: f32, param1: f32 };
@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= uniforms.width || id.y >= uniforms.height) { return; }
    let idx = id.y * uniforms.width + id.x;
    let pixel = buf[idx];
    let b = f32(pixel & 0xFFu);
    let g = f32((pixel >> 8u) & 0xFFu);
    let r = f32((pixel >> 16u) & 0xFFu);
    let a = (pixel >> 24u) & 0xFFu;

    // Simple average luma (matching CPU behavior)
    let luma = (r + g + b) / 3.0;
    let threshold = uniforms.param0 * 255.0;
    let val = u32(select(0, 255, luma > threshold));

    buf[idx] = val | (val << 8u) | (val << 16u) | (a << 24u);
}
"#;

        let box_blur_shader = r#"
struct Uniforms { width: u32, height: u32, param0: f32, param1: f32 };
@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= uniforms.width || id.y >= uniforms.height) { return; }
    let idx = id.y * uniforms.width + id.x;
    let pixel = buf[idx];

    var sum_r: u32 = 0u;
    var sum_g: u32 = 0u;
    var sum_b: u32 = 0u;
    var count: u32 = 0u;

    // 3×3 uniform box blur
    for (var dy = -1i; dy <= 1i; dy++) {
        for (var dx = -1i; dx <= 1i; dx++) {
            let nx = i32(id.x) + dx;
            let ny = i32(id.y) + dy;
            if (nx >= 0i && nx < i32(uniforms.width) && ny >= 0i && ny < i32(uniforms.height)) {
                let ni = u32(ny) * uniforms.width + u32(nx);
                let np = buf[ni];
                sum_r += (np >> 16u) & 0xFFu;
                sum_g += (np >> 8u) & 0xFFu;
                sum_b += np & 0xFFu;
                count++;
            }
        }
    }

    let b = (sum_b / count) & 0xFFu;
    let g = (sum_g / count) & 0xFFu;
    let r = (sum_r / count) & 0xFFu;
    let a = (pixel >> 24u) & 0xFFu;
    buf[idx] = b | (g << 8u) | (r << 16u) | (a << 24u);
}
"#;

        let emboss_shader = r#"
struct Uniforms { width: u32, height: u32, param0: f32, param1: f32 };
@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= uniforms.width || id.y >= uniforms.height) { return; }
    let idx = id.y * uniforms.width + id.x;
    let pixel = buf[idx];
    let a = (pixel >> 24u) & 0xFFu;

    var acc: i32 = 0;

    for (var ky = -1i; ky <= 1i; ky++) {
        for (var kx = -1i; kx <= 1i; kx++) {
            let nx = i32(id.x) + kx;
            let ny = i32(id.y) + ky;
            if (nx >= 0i && nx < i32(uniforms.width) && ny >= 0i && ny < i32(uniforms.height)) {
                let ni = u32(ny) * uniforms.width + u32(nx);
                let np = buf[ni];
                let nb = f32(np & 0xFFu);
                let ng = f32((np >> 8u) & 0xFFu);
                let nr = f32((np >> 16u) & 0xFFu);
                let lum = i32(nr * 0.2126 + ng * 0.7152 + nb * 0.0722);

                // Emboss kernel: kx+ky, center=1
                let coeff = kx + ky + select(0, 1, kx == 0 && ky == 0);
                acc += lum * coeff;
            }
        }
    }

    // Bias by 128 for raised-relief effect
    let val = u32(clamp(acc + 128, 0, 255));
    buf[idx] = val | (val << 8u) | (val << 16u) | (a << 24u);
}
"#;

        let pixelate_shader = r#"
struct Uniforms { width: u32, height: u32, param0: f32, param1: f32 };
@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= uniforms.width || id.y >= uniforms.height) { return; }
    let idx = id.y * uniforms.width + id.x;
    let pixel = buf[idx];
    let a = (pixel >> 24u) & 0xFFu;

    let block_size = max(u32(uniforms.param0), 1u);
    let bx = id.x / block_size;
    let by = id.y / block_size;
    let start_x = bx * block_size;
    let start_y = by * block_size;
    let end_x = min(start_x + block_size, uniforms.width);
    let end_y = min(start_y + block_size, uniforms.height);

    var sum_r: u32 = 0u;
    var sum_g: u32 = 0u;
    var sum_b: u32 = 0u;
    var count: u32 = 0u;

    for (var py = start_y; py < end_y; py++) {
        for (var px = start_x; px < end_x; px++) {
            let ni = py * uniforms.width + px;
            let np = buf[ni];
            sum_r += (np >> 16u) & 0xFFu;
            sum_g += (np >> 8u) & 0xFFu;
            sum_b += np & 0xFFu;
            count++;
        }
    }

    let avg_r = (sum_r / count) & 0xFFu;
    let avg_g = (sum_g / count) & 0xFFu;
    let avg_b = (sum_b / count) & 0xFFu;

    buf[idx] = avg_b | (avg_g << 8u) | (avg_r << 16u) | (a << 24u);
}
"#;

        // ── Create effects ─────────────────────────────────────
        let effects: Vec<Box<dyn ComputeEffect>> = vec![
            Box::new(WgpuEffect::new(
                "brightness".into(),
                EffectKind::Brightness,
                device.clone(),
                queue.clone(),
                brightness_shader,
                "main",
            )?),
            Box::new(WgpuEffect::new(
                "contrast".into(),
                EffectKind::Contrast,
                device.clone(),
                queue.clone(),
                contrast_shader,
                "main",
            )?),
            Box::new(WgpuEffect::new(
                "grayscale".into(),
                EffectKind::Grayscale,
                device.clone(),
                queue.clone(),
                grayscale_shader,
                "main",
            )?),
            Box::new(WgpuEffect::new(
                "invert".into(),
                EffectKind::Invert,
                device.clone(),
                queue.clone(),
                invert_shader,
                "main",
            )?),
            Box::new(WgpuEffect::new(
                "blur".into(),
                EffectKind::Blur,
                device.clone(),
                queue.clone(),
                blur_shader,
                "main",
            )?),
            Box::new(WgpuEffect::new(
                "sharpen".into(),
                EffectKind::Sharpen,
                device.clone(),
                queue.clone(),
                sharpen_shader,
                "main",
            )?),
            Box::new(WgpuEffect::new(
                "edge_detect".into(),
                EffectKind::EdgeDetect,
                device.clone(),
                queue.clone(),
                edge_detect_shader,
                "main",
            )?),
            Box::new(WgpuEffect::new(
                "sepia".into(),
                EffectKind::Sepia,
                device.clone(),
                queue.clone(),
                sepia_shader,
                "main",
            )?),
            Box::new(WgpuEffect::new(
                "threshold".into(),
                EffectKind::Threshold,
                device.clone(),
                queue.clone(),
                threshold_shader,
                "main",
            )?),
            Box::new(WgpuEffect::new(
                "box_blur".into(),
                EffectKind::BoxBlur,
                device.clone(),
                queue.clone(),
                box_blur_shader,
                "main",
            )?),
            Box::new(WgpuEffect::new(
                "emboss".into(),
                EffectKind::Emboss,
                device.clone(),
                queue.clone(),
                emboss_shader,
                "main",
            )?),
            Box::new(WgpuEffect::new(
                "pixelate".into(),
                EffectKind::Pixelate,
                device.clone(),
                queue.clone(),
                pixelate_shader,
                "main",
            )?),
        ];

        Ok(WgpuBackend {
            device,
            queue,
            effects,
        })
    }
}

impl ComputeBackend for WgpuBackend {
    fn name(&self) -> &str {
        "wgpu"
    }

    fn alloc_buffer(&self, size: usize) -> MediaResult<Box<dyn ComputeBuffer>> {
        let buf = WgpuBuffer::new(self.device.clone(), self.queue.clone(), size)?;
        Ok(Box::new(buf))
    }

    fn register_buffer(&self, data: &[u8]) -> MediaResult<Box<dyn ComputeBuffer>> {
        let buf = WgpuBuffer::from_slice(self.device.clone(), self.queue.clone(), data)?;
        Ok(Box::new(buf))
    }

    fn available_effects(&self) -> Vec<media_compute::kernel::EffectDesc> {
        self.effects
            .iter()
            .map(|e| {
                media_compute::kernel::EffectDesc::new(match e.name() {
                    "brightness" => EffectKind::Brightness,
                    "contrast" => EffectKind::Contrast,
                    "grayscale" => EffectKind::Grayscale,
                    "invert" => EffectKind::Invert,
                    "blur" => EffectKind::Blur,
                    "sharpen" => EffectKind::Sharpen,
                    "edge_detect" => EffectKind::EdgeDetect,
                    "sepia" => EffectKind::Sepia,
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
            EffectKind::EdgeDetect => e.name() == "edge_detect",
            EffectKind::Sepia => e.name() == "sepia",
            EffectKind::Threshold => e.name() == "threshold",
            EffectKind::BoxBlur => e.name() == "box_blur",
            EffectKind::Emboss => e.name() == "emboss",
            EffectKind::Pixelate => e.name() == "pixelate",
            EffectKind::Custom(name) => e.name() == name,
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
                EffectKind::EdgeDetect => e.name() == "edge_detect",
                EffectKind::Sepia => e.name() == "sepia",
                EffectKind::Threshold => e.name() == "threshold",
                EffectKind::BoxBlur => e.name() == "box_blur",
                EffectKind::Emboss => e.name() == "emboss",
                EffectKind::Pixelate => e.name() == "pixelate",
                EffectKind::Custom(name) => e.name() == name,
            })
            .map(|b| b.as_ref())
    }
}
