//! WGPU effect implementations using compute shaders.
//!
//! Each effect creates a WGSL compute pipeline with a 2-binding layout:
//! - binding 0: read-write storage buffer (pixel data in BGRA8 u32 format)
//! - binding 1: uniform buffer  (width, height, param0, param1)

use crate::buffer::WgpuBuffer;
use media_compute::{ComputeBuffer, ComputeEffect, EffectDesc, EffectKind, EffectParam};
use media_core::error::{MediaError, MediaResult};
use media_core::Frame;
use std::mem::size_of;
use std::sync::Arc;

/// CPU-side representation of the WGSL `Uniforms` struct.
///
/// Layout (16 bytes, no padding):
/// | offset | field   | type |
/// |--------|---------|------|
/// | 0      | width   | u32  |
/// | 4      | height  | u32  |
/// | 8      | param0  | f32  |
/// | 12     | param1  | f32  |
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct EffectUniforms {
    width: u32,
    height: u32,
    param0: f32,
    param1: f32,
}

// SAFETY: EffectUniforms is a plain-old-data struct with no padding; all
// field types (u32, f32) are themselves Pod / Zeroable.
unsafe impl bytemuck::Zeroable for EffectUniforms {}
unsafe impl bytemuck::Pod for EffectUniforms {}

/// Wrapper for a WGPU compute pipeline acting as an effect.
pub struct WgpuEffect {
    name: String,
    #[allow(dead_code)]
    kind: EffectKind,
    pipeline: wgpu::ComputePipeline,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl WgpuEffect {
    pub fn new(
        name: String,
        kind: EffectKind,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        shader_source: &str,
        entry_point: &str,
    ) -> MediaResult<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{name} shader")),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{name} bind group layout")),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{name} pipeline layout")),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{name} pipeline")),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry_point),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Ok(WgpuEffect {
            name,
            kind,
            pipeline,
            device,
            queue,
            bind_group_layout,
        })
    }

    /// Create a uniform buffer + bind group for a single dispatch.
    ///
    /// The returned `wgpu::Buffer` must remain alive until after
    /// `queue.submit()` finishes (wgpu uses internal ref-counting so
    /// a local variable held until after submit is sufficient).
    fn create_dispatch_bind_group(
        &self,
        storage: &wgpu::Buffer,
        width: u32,
        height: u32,
        params: &[EffectParam],
    ) -> (wgpu::BindGroup, wgpu::Buffer) {
        let param0 = params
            .first()
            .map(|p| match p {
                EffectParam::Float(v) => *v,
                EffectParam::Int(v) => *v as f32,
                EffectParam::U32(v) => *v as f32,
            })
            .unwrap_or(0.0);
        let param1 = params
            .get(1)
            .map(|p| match p {
                EffectParam::Float(v) => *v,
                EffectParam::Int(v) => *v as f32,
                EffectParam::U32(v) => *v as f32,
            })
            .unwrap_or(0.0);

        let uniforms = EffectUniforms {
            width,
            height,
            param0,
            param1,
        };

        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} uniforms", self.name)),
            size: size_of::<EffectUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} bind group", self.name)),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        (bind_group, uniform_buf)
    }
}

impl ComputeEffect for WgpuEffect {
    fn name(&self) -> &str {
        &self.name
    }

    fn apply(&self, src: &Frame, desc: &EffectDesc) -> MediaResult<Frame> {
        let mut dst = src.clone();

        let width = src.width;
        let height = src.height;
        let cpu = src
            .as_cpu()
            .ok_or_else(|| MediaError::Other("WGPU effect requires a CPU-backed frame".into()))?;
        let data_len = cpu.data.len();

        // Allocate device buffer and copy data.
        let mut dev_buf = WgpuBuffer::new(self.device.clone(), self.queue.clone(), data_len)?;
        dev_buf.host.copy_from_slice(&cpu.data);
        dev_buf.copy_to_device()?;

        // Create bind group with uniform buffer containing width/height/params.
        let (bind_group, _uniform_buf) =
            self.create_dispatch_bind_group(dev_buf.gpu_buffer(), width, height, &desc.params);

        // Create command encoder and dispatch.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{} encoder", self.name)),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{} pass", self.name)),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);

            // Dispatch with workgroup sizes matching shader expectations.
            let workgroup_size = 16u32;
            let grid_x = (width + workgroup_size - 1) / workgroup_size;
            let grid_y = (height + workgroup_size - 1) / workgroup_size;
            cpass.dispatch_workgroups(grid_x, grid_y, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        // Read back.
        let result = dev_buf.read_into_vec()?;
        if let Some(cpu) = dst.as_cpu_mut() {
            cpu.data = result;
        }
        Ok(dst)
    }

    fn supports_in_place(&self) -> bool {
        true
    }

    fn apply_in_place(&self, frame: &mut Frame, desc: &EffectDesc) -> MediaResult<()> {
        let width = frame.width;
        let height = frame.height;
        let cpu = frame
            .as_cpu()
            .ok_or_else(|| MediaError::Other("WGPU effect requires a CPU-backed frame".into()))?;
        let data_len = cpu.data.len();

        let mut dev_buf = WgpuBuffer::new(self.device.clone(), self.queue.clone(), data_len)?;
        dev_buf.host.copy_from_slice(&cpu.data);
        dev_buf.copy_to_device()?;

        let (bind_group, _uniform_buf) =
            self.create_dispatch_bind_group(dev_buf.gpu_buffer(), width, height, &desc.params);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{} encoder", self.name)),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{} pass", self.name)),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);

            let workgroup_size = 16u32;
            let grid_x = (width + workgroup_size - 1) / workgroup_size;
            let grid_y = (height + workgroup_size - 1) / workgroup_size;
            cpass.dispatch_workgroups(grid_x, grid_y, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        let result = dev_buf.read_into_vec()?;
        if let Some(cpu) = frame.as_cpu_mut() {
            cpu.data = result;
        }
        Ok(())
    }
}
