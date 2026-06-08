//! WGPU device buffer implementing `ComputeBuffer`.

use media_compute::ComputeBuffer;
use media_core::error::{MediaError, MediaResult};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Monotonically increasing buffer ID counter.
static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

/// A GPU-resident buffer backed by `wgpu::Buffer` with a host-side `Vec<u8>` mirror.
pub struct WgpuBuffer {
    /// Unique identifier for this buffer.
    buffer_id: u64,
    /// WGPU device (ref-counted so the buffer can outlive a single backend ref).
    device: Arc<wgpu::Device>,
    /// WGPU queue (ref-counted, same reason).
    queue: Arc<wgpu::Queue>,
    /// GPU-side buffer.
    gpu_buffer: wgpu::Buffer,
    /// Host-side data mirror.
    pub(crate) host: Vec<u8>,
    /// Allocation size in bytes.
    alloc_size: usize,
    /// Whether the host mirror is in sync with the device.
    mapped: bool,
}

impl WgpuBuffer {
    /// Allocate device memory of the given size and zero the host mirror.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        size: usize,
    ) -> MediaResult<Self> {
        let gpu_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WgpuBuffer"),
            size: size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(WgpuBuffer {
            buffer_id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            device,
            queue,
            gpu_buffer,
            host: vec![0u8; size],
            alloc_size: size,
            mapped: false,
        })
    }

    /// Create a buffer initialized from a byte slice (host → device).
    pub fn from_slice(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        data: &[u8],
    ) -> MediaResult<Self> {
        let mut buf = Self::new(device, queue, data.len())?;
        buf.host.copy_from_slice(data);
        buf.copy_to_device()?;
        Ok(buf)
    }

    /// The raw `&wgpu::Buffer` (for bind group creation).
    pub fn gpu_buffer(&self) -> &wgpu::Buffer {
        &self.gpu_buffer
    }

    /// Copy host mirror → device.
    pub fn copy_to_device(&self) -> MediaResult<()> {
        self.queue.write_buffer(&self.gpu_buffer, 0, &self.host);
        Ok(())
    }

    /// Copy device → host mirror.
    pub fn sync_host(&mut self) -> MediaResult<()> {
        // Create a staging buffer to read back.
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WgpuBuffer::sync_host staging"),
            size: self.alloc_size as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("WgpuBuffer::sync_host encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.gpu_buffer, 0, &staging, 0, self.alloc_size as u64);
        self.queue.submit(Some(encoder.finish()));

        // Map staging buffer for reading — this is synchronous via pollster.
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        self.host.copy_from_slice(&data);
        drop(data);
        staging.unmap();

        Ok(())
    }
}

impl ComputeBuffer for WgpuBuffer {
    fn size(&self) -> usize {
        self.alloc_size
    }

    fn id(&self) -> u64 {
        self.buffer_id
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
            // Read directly from GPU via staging buffer.
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("WgpuBuffer::read_into_vec staging"),
                size: self.alloc_size as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("WgpuBuffer::read_into_vec encoder"),
                });
            encoder.copy_buffer_to_buffer(&self.gpu_buffer, 0, &staging, 0, self.alloc_size as u64);
            self.queue.submit(Some(encoder.finish()));

            let slice = staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.device.poll(wgpu::Maintain::Wait);

            let data = slice.get_mapped_range();
            let result = data.to_vec();
            drop(data);
            staging.unmap();

            Ok(result)
        }
    }

    fn write_from_slice(&mut self, offset: usize, data: &[u8]) -> MediaResult<()> {
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
            // Write directly to device buffer.
            self.queue
                .write_buffer(&self.gpu_buffer, offset as u64, data);
        }
        Ok(())
    }

    fn copy_to(&self, dst: &mut dyn ComputeBuffer, range: Range<usize>) -> MediaResult<()> {
        let data = self.read_into_vec()?;
        dst.write_from_slice(range.start, &data[range])?;
        Ok(())
    }
}

// SAFETY: WGPU buffers are thread-safe.
unsafe impl Send for WgpuBuffer {}
unsafe impl Sync for WgpuBuffer {}
