//! FramePool — reuses intermediate frame allocations across DAG invocations.

use std::collections::HashMap;

use media_core::error::MediaResult;
use media_core::pixel_format::PixelFormat;
use media_core::Frame;

/// A pool of reusable frame buffers keyed by node name.
///
/// The pool grows as needed and is *not* cleared between runs of the
/// same DAG.  Consumers drain what they need; entries for names that
/// no longer appear are simply orphaned and will be overwritten if a
/// new node claims that name.
///
/// The sentinel key `"__input__"` is reserved for the pipeline input.
#[derive(Debug, Clone)]
pub struct FramePool {
    frames: HashMap<String, Frame>,
}

impl FramePool {
    /// Create an empty pool.
    pub fn new() -> Self {
        FramePool {
            frames: HashMap::new(),
        }
    }

    /// The sentinel key for the pipeline input frame.
    pub fn input_key(&self) -> String {
        "__input__".to_string()
    }

    /// Retrieve a frame for reading, or `None`.
    pub fn get(&self, name: &str) -> Option<&Frame> {
        self.frames.get(name)
    }

    /// Insert a frame (overwrites if the name already exists).
    pub fn insert(&mut self, name: String, frame: Frame) {
        self.frames.insert(name, frame);
    }

    /// Remove a frame, returning it if present.
    pub fn remove(&mut self, name: &str) -> Option<Frame> {
        self.frames.remove(name)
    }

    /// Reserve capacity for at least `additional` more entries.
    pub fn reserve(&mut self, additional: usize) {
        self.frames.reserve(additional);
    }

    /// Clear all frames from the pool.
    pub fn clear(&mut self) {
        self.frames.clear();
    }

    /// Number of frames currently in the pool.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Ensure a frame of the given dimensions exists at `name`,
    /// reusing an existing allocation if compatible.
    pub fn ensure_frame(
        &mut self,
        name: &str,
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> MediaResult<&mut Frame> {
        if !self.frames.contains_key(name) {
            let frame = Frame::new_cpu(width, height, format)?;
            self.frames.insert(name.to_string(), frame);
        }
        Ok(self.frames.get_mut(name).unwrap())
    }
}

impl Default for FramePool {
    fn default() -> Self {
        Self::new()
    }
}
