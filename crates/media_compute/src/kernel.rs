//! Kernel abstraction for compute operations.

/// A descriptor for a compiled kernel.
#[derive(Debug, Clone)]
pub struct KernelDesc {
    pub name: String,
    pub source: Option<String>,
    pub entry_point: String,
}

/// Trait for a compiled kernel that can be dispatched.
pub trait Kernel: Send + Sync {
    fn name(&self) -> &str;
    fn desc(&self) -> &KernelDesc;
}

/// Trait for objects that can serve as kernel arguments.
pub unsafe trait KernelArg: Send + Sync {}

// Safe implementations for basic types.
unsafe impl KernelArg for i32 {}
unsafe impl KernelArg for u32 {}
unsafe impl KernelArg for f32 {}
unsafe impl KernelArg for f64 {}
unsafe impl KernelArg for usize {}
unsafe impl KernelArg for (u32, u32) {}
unsafe impl KernelArg for (u32, u32, u32) {}

/// A set of kernel arguments (boxed to erase type).
pub struct Args {
    pub(crate) inner: Vec<Box<dyn std::any::Any + Send>>,
}

impl Args {
    pub fn new() -> Self {
        Args { inner: Vec::new() }
    }

    pub fn push<T: 'static + Send>(mut self, arg: T) -> Self {
        self.inner.push(Box::new(arg));
        self
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get a reference to an argument by index (downcasted).
    pub fn get<T: 'static + Clone + Send>(&self, index: usize) -> Option<T> {
        self.inner.get(index)?.downcast_ref::<T>().cloned()
    }
}

impl Default for Args {
    fn default() -> Self {
        Self::new()
    }
}

/// Half of a domain x/y (e.g. 16x16 threads per block, or 256 threads flat).
#[derive(Debug, Clone, Copy)]
pub struct LaunchConfig {
    pub grid_x: u32,
    pub grid_y: u32,
    pub grid_z: u32,
    pub block_x: u32,
    pub block_y: u32,
    pub block_z: u32,
}

impl LaunchConfig {
    pub fn flat(threads: u32) -> Self {
        LaunchConfig {
            grid_x: threads,
            grid_y: 1,
            grid_z: 1,
            block_x: 64,
            block_y: 1,
            block_z: 1,
        }
    }

    pub fn xy(width: u32, height: u32, block_x: u32, block_y: u32) -> Self {
        LaunchConfig {
            grid_x: (width + block_x - 1) / block_x,
            grid_y: (height + block_y - 1) / block_y,
            grid_z: 1,
            block_x,
            block_y,
            block_z: 1,
        }
    }
}

/// Compute effect kinds for standard image operations.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EffectKind {
    Brightness,
    Contrast,
    Grayscale,
    Invert,
    Blur,
    Sharpen,
    Custom(String),
}

/// Parameters for a specific effect.
#[derive(Debug, Clone)]
pub enum EffectParam {
    Float(f32),
    Int(i32),
    U32(u32),
}

impl From<f32> for EffectParam {
    fn from(v: f32) -> Self {
        EffectParam::Float(v)
    }
}

impl From<i32> for EffectParam {
    fn from(v: i32) -> Self {
        EffectParam::Int(v)
    }
}

impl From<u32> for EffectParam {
    fn from(v: u32) -> Self {
        EffectParam::U32(v)
    }
}

/// An effect descriptor.
#[derive(Debug, Clone)]
pub struct EffectDesc {
    pub kind: EffectKind,
    pub params: Vec<EffectParam>,
}

impl EffectDesc {
    pub fn new(kind: EffectKind) -> Self {
        EffectDesc {
            kind,
            params: Vec::new(),
        }
    }

    pub fn with(mut self, param: impl Into<EffectParam>) -> Self {
        self.params.push(param.into());
        self
    }
}
