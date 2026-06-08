//! # Media Compute Pipeline
//!
//! Composes multiple compute effects into a single processing
//! pipeline.  Two execution modes are provided:
//!
//! 1. **Sequential** — [`ComputePipeline`] applies steps one after
//!    another, suitable for simple effect chains.
//! 2. **DAG** — [`dag::ComputeDag`] models a directed acyclic graph of
//!    named nodes with explicit dependencies and executes them with
//!    parallel branch scheduling.

pub mod dag;
pub mod pool;

use media_compute::{ComputeEffect, EffectDesc, EffectKind};
use media_core::error::{MediaError, MediaResult};
use media_core::Frame;

/// A sequence of effects to apply to a frame.
pub struct PipelineStep {
    /// Which effect to apply.
    pub kind: EffectKind,
    /// Per-effect parameters.
    pub params: Vec<media_compute::EffectParam>,
}

/// An ordered list of `PipelineStep`s that is applied sequentially.
pub struct ComputePipeline {
    steps: Vec<PipelineStep>,
}

impl ComputePipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        ComputePipeline { steps: Vec::new() }
    }

    /// Create a pipeline from a list of steps.
    pub fn from_steps(steps: Vec<PipelineStep>) -> Self {
        ComputePipeline { steps }
    }

    /// Add a step to the end of the pipeline.
    pub fn add_step(&mut self, step: PipelineStep) {
        self.steps.push(step);
    }

    /// Apply the whole pipeline to `src` using the given effect provider.
    ///
    /// `effects` can be any type that implements `EffectProvider` —
    /// typically a `ComputeBackend`.
    pub fn apply(&self, src: &Frame, effects: &dyn EffectProvider) -> MediaResult<Frame> {
        if self.steps.is_empty() {
            return Ok(src.clone());
        }

        let mut frame = src.clone();
        for step in &self.steps {
            let effect = effects.get_effect(&step.kind).ok_or_else(|| {
                MediaError::Other(format!("Effect {:?} not available in pipeline", step.kind,))
            })?;

            let mut desc = EffectDesc::new(step.kind.clone());
            for p in &step.params {
                use media_compute::EffectParam;
                desc = desc.with(match p {
                    EffectParam::Float(v) => EffectParam::Float(*v),
                    EffectParam::Int(v) => EffectParam::Int(*v),
                    EffectParam::U32(v) => EffectParam::U32(*v),
                });
            }
            frame = effect.apply(&frame, &desc)?;
        }
        Ok(frame)
    }

    /// Apply the whole pipeline in-place.
    pub fn apply_in_place(
        &self,
        frame: &mut Frame,
        effects: &dyn EffectProvider,
    ) -> MediaResult<()> {
        if self.steps.is_empty() {
            return Ok(());
        }

        for step in &self.steps {
            let effect = effects.get_effect(&step.kind).ok_or_else(|| {
                MediaError::Other(format!("Effect {:?} not available in pipeline", step.kind,))
            })?;

            let mut desc = EffectDesc::new(step.kind.clone());
            for p in &step.params {
                use media_compute::EffectParam;
                desc = desc.with(match p {
                    EffectParam::Float(v) => EffectParam::Float(*v),
                    EffectParam::Int(v) => EffectParam::Int(*v),
                    EffectParam::U32(v) => EffectParam::U32(*v),
                });
            }
            if effect.supports_in_place() {
                effect.apply_in_place(frame, &desc)?;
            } else {
                // Fall back to copy-based apply.
                *frame = effect.apply(frame, &desc)?;
            }
        }
        Ok(())
    }
}

/// Trait for types that can provide effects by kind.
///
/// # Thread safety
///
/// `EffectProvider` requires `Sync` so that a single shared reference
/// can be used from multiple rayon-scoped tasks concurrently.  The
/// blanket impl for `ComputeBackend` satisfies this automatically
/// because `ComputeBackend: Send + Sync`.
pub trait EffectProvider: Sync {
    /// Get an effect by its kind, if available.
    fn get_effect(&self, kind: &EffectKind) -> Option<&dyn ComputeEffect>;

    /// Convenience: look up an effect by kind and apply it in one call.
    fn apply_effect(
        &self,
        kind: &EffectKind,
        src: &Frame,
        desc: &EffectDesc,
    ) -> MediaResult<Frame> {
        let effect = self
            .get_effect(kind)
            .ok_or_else(|| MediaError::Other(format!("Effect {:?} not available", kind)))?;
        effect.apply(src, desc)
    }
}

// Blanket impl for any `ComputeBackend`.
impl<T: media_compute::ComputeBackend> EffectProvider for T {
    fn get_effect(&self, kind: &EffectKind) -> Option<&dyn ComputeEffect> {
        media_compute::ComputeBackend::get_effect(self, kind)
    }
}
