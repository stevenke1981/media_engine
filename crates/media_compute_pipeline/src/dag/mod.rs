//! # DAG Pipeline — directed acyclic graph of compute effects.
//!
//! Extends the basic [`ComputePipeline`](crate::ComputePipeline) with
//! named stages, dependency declarations, topological scheduling,
//! parallel branch execution, and per-step timing statistics.

use std::collections::HashMap;
use std::time::Duration;

use media_compute::{EffectKind, EffectParam};
use media_core::error::MediaResult;
use media_core::Frame;

use crate::EffectProvider;

pub mod config;
pub mod execute;
pub mod validate;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A named node in the processing graph.
///
/// Each node declares which earlier node outputs it depends on.
/// Empty `depends_on` means the node reads from the pipeline's
/// input frame.
#[derive(Debug, Clone)]
pub struct DagNode {
    /// Unique output name for this node.
    pub name: String,
    /// Which effect to instantiate.
    pub kind: EffectKind,
    /// Effect parameters.
    pub params: Vec<EffectParam>,
    /// Names of node outputs this node reads from.
    /// Empty = reads the pipeline input frame.
    pub depends_on: Vec<String>,
}

impl DagNode {
    /// Create a new DAG node with no dependencies.
    pub fn new(name: impl Into<String>, kind: EffectKind) -> Self {
        DagNode {
            name: name.into(),
            kind,
            params: Vec::new(),
            depends_on: Vec::new(),
        }
    }

    /// Add a parameter.
    pub fn with_param(mut self, param: impl Into<EffectParam>) -> Self {
        self.params.push(param.into());
        self
    }

    /// Declare dependency on one or more named nodes.
    pub fn depends_on(mut self, names: &[&str]) -> Self {
        self.depends_on = names.iter().map(|s| s.to_string()).collect();
        self
    }
}

// ---------------------------------------------------------------------------
// ComputeDag
// ---------------------------------------------------------------------------

/// A validated, topologically-sorted directed acyclic graph of compute
/// effects that can be executed with parallel branch scheduling.
#[derive(Debug, Clone)]
pub struct ComputeDag {
    /// User-supplied nodes.
    nodes: Vec<DagNode>,

    /// Validated topological order (indices into `nodes`).
    sorted: Vec<usize>,

    /// name → index.
    name_map: HashMap<String, usize>,

    /// Topological depth of each node (0 = reads pipeline input).
    depth: Vec<usize>,

    /// Children indices (reverse edges for output detection).
    children: Vec<Vec<usize>>,
}

impl ComputeDag {
    /// Build a DAG from a list of nodes, validating the graph.
    pub fn from_nodes(nodes: Vec<DagNode>) -> MediaResult<Self> {
        let mut dag = ComputeDag {
            nodes,
            sorted: Vec::new(),
            name_map: HashMap::new(),
            depth: Vec::new(),
            children: Vec::new(),
        };
        dag.rebuild()?;
        Ok(dag)
    }

    /// Re-validate and re-sort after modification.
    pub fn rebuild(&mut self) -> MediaResult<()> {
        let state = validate::validate(&self.nodes)?;
        self.sorted = state.sorted;
        self.name_map = state.name_map;
        self.depth = state.depth;
        self.children = state.children;
        Ok(())
    }

    // -- accessors ----------------------------------------------------------

    pub fn nodes(&self) -> &[DagNode] {
        &self.nodes
    }

    /// Names of nodes that have no children (i.e. pipeline outputs).
    pub fn output_names(&self) -> Vec<&str> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(i, _)| self.children[*i].is_empty())
            .map(|(_, n)| n.name.as_str())
            .collect()
    }

    // -- execution ----------------------------------------------------------

    /// Apply the full DAG, allocating fresh intermediate frames.
    pub fn apply(&self, input: &Frame, effects: &dyn EffectProvider) -> MediaResult<DagResult> {
        execute::apply_dag(self, input, effects)
    }

    /// Apply the DAG with a frame pool to reuse intermediate allocations.
    pub fn apply_with_pool(
        &self,
        input: &Frame,
        effects: &dyn EffectProvider,
        pool: &mut crate::pool::FramePool,
    ) -> MediaResult<DagResult> {
        execute::apply_dag_with_pool(self, input, effects, pool)
    }
}

// ---------------------------------------------------------------------------
// Results & statistics
// ---------------------------------------------------------------------------

/// Result of a DAG execution.
#[derive(Debug, Clone)]
pub struct DagResult {
    /// Named output frames (one per terminal node).
    pub outputs: HashMap<String, Frame>,
    /// Per-step and aggregate timing statistics.
    pub stats: DagStats,
}

/// Timing statistics for a single DAG execution.
#[derive(Debug, Clone)]
pub struct DagStats {
    /// Per-node timing, in topological order.
    pub per_node: Vec<NodeStat>,
    /// Total wall-clock duration (includes parallelism).
    pub total: Duration,
}

/// Timing for a single node.
#[derive(Debug, Clone)]
pub struct NodeStat {
    pub name: String,
    pub kind: EffectKind,
    pub duration: Duration,
}

impl DagStats {
    /// Empty statistics (for empty DAG).
    pub fn empty() -> Self {
        DagStats {
            per_node: Vec::new(),
            total: Duration::ZERO,
        }
    }
}

// ---------------------------------------------------------------------------
// Serialisation (serde)
// ---------------------------------------------------------------------------

/// JSON-friendly serialisation helper for [`DagNode`].
///
/// Fields are spelled as lowercase strings so JSON/TOML/YAML are
/// readable without Rust enum qualifiers.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DagNodeDef {
    pub name: String,
    pub kind: String,
    #[serde(default)]
    pub params: Vec<ParamDef>,
    #[serde(default)]
    pub depends_on: Vec<String>,
}

/// JSON-friendly parameter value.
///
/// Order matters for the untagged deserialiser: try integer types
/// first so that `42` maps to `U64` / `I64` and only falls through
/// to `F64` when the value contains a decimal point.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum ParamDef {
    U64(u64),
    I64(i64),
    F64(f64),
}

impl From<&ParamDef> for EffectParam {
    fn from(p: &ParamDef) -> Self {
        match p {
            ParamDef::U64(v) => EffectParam::U32(*v as u32),
            ParamDef::I64(v) => EffectParam::Int(*v as i32),
            ParamDef::F64(v) => EffectParam::Float(*v as f32),
        }
    }
}

impl From<DagNodeDef> for DagNode {
    fn from(def: DagNodeDef) -> Self {
        let kind = match def.kind.to_lowercase().as_str() {
            "brightness" => EffectKind::Brightness,
            "contrast" => EffectKind::Contrast,
            "grayscale" => EffectKind::Grayscale,
            "invert" => EffectKind::Invert,
            "blur" => EffectKind::Blur,
            "sharpen" => EffectKind::Sharpen,
            "edgedetect" | "edge_detect" | "edge-detect" => EffectKind::EdgeDetect,
            "sepia" => EffectKind::Sepia,
            "threshold" => EffectKind::Threshold,
            "boxblur" | "box_blur" | "box-blur" => EffectKind::BoxBlur,
            "emboss" => EffectKind::Emboss,
            "pixelate" => EffectKind::Pixelate,
            other => EffectKind::Custom(other.to_string()),
        };
        let params: Vec<EffectParam> = def.params.iter().map(EffectParam::from).collect();
        DagNode {
            name: def.name,
            kind,
            params,
            depends_on: def.depends_on,
        }
    }
}

// ---------------------------------------------------------------------------
// Builder helper
// ---------------------------------------------------------------------------

/// Convenience builder for constructing a [`ComputeDag`] incrementally.
#[derive(Debug, Default)]
pub struct DagBuilder {
    nodes: Vec<DagNode>,
}

impl DagBuilder {
    pub fn new() -> Self {
        DagBuilder { nodes: Vec::new() }
    }

    /// Add a node to the DAG.
    pub fn add(mut self, node: DagNode) -> Self {
        self.nodes.push(node);
        self
    }

    /// Build and validate the DAG.
    pub fn build(self) -> MediaResult<ComputeDag> {
        ComputeDag::from_nodes(self.nodes)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
