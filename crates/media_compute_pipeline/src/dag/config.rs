//! # Pipeline configuration — load pipeline DAGs from JSON/TOML files.
//!
//! [`PipelineConfig`] is a serialisable description of a compute pipeline
//! that can be loaded from disk and converted into an executable
//! [`ComputeDag`](super::ComputeDag).
//!
//! ## Example (TOML)
//!
//! ```toml
//! input = "photo.png"
//! output = "result.png"
//!
//! [[nodes]]
//! name = "gray"
//! kind = "Grayscale"
//!
//! [[nodes]]
//! name = "edges"
//! kind = "EdgeDetect"
//! depends_on = ["gray"]
//!
//! [[nodes]]
//! name = "brighten"
//! kind = "Brightness"
//! params = [0.2]
//! depends_on = ["edges"]
//! ```

use serde::{Deserialize, Serialize};

use crate::dag::{ComputeDag, DagNode, DagNodeDef};
use media_core::error::MediaResult;

/// A serialisable pipeline configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Optional input file path (can be overridden on the CLI).
    #[serde(default)]
    pub input: Option<String>,

    /// Optional output file path (can be overridden on the CLI).
    #[serde(default)]
    pub output: Option<String>,

    /// Name of the node whose output should be saved.
    ///
    /// If omitted, the last node in the list is used.
    #[serde(default)]
    pub output_node: Option<String>,

    /// Pipeline nodes.
    pub nodes: Vec<DagNodeDef>,
}

impl PipelineConfig {
    /// Build and validate a [`ComputeDag`] from this configuration.
    pub fn build_dag(&self) -> MediaResult<ComputeDag> {
        let dag_nodes: Vec<DagNode> = self.nodes.iter().map(|def| def.clone().into()).collect();
        ComputeDag::from_nodes(dag_nodes)
    }

    /// Load a [`PipelineConfig`] from a JSON string.
    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }

    /// Load a [`PipelineConfig`] from a TOML string.
    pub fn from_toml(toml: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(toml)
    }

    /// Serialise this config to pretty-printed JSON.
    pub fn to_json_pretty(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// Determine the output node name.
    ///
    /// Returns `output_node` if set, otherwise the name of the last node.
    pub fn output_name(&self) -> Option<&str> {
        self.output_node
            .as_deref()
            .or_else(|| self.nodes.last().map(|n| n.name.as_str()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_json_roundtrip() {
        let json = r#"{
            "input": "in.png",
            "output": "out.png",
            "nodes": [
                {"name": "bright", "kind": "Brightness", "params": [0.1]},
                {"name": "blur", "kind": "Blur", "depends_on": ["bright"]}
            ]
        }"#;
        let cfg: PipelineConfig = PipelineConfig::from_json(json).unwrap();
        assert_eq!(cfg.input.as_deref(), Some("in.png"));
        assert_eq!(cfg.output.as_deref(), Some("out.png"));
        assert_eq!(cfg.nodes.len(), 2);
        assert_eq!(cfg.nodes[0].name, "bright");
        assert_eq!(cfg.nodes[1].depends_on, vec!["bright"]);
    }

    #[test]
    fn test_from_toml() {
        let toml = r#"
input = "in.png"
output = "out.png"

[[nodes]]
name = "gray"
kind = "Grayscale"

[[nodes]]
name = "sharpen"
kind = "Sharpen"
params = [0.5]
depends_on = ["gray"]
"#;
        let cfg: PipelineConfig = PipelineConfig::from_toml(toml).unwrap();
        assert_eq!(cfg.input.as_deref(), Some("in.png"));
        assert_eq!(cfg.nodes.len(), 2);
        assert_eq!(cfg.nodes[1].params.len(), 1);
    }

    #[test]
    fn test_output_name_defaults_to_last() {
        let cfg = PipelineConfig {
            input: None,
            output: None,
            output_node: None,
            nodes: vec![
                DagNodeDef {
                    name: "a".into(),
                    kind: "Brightness".into(),
                    params: vec![],
                    depends_on: vec![],
                },
                DagNodeDef {
                    name: "b".into(),
                    kind: "Blur".into(),
                    params: vec![],
                    depends_on: vec!["a".into()],
                },
            ],
        };
        assert_eq!(cfg.output_name(), Some("b"));
    }

    #[test]
    fn test_output_name_explicit() {
        let cfg = PipelineConfig {
            input: None,
            output: None,
            output_node: Some("a".into()),
            nodes: vec![
                DagNodeDef {
                    name: "a".into(),
                    kind: "Brightness".into(),
                    params: vec![],
                    depends_on: vec![],
                },
                DagNodeDef {
                    name: "b".into(),
                    kind: "Blur".into(),
                    params: vec![],
                    depends_on: vec!["a".into()],
                },
            ],
        };
        assert_eq!(cfg.output_name(), Some("a"));
    }

    #[test]
    fn test_build_dag_ok() {
        let cfg = PipelineConfig {
            input: None,
            output: None,
            output_node: None,
            nodes: vec![
                DagNodeDef {
                    name: "a".into(),
                    kind: "Brightness".into(),
                    params: vec![],
                    depends_on: vec![],
                },
                DagNodeDef {
                    name: "b".into(),
                    kind: "Grayscale".into(),
                    params: vec![],
                    depends_on: vec!["a".into()],
                },
            ],
        };
        let dag = cfg.build_dag().unwrap();
        assert_eq!(dag.nodes().len(), 2);
        assert!(!dag.output_names().is_empty());
    }

    #[test]
    fn test_build_dag_cycle_fails() {
        let cfg = PipelineConfig {
            input: None,
            output: None,
            output_node: None,
            nodes: vec![
                DagNodeDef {
                    name: "a".into(),
                    kind: "Brightness".into(),
                    params: vec![],
                    depends_on: vec!["b".into()],
                },
                DagNodeDef {
                    name: "b".into(),
                    kind: "Blur".into(),
                    params: vec![],
                    depends_on: vec!["a".into()],
                },
            ],
        };
        assert!(cfg.build_dag().is_err());
    }
}
