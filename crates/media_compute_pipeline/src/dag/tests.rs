//! DAG integration tests — validation, serialisation, builder,
//! and parallel execution with a mock backend.

use media_compute::{EffectDesc, EffectKind, EffectParam};
use media_core::error::MediaResult;
use media_core::pixel_format::PixelFormat;
use media_core::Frame;

use crate::dag::{DagBuilder, DagNode, DagNodeDef, ParamDef};
use crate::EffectProvider;

// ---------------------------------------------------------------------------
// Mock backend (returns transformed frames)
// ---------------------------------------------------------------------------

/// Minimal mock that simulates effect application.
struct MockBackend;

impl EffectProvider for MockBackend {
    fn get_effect(&self, _kind: &EffectKind) -> Option<&dyn media_compute::ComputeEffect> {
        // Return a simple identity-like effect for any kind.
        use media_compute::ComputeEffect;
        struct Id;
        impl ComputeEffect for Id {
            fn apply(&self, src: &Frame, _desc: &EffectDesc) -> MediaResult<Frame> {
                Ok(src.clone())
            }
            fn name(&self) -> &str {
                "mock-identity"
            }
        }
        // We return the same static instance for any request.
        static ID: Id = Id;
        Some(&ID)
    }
}

// ---------------------------------------------------------------------------
// Validation tests
// ---------------------------------------------------------------------------

#[test]
fn test_empty_dag() {
    let dag = DagBuilder::new().build().unwrap();
    assert!(dag.nodes().is_empty());
    assert!(dag.output_names().is_empty());
}

#[test]
fn test_single_node() {
    let dag = DagBuilder::new()
        .add(DagNode::new("a", EffectKind::Grayscale))
        .build()
        .unwrap();
    assert_eq!(dag.nodes().len(), 1);
    assert_eq!(dag.output_names(), vec!["a"]);
}

#[test]
fn test_chain() {
    let dag = DagBuilder::new()
        .add(DagNode::new("a", EffectKind::Grayscale))
        .add(
            DagNode::new("b", EffectKind::Blur)
                .with_param(EffectParam::Float(2.0))
                .depends_on(&["a"]),
        )
        .build()
        .unwrap();
    assert_eq!(dag.nodes().len(), 2);
    assert_eq!(dag.output_names(), vec!["b"]);
}

#[test]
fn test_diamond() {
    let dag = DagBuilder::new()
        .add(DagNode::new("input", EffectKind::Invert))
        .add(DagNode::new("blur_a", EffectKind::Blur).depends_on(&["input"]))
        .add(DagNode::new("blur_b", EffectKind::Blur).depends_on(&["input"]))
        .add(DagNode::new("final", EffectKind::Sharpen).depends_on(&["blur_a", "blur_b"]))
        .build()
        .unwrap();
    assert_eq!(dag.nodes().len(), 4);
    // Both blur_a and blur_b are leaves (no children — final depends on them
    // but they don't depend on each other), so they are "outputs" per our
    // definition (no outgoing edges).  Actually `output_names` returns
    // nodes with no children.  "final" has children?  Let's check.
    //   input → blur_a → final
    //        ↘ blur_b ↗
    // children: input=[blur_a,blur_b], blur_a=[final], blur_b=[final], final=[]
    // So output_names = ["final"].
    assert_eq!(dag.output_names(), vec!["final"]);
}

#[test]
fn test_cycle_detected() {
    let err = DagBuilder::new()
        .add(DagNode::new("a", EffectKind::Invert))
        .add(DagNode::new("b", EffectKind::Blur).depends_on(&["c"]))
        .add(DagNode::new("c", EffectKind::Grayscale).depends_on(&["b"]))
        .build()
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("cycle") || msg.contains("Cycle"),
        "expected cycle error, got: {msg}"
    );
}

#[test]
fn test_missing_dep() {
    let err = DagBuilder::new()
        .add(DagNode::new("a", EffectKind::Blur).depends_on(&["nonexistent"]))
        .build()
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("nonexistent"),
        "expected missing-dependency error, got: {msg}"
    );
}

#[test]
fn test_duplicate_name() {
    let err = DagBuilder::new()
        .add(DagNode::new("dup", EffectKind::Grayscale))
        .add(DagNode::new("dup", EffectKind::Invert))
        .build()
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("dup"),
        "expected duplicate-name error, got: {msg}"
    );
}

#[test]
fn test_self_cycle() {
    let err = DagBuilder::new()
        .add(DagNode::new("a", EffectKind::Invert).depends_on(&["a"]))
        .build()
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("cycle") || msg.contains("self"),
        "expected cycle/self-dependency error, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Serialisation round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_json_roundtrip() {
    let defs = vec![
        DagNodeDef {
            name: "bright".into(),
            kind: "brightness".into(),
            params: vec![ParamDef::F64(0.5)],
            depends_on: vec![],
        },
        DagNodeDef {
            name: "blurred".into(),
            kind: "blur".into(),
            params: vec![ParamDef::F64(3.0)],
            depends_on: vec!["bright".into()],
        },
        DagNodeDef {
            name: "custom_fx".into(),
            kind: "my_filter".into(),
            params: vec![],
            depends_on: vec!["blurred".into()],
        },
    ];

    let json = serde_json::to_string_pretty(&defs).unwrap();
    let back: Vec<DagNodeDef> = serde_json::from_str(&json).unwrap();
    assert_eq!(back.len(), 3);
    assert_eq!(back[0].name, "bright");
    assert_eq!(back[1].kind, "blur");
    assert_eq!(back[2].kind, "my_filter");

    // Convert to real DagNodes and build DAG.
    let nodes: Vec<_> = back.into_iter().map(|d| d.into()).collect();
    let dag = crate::dag::ComputeDag::from_nodes(nodes).unwrap();
    assert_eq!(dag.nodes().len(), 3);
    assert_eq!(dag.output_names(), vec!["custom_fx"]);
}

#[test]
fn test_param_roundtrip_all_types() {
    // Single node with every param type.
    let def = DagNodeDef {
        name: "p".into(),
        kind: "sharpen".into(),
        // Order in Def doesn't matter — we assert on the converted DagNode.
        params: vec![ParamDef::U64(42), ParamDef::I64(-5), ParamDef::F64(1.0)],
        depends_on: vec![],
    };
    let json = serde_json::to_string(&def).unwrap();
    let back: DagNodeDef = serde_json::from_str(&json).unwrap();
    assert_eq!(back.params.len(), 3);

    let node: DagNode = back.into();
    assert_eq!(node.params.len(), 3);
    // Check conversions: U64→U32, I64→Int, F64→Float
    assert!(matches!(node.params[0], EffectParam::U32(42)));
    assert!(matches!(node.params[1], EffectParam::Int(-5)));
    assert!(matches!(node.params[2], EffectParam::Float(1.0)));
}

// ---------------------------------------------------------------------------
// Execution (requires Frame + mock backend)
// ---------------------------------------------------------------------------

/// Create a small test frame.
fn test_frame() -> Frame {
    Frame::new_cpu(64, 48, PixelFormat::Rgba8).unwrap()
}

#[test]
fn test_execute_single_node() {
    let dag = DagBuilder::new()
        .add(DagNode::new("out", EffectKind::Grayscale))
        .build()
        .unwrap();
    let backend = MockBackend;
    let result = dag.apply(&test_frame(), &backend).unwrap();
    assert_eq!(result.outputs.len(), 1);
    assert!(result.outputs.contains_key("out"));
    assert_eq!(result.stats.per_node.len(), 1);
    assert_eq!(result.stats.per_node[0].name, "out");
}

#[test]
fn test_execute_chain() {
    let dag = DagBuilder::new()
        .add(DagNode::new("a", EffectKind::Grayscale))
        .add(
            DagNode::new("b", EffectKind::Blur)
                .with_param(EffectParam::Float(3.0))
                .depends_on(&["a"]),
        )
        .build()
        .unwrap();
    let backend = MockBackend;
    let result = dag.apply(&test_frame(), &backend).unwrap();
    assert_eq!(result.outputs.len(), 1);
    assert!(result.outputs.contains_key("b"));
    assert_eq!(result.stats.per_node.len(), 2);
}

#[test]
fn test_execute_diamond() {
    let dag = DagBuilder::new()
        .add(DagNode::new("input", EffectKind::Invert))
        .add(
            DagNode::new("left", EffectKind::Blur)
                .with_param(EffectParam::Float(2.0))
                .depends_on(&["input"]),
        )
        .add(
            DagNode::new("right", EffectKind::Blur)
                .with_param(EffectParam::Float(4.0))
                .depends_on(&["input"]),
        )
        .add(DagNode::new("out", EffectKind::Sharpen).depends_on(&["left", "right"]))
        .build()
        .unwrap();
    let backend = MockBackend;
    let result = dag.apply(&test_frame(), &backend).unwrap();
    assert_eq!(result.outputs.len(), 1);
    assert!(result.outputs.contains_key("out"));
    // Diamond has 4 nodes.
    assert_eq!(result.stats.per_node.len(), 4);
}

#[test]
fn test_execute_multiple_outputs() {
    let dag = DagBuilder::new()
        .add(DagNode::new("a", EffectKind::Grayscale))
        .add(
            DagNode::new("b1", EffectKind::Blur)
                .with_param(EffectParam::Float(1.0))
                .depends_on(&["a"]),
        )
        .add(DagNode::new("b2", EffectKind::Sharpen).depends_on(&["a"]))
        .build()
        .unwrap();
    let backend = MockBackend;
    let result = dag.apply(&test_frame(), &backend).unwrap();
    assert_eq!(result.outputs.len(), 2);
    assert!(result.outputs.contains_key("b1"));
    assert!(result.outputs.contains_key("b2"));
}

#[test]
fn test_execute_empty() {
    let dag = DagBuilder::new().build().unwrap();
    let backend = MockBackend;
    let result = dag.apply(&test_frame(), &backend).unwrap();
    assert!(result.outputs.is_empty());
    assert!(result.stats.per_node.is_empty());
    assert_eq!(result.stats.total.as_nanos(), 0);
}
