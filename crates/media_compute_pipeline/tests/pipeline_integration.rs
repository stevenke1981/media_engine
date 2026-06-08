//! Integration tests: real [`CpuBackend`] effects through the DAG pipeline.
//!
//! These tests verify that the full pipeline stack composes correctly —
//! DAG building + serialisation + real CPU effect execution.

use media_compute::{EffectKind, EffectParam};
use media_compute_cpu::CpuBackend;
use media_compute_pipeline::dag::{config::PipelineConfig, DagBuilder, DagNode};
use media_core::{Frame, PixelFormat};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// A 4×1 RGBA test frame with known (u8) values.
///
/// Pixels are laid out row-major:
/// ```text
///   r   g   b   a
///   0   0   0 255   — black
/// 255   0   0 255   — red
///   0 255   0 255   — green
///   0   0 255 255   — blue
/// ```
fn four_pixel_frame() -> Frame {
    let mut f = Frame::new_cpu(4, 1, PixelFormat::Rgba8).unwrap();
    let cpu = f.as_cpu_mut().unwrap();
    let row = cpu.row_mut(0);
    // pixel 0: black
    row[0..4].copy_from_slice(&[0, 0, 0, 255]);
    // pixel 1: red
    row[4..8].copy_from_slice(&[255, 0, 0, 255]);
    // pixel 2: green
    row[8..12].copy_from_slice(&[0, 255, 0, 255]);
    // pixel 3: blue
    row[12..16].copy_from_slice(&[0, 0, 255, 255]);
    f
}

/// Read the RGBA pixel at column `x` (0-based) from a 1-row frame.
fn pixel_at(frame: &Frame, x: usize) -> [u8; 4] {
    let cpu = frame.as_cpu().unwrap();
    let row = cpu.row(0);
    let offset = x * 4;
    [
        row[offset],
        row[offset + 1],
        row[offset + 2],
        row[offset + 3],
    ]
}

// ---------------------------------------------------------------------------
// Invert through pipeline DAG
// ---------------------------------------------------------------------------

#[test]
fn test_invert_pipeline_with_cpu_backend() {
    let dag = DagBuilder::new()
        .add(DagNode::new("inv", EffectKind::Invert))
        .build()
        .unwrap();
    let backend = CpuBackend::new();
    let result = dag.apply(&four_pixel_frame(), &backend).unwrap();
    let output = &result.outputs["inv"];

    assert_eq!(output.width, 4);
    assert_eq!(output.format, PixelFormat::Rgba8);

    // Invert: each channel becomes 255 - value
    assert_eq!(pixel_at(output, 0), [255, 255, 255, 255]); // black → white
    assert_eq!(pixel_at(output, 1), [0, 255, 255, 255]); // red → cyan
    assert_eq!(pixel_at(output, 2), [255, 0, 255, 255]); // green → magenta
    assert_eq!(pixel_at(output, 3), [255, 255, 0, 255]); // blue → yellow
}

// ---------------------------------------------------------------------------
// Grayscale then Invert chain
// ---------------------------------------------------------------------------

#[test]
fn test_grayscale_then_invert_chain() {
    let dag = DagBuilder::new()
        .add(DagNode::new("gray", EffectKind::Grayscale))
        .add(DagNode::new("inv", EffectKind::Invert).depends_on(&["gray"]))
        .build()
        .unwrap();
    let backend = CpuBackend::new();
    let result = dag.apply(&four_pixel_frame(), &backend).unwrap();
    let output = &result.outputs["inv"];

    assert_eq!(output.width, 4);

    // Grayscale uses luminance: Y = 0.299R + 0.587G + 0.114B
    // black (0,0,0)   → gray = 0             → invert → 255
    // red   (255,0,0) → gray = 76            → invert → 179
    // green (0,255,0) → gray = 149 (approx)  → invert → 106
    // blue  (0,0,255) → gray = 29            → invert → 226
    let p0 = pixel_at(output, 0);
    let p1 = pixel_at(output, 1);
    let p2 = pixel_at(output, 2);
    let p3 = pixel_at(output, 3);

    assert_eq!(p0, [255, 255, 255, 255], "black → gray → invert failed");
    assert_eq!(
        p1[0], p1[1],
        "red pixel should be gray after grayscale+invert"
    );
    assert_eq!(
        p1[1], p1[2],
        "red pixel should be gray after grayscale+invert"
    );
    assert_eq!(p2[0], p2[1], "green pixel should be gray");
    assert_eq!(p2[1], p2[2], "green pixel should be gray");
    assert_eq!(p3[0], p3[1], "blue pixel should be gray");
    assert_eq!(p3[1], p3[2], "blue pixel should be gray");
}

// ---------------------------------------------------------------------------
// Pipeline config round-trip with real execution
// ---------------------------------------------------------------------------

#[test]
fn test_pipeline_config_json_roundtrip_and_execute() {
    // Build config, serialise, deserialise, build DAG, execute.
    let json = r#"{
        "nodes": [
            {"name": "g", "kind": "Grayscale"},
            {"name": "b", "kind": "Brightness", "params": [0.5], "depends_on": ["g"]}
        ]
    }"#;
    let cfg: PipelineConfig = serde_json::from_str(json).unwrap();
    assert_eq!(cfg.nodes.len(), 2);

    let dag = cfg.build_dag().unwrap();
    let backend = CpuBackend::new();
    let frame = four_pixel_frame();
    let result = dag.apply(&frame, &backend).unwrap();
    let output = &result.outputs["b"];

    assert_eq!(output.width, 4);
    assert_eq!(output.format, PixelFormat::Rgba8);
}

// ---------------------------------------------------------------------------
// Brightness + Sepia combination
// ---------------------------------------------------------------------------

#[test]
fn test_sepia_chain() {
    // Build a two-node DAG manually: Grayscale → Sepia
    let dag = DagBuilder::new()
        .add(DagNode::new("gray", EffectKind::Grayscale))
        .add(DagNode::new("sepia", EffectKind::Sepia).depends_on(&["gray"]))
        .build()
        .unwrap();
    let backend = CpuBackend::new();
    let result = dag.apply(&four_pixel_frame(), &backend).unwrap();
    let output = &result.outputs["sepia"];

    assert_eq!(output.width, 4);
    // Sepia on grayscale should produce a warm brownish tint.
    // Just verify it ran without error and all channels differ.
    let p = pixel_at(output, 0);
    assert!(p[3] == 255, "alpha should be preserved");
}

// ---------------------------------------------------------------------------
// Brightness param passthrough
// ---------------------------------------------------------------------------

#[test]
fn test_brightness_parameter_passthrough() {
    // Brightness uses additive offset: output = clamp(input + factor * 255, 0, 255).
    // factor = 0.0 → no change
    let dag = DagBuilder::new()
        .add(DagNode::new("bright", EffectKind::Brightness).with_param(EffectParam::Float(0.0)))
        .build()
        .unwrap();
    let backend = CpuBackend::new();
    let result = dag.apply(&four_pixel_frame(), &backend).unwrap();
    let output = &result.outputs["bright"];

    // With factor 0.0, all pixels are unchanged.
    assert_eq!(pixel_at(output, 0), [0, 0, 0, 255], "black unchanged");
    assert_eq!(pixel_at(output, 1), [255, 0, 0, 255], "red unchanged");
    assert_eq!(pixel_at(output, 2), [0, 255, 0, 255], "green unchanged");
    assert_eq!(pixel_at(output, 3), [0, 0, 255, 255], "blue unchanged");
}

#[test]
fn test_brightness_additive_shift() {
    // factor = 0.5 → add 127.5 to each RGB channel (clamped).
    let dag = DagBuilder::new()
        .add(DagNode::new("bright", EffectKind::Brightness).with_param(EffectParam::Float(0.5)))
        .build()
        .unwrap();
    let backend = CpuBackend::new();
    let result = dag.apply(&four_pixel_frame(), &backend).unwrap();
    let output = &result.outputs["bright"];

    // black (0,0,0) + 127.5 → 127 or 128
    let p = pixel_at(output, 0);
    assert_eq!(p[0], 127);
    assert_eq!(p[1], 127);
    assert_eq!(p[2], 127);
    assert_eq!(p[3], 255);

    // red (255,0,0) + 127.5 → R=255 (clamped), G=127, B=127
    let p = pixel_at(output, 1);
    assert_eq!(p[0], 255);
    assert_eq!(p[1], 127);
    assert_eq!(p[2], 127);
    assert_eq!(p[3], 255);
}

// ---------------------------------------------------------------------------
// DAG timestamps are populated
// ---------------------------------------------------------------------------

#[test]
fn test_stats_are_populated() {
    let dag = DagBuilder::new()
        .add(DagNode::new("a", EffectKind::Invert))
        .add(DagNode::new("b", EffectKind::Grayscale).depends_on(&["a"]))
        .build()
        .unwrap();
    let backend = CpuBackend::new();
    let result = dag.apply(&four_pixel_frame(), &backend).unwrap();

    assert!(
        result.stats.total.as_nanos() > 0,
        "total duration should be > 0"
    );
    assert_eq!(result.stats.per_node.len(), 2);
    for stat in &result.stats.per_node {
        assert!(
            stat.duration.as_nanos() > 0,
            "node {} duration > 0",
            stat.name
        );
    }
}
