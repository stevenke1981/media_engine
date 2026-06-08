//! CLI subcommand implementations.

use std::path::Path;

use anyhow::{bail, Context, Result};
use media_compute::{EffectDesc, EffectKind, ImageCodec};
use media_compute_cpu::CpuBackend;
use media_compute_pipeline::dag::config::PipelineConfig;
use media_compute_pipeline::EffectProvider;
use media_core::PixelFormat;
use media_image::{probe_from_path, EncoderFormat};

use crate::BackendKind;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse an effect name string to an `EffectKind`.
fn parse_effect_kind(name: &str) -> Result<EffectKind> {
    Ok(match name.to_lowercase().as_str() {
        "brightness" => EffectKind::Brightness,
        "contrast" => EffectKind::Contrast,
        "grayscale" => EffectKind::Grayscale,
        "invert" => EffectKind::Invert,
        "blur" => EffectKind::Blur,
        "sharpen" => EffectKind::Sharpen,
        "sepia" => EffectKind::Sepia,
        "edge_detect" | "edgedetect" => EffectKind::EdgeDetect,
        "threshold" => EffectKind::Threshold,
        "box_blur" | "boxblur" => EffectKind::BoxBlur,
        "emboss" => EffectKind::Emboss,
        "pixelate" => EffectKind::Pixelate,
        _ => bail!(
            "Unknown effect '{name}' — expected one of: \
            brightness, contrast, grayscale, invert, blur, sharpen, \
            sepia, edge_detect, threshold, box_blur, emboss, pixelate"
        ),
    })
}

/// Parse `key=value` CLI parameters into `EffectParam` values.
///
/// The key is treated as documentation only; the value is parsed as:
/// - `EffectParam::Float` if it contains `.`
/// - `EffectParam::Int` if it parses as i32
/// - `EffectParam::Float` otherwise (fallback)
fn parse_params(raw: &[String]) -> Vec<media_compute::EffectParam> {
    fn parse_one(s: &str) -> media_compute::EffectParam {
        if s.contains('.') {
            s.parse::<f32>()
                .map(media_compute::EffectParam::Float)
                .unwrap_or(media_compute::EffectParam::Float(0.0))
        } else if let Ok(v) = s.parse::<i32>() {
            media_compute::EffectParam::Int(v)
        } else if let Ok(v) = s.parse::<u32>() {
            media_compute::EffectParam::U32(v)
        } else {
            media_compute::EffectParam::Float(0.0)
        }
    }

    raw.iter()
        .map(|s| {
            if let Some((_key, value)) = s.split_once('=') {
                parse_one(value)
            } else {
                parse_one(s)
            }
        })
        .collect()
}

/// Create a `Box<dyn EffectProvider>` from the CLI backend kind.
fn create_backend(kind: &BackendKind) -> Result<Box<dyn EffectProvider>> {
    match kind {
        BackendKind::Cpu => Ok(Box::new(CpuBackend::new())),
        #[cfg(feature = "wgpu-backend")]
        BackendKind::Wgpu => Ok(Box::new(
            media_compute_wgpu::WgpuBackend::new(None)
                .map_err(|e| anyhow::anyhow!("Failed to create WGPU backend: {e}"))?,
        )),
        #[cfg(feature = "cuda-backend")]
        BackendKind::Cuda => Ok(Box::new(
            media_compute_cuda::CudaBackend::new()
                .map_err(|e| anyhow::anyhow!("Failed to create CUDA backend: {e}"))?,
        )),
    }
}

// ---------------------------------------------------------------------------
// Subcommand implementations
// ---------------------------------------------------------------------------

/// Run the "image-info" subcommand.
pub fn cmd_image_info(path: &str) -> Result<()> {
    let info = probe_from_path(path).with_context(|| format!("Failed to probe image: {path}"))?;

    println!("Image:        {}", path);
    println!("Format:       {}", info.format_name);
    println!("Dimensions:   {} x {}", info.width, info.height);

    Ok(())
}

/// Run the "apply" subcommand.
///
/// Decodes the input image, applies the named effect on the chosen backend,
/// and writes the result.
pub fn cmd_apply(
    input: &str,
    output: &str,
    effect_name: &str,
    raw_params: &[String],
    backend_kind: &BackendKind,
    fmt: EncoderFormat,
) -> Result<()> {
    let kind = parse_effect_kind(effect_name)?;
    let params = parse_params(raw_params);
    let desc = EffectDesc {
        kind: kind.clone(),
        params,
    };

    // Use CpuBackend for all I/O (encode/decode). All backends accept a
    // CPU-backed frame, so we use CPU for decode and then route the actual
    // effect to whichever backend was selected.
    let io = CpuBackend::new();
    let input_bytes =
        std::fs::read(input).with_context(|| format!("Failed to read input: {input}"))?;
    let frame = io
        .decode_from_bytes(&input_bytes, None)
        .with_context(|| format!("Failed to decode input: {input}"))?;

    let frame = if frame.format != PixelFormat::Rgba8 {
        media_image::convert::convert_cpu_format(&frame, PixelFormat::Rgba8)?
    } else {
        frame
    };

    // Create the selected backend and apply the effect.
    let backend = create_backend(backend_kind)?;
    let effect = backend.get_effect(&desc.kind).ok_or_else(|| {
        anyhow::anyhow!(
            "Effect '{effect_name}' not available on {:?} backend",
            backend_kind
        )
    })?;

    let result = effect.apply(&frame, &desc)?;

    // Encode and write.
    let bytes = io
        .encode_to_bytes(&result, fmt)
        .with_context(|| format!("Failed to encode output: {output}"))?;
    std::fs::write(output, &bytes).with_context(|| format!("Failed to write output: {output}"))?;

    eprintln!(
        "Applied {effect_name} ({:?}) — written to {output}",
        backend_kind,
    );
    Ok(())
}

/// Pick an output frame from a DAG result by output node name or fallback.
fn pick_output(
    config: &PipelineConfig,
    outputs: &std::collections::HashMap<String, media_core::Frame>,
) -> Result<media_core::Frame> {
    let name = config
        .output_name()
        .context("Pipeline has no nodes — nothing to output")?;
    let frame = outputs
        .get(name)
        .with_context(|| format!("Output node '{name}' not found in DAG results"))?;
    Ok(frame.clone())
}

/// Run the "pipeline-run" subcommand.
pub fn cmd_pipeline_run(
    config_path: &str,
    input_override: Option<String>,
    output_override: Option<String>,
    backend_kind: &BackendKind,
    fmt: Option<EncoderFormat>,
) -> Result<()> {
    // --- Load config --------------------------------------------------------
    let config_text = std::fs::read_to_string(config_path)
        .with_context(|| format!("Failed to read config: {config_path}"))?;

    let ext = Path::new(config_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    let config = match ext.to_lowercase().as_str() {
        "json" => PipelineConfig::from_json(&config_text)
            .with_context(|| format!("Failed to parse JSON config: {config_path}"))?,
        "toml" => PipelineConfig::from_toml(&config_text)
            .with_context(|| format!("Failed to parse TOML config: {config_path}"))?,
        other => bail!("Unsupported config extension '.{other}' — use .json or .toml"),
    };
    eprintln!("Loaded pipeline with {} node(s)", config.nodes.len());

    // --- Build DAG -----------------------------------------------------------
    let dag = config
        .build_dag()
        .context("Failed to build/validate pipeline DAG")?;

    // --- Determine input/output paths ----------------------------------------
    let input_path = input_override
        .or_else(|| config.input.clone())
        .context("No input path specified (provide via config or --input)")?;

    let output_path = output_override
        .or_else(|| config.output.clone())
        .context("No output path specified (provide via config or --output)")?;

    // --- Decode input (always CPU for I/O) -----------------------------------
    let io = CpuBackend::new();
    let input_bytes = std::fs::read(&input_path)
        .with_context(|| format!("Failed to read input: {input_path}"))?;
    let frame = io
        .decode_from_bytes(&input_bytes, None)
        .with_context(|| format!("Failed to decode input: {input_path}"))?;

    // Ensure RGBA8.
    let frame = if frame.format != PixelFormat::Rgba8 {
        media_image::convert::convert_cpu_format(&frame, PixelFormat::Rgba8)?
    } else {
        frame
    };

    // --- Create the requested backend and execute DAG ------------------------
    let backend = create_backend(backend_kind)?;
    let result = dag
        .apply(&frame, &*backend)
        .context("Pipeline DAG execution failed")?;

    // --- Report stats --------------------------------------------------------
    let stats = &result.stats;
    eprintln!(
        "Pipeline completed ({:?}) in {:?}",
        backend_kind, stats.total,
    );
    for stat in &stats.per_node {
        eprintln!("  {:<20} {:?}", stat.name, stat.duration);
    }

    // --- Pick and encode output ----------------------------------------------
    let output_frame = pick_output(&config, &result.outputs)?;

    let out_fmt = fmt.unwrap_or(EncoderFormat::Png);
    let bytes = io
        .encode_to_bytes(&output_frame, out_fmt)
        .with_context(|| format!("Failed to encode output: {output_path}"))?;
    std::fs::write(&output_path, &bytes)
        .with_context(|| format!("Failed to write output: {output_path}"))?;

    eprintln!("Pipeline output written to {output_path}");
    Ok(())
}
