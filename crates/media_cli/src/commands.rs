//! CLI subcommand implementations.

use std::path::Path;

use anyhow::{bail, Context, Result};
use media_compute::{ComputeEffect, EffectDesc, EffectKind, ImageCodec};
use media_compute_cpu::{
    BlurEffect, BrightnessEffect, ContrastEffect, CpuBackend, GrayscaleEffect, InvertEffect,
    SharpenEffect,
};
use media_compute_pipeline::dag::config::PipelineConfig;
use media_core::PixelFormat;
use media_image::{probe_from_path, EncoderFormat};

/// Run the "image-info" subcommand.
pub fn cmd_image_info(path: &str) -> Result<()> {
    let info = probe_from_path(path).with_context(|| format!("Failed to probe image: {path}"))?;

    println!("Image:        {}", path);
    println!("Format:       {}", info.format_name);
    println!("Dimensions:   {} x {}", info.width, info.height);

    Ok(())
}

/// Run the "cpu-brightness" subcommand.
pub fn cmd_cpu_brightness(
    input: &str,
    output: &str,
    factor: f32,
    fmt: EncoderFormat,
) -> Result<()> {
    let backend = CpuBackend::new();
    let input_bytes =
        std::fs::read(input).with_context(|| format!("Failed to read input: {input}"))?;
    let frame = backend
        .decode_from_bytes(&input_bytes, None)
        .with_context(|| format!("Failed to decode input: {input}"))?;

    // Ensure it's RGBA8.
    let frame = if frame.format != PixelFormat::Rgba8 {
        media_image::convert::convert_cpu_format(&frame, PixelFormat::Rgba8)?
    } else {
        frame
    };

    let effect = BrightnessEffect;
    let desc = EffectDesc::new(EffectKind::Brightness).with(factor);
    let result = effect.apply(&frame, &desc)?;

    let bytes = backend
        .encode_to_bytes(&result, fmt)
        .with_context(|| format!("Failed to encode output: {output}"))?;
    std::fs::write(output, &bytes).with_context(|| format!("Failed to write output: {output}"))?;

    eprintln!("Applied brightness (factor={factor}) — written to {output}");
    Ok(())
}

/// Run the "cpu-contrast" subcommand.
pub fn cmd_cpu_contrast(input: &str, output: &str, factor: f32, fmt: EncoderFormat) -> Result<()> {
    let backend = CpuBackend::new();
    let input_bytes =
        std::fs::read(input).with_context(|| format!("Failed to read input: {input}"))?;
    let frame = backend
        .decode_from_bytes(&input_bytes, None)
        .with_context(|| format!("Failed to decode input: {input}"))?;

    let frame = if frame.format != PixelFormat::Rgba8 {
        media_image::convert::convert_cpu_format(&frame, PixelFormat::Rgba8)?
    } else {
        frame
    };

    let effect = ContrastEffect;
    let desc = EffectDesc::new(EffectKind::Contrast).with(factor);
    let result = effect.apply(&frame, &desc)?;

    let bytes = backend
        .encode_to_bytes(&result, fmt)
        .with_context(|| format!("Failed to encode output: {output}"))?;
    std::fs::write(output, &bytes).with_context(|| format!("Failed to write output: {output}"))?;

    eprintln!("Applied contrast (factor={factor}) — written to {output}");
    Ok(())
}

/// Run the "cpu-grayscale" subcommand.
pub fn cmd_cpu_grayscale(input: &str, output: &str, fmt: EncoderFormat) -> Result<()> {
    let backend = CpuBackend::new();
    let input_bytes =
        std::fs::read(input).with_context(|| format!("Failed to read input: {input}"))?;
    let frame = backend
        .decode_from_bytes(&input_bytes, None)
        .with_context(|| format!("Failed to decode input: {input}"))?;

    let frame = if frame.format != PixelFormat::Rgba8 {
        media_image::convert::convert_cpu_format(&frame, PixelFormat::Rgba8)?
    } else {
        frame
    };

    let effect = GrayscaleEffect;
    let desc = EffectDesc::new(EffectKind::Grayscale);
    let result = effect.apply(&frame, &desc)?;

    let bytes = backend
        .encode_to_bytes(&result, fmt)
        .with_context(|| format!("Failed to encode output: {output}"))?;
    std::fs::write(output, &bytes).with_context(|| format!("Failed to write output: {output}"))?;

    eprintln!("Applied grayscale — written to {output}");
    Ok(())
}

/// Run the "cpu-invert" subcommand.
pub fn cmd_cpu_invert(input: &str, output: &str, fmt: EncoderFormat) -> Result<()> {
    let backend = CpuBackend::new();
    let input_bytes =
        std::fs::read(input).with_context(|| format!("Failed to read input: {input}"))?;
    let frame = backend
        .decode_from_bytes(&input_bytes, None)
        .with_context(|| format!("Failed to decode input: {input}"))?;

    let frame = if frame.format != PixelFormat::Rgba8 {
        media_image::convert::convert_cpu_format(&frame, PixelFormat::Rgba8)?
    } else {
        frame
    };

    let effect = InvertEffect;
    let desc = EffectDesc::new(EffectKind::Invert);
    let result = effect.apply(&frame, &desc)?;

    let bytes = backend
        .encode_to_bytes(&result, fmt)
        .with_context(|| format!("Failed to encode output: {output}"))?;
    std::fs::write(output, &bytes).with_context(|| format!("Failed to write output: {output}"))?;

    eprintln!("Applied invert — written to {output}");
    Ok(())
}

/// Run the "cpu-blur" subcommand.
pub fn cmd_cpu_blur(input: &str, output: &str, fmt: EncoderFormat) -> Result<()> {
    let backend = CpuBackend::new();
    let input_bytes =
        std::fs::read(input).with_context(|| format!("Failed to read input: {input}"))?;
    let frame = backend
        .decode_from_bytes(&input_bytes, None)
        .with_context(|| format!("Failed to decode input: {input}"))?;

    let frame = if frame.format != PixelFormat::Rgba8 {
        media_image::convert::convert_cpu_format(&frame, PixelFormat::Rgba8)?
    } else {
        frame
    };

    let effect = BlurEffect;
    let desc = EffectDesc::new(EffectKind::Blur);
    let result = effect.apply(&frame, &desc)?;

    let bytes = backend
        .encode_to_bytes(&result, fmt)
        .with_context(|| format!("Failed to encode output: {output}"))?;
    std::fs::write(output, &bytes).with_context(|| format!("Failed to write output: {output}"))?;

    eprintln!("Applied blur — written to {output}");
    Ok(())
}

/// Run the "cpu-sharpen" subcommand.
pub fn cmd_cpu_sharpen(input: &str, output: &str, strength: f32, fmt: EncoderFormat) -> Result<()> {
    let backend = CpuBackend::new();
    let input_bytes =
        std::fs::read(input).with_context(|| format!("Failed to read input: {input}"))?;
    let frame = backend
        .decode_from_bytes(&input_bytes, None)
        .with_context(|| format!("Failed to decode input: {input}"))?;

    let frame = if frame.format != PixelFormat::Rgba8 {
        media_image::convert::convert_cpu_format(&frame, PixelFormat::Rgba8)?
    } else {
        frame
    };

    let effect = SharpenEffect;
    let desc = EffectDesc::new(EffectKind::Sharpen).with(strength);
    let result = effect.apply(&frame, &desc)?;

    let bytes = backend
        .encode_to_bytes(&result, fmt)
        .with_context(|| format!("Failed to encode output: {output}"))?;
    std::fs::write(output, &bytes).with_context(|| format!("Failed to write output: {output}"))?;

    eprintln!("Applied sharpen (strength={strength}) — written to {output}");
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

    // --- Decode input --------------------------------------------------------
    let backend = CpuBackend::new();
    let input_bytes = std::fs::read(&input_path)
        .with_context(|| format!("Failed to read input: {input_path}"))?;
    let frame = backend
        .decode_from_bytes(&input_bytes, None)
        .with_context(|| format!("Failed to decode input: {input_path}"))?;

    // Ensure RGBA8.
    let frame = if frame.format != PixelFormat::Rgba8 {
        media_image::convert::convert_cpu_format(&frame, PixelFormat::Rgba8)?
    } else {
        frame
    };

    // --- Execute DAG ---------------------------------------------------------
    let result = dag
        .apply(&frame, &backend)
        .context("Pipeline DAG execution failed")?;

    // --- Report stats --------------------------------------------------------
    let stats = &result.stats;
    eprintln!("Pipeline completed in {:?}", stats.total);
    for stat in &stats.per_node {
        eprintln!("  {:<20} {:?}", stat.name, stat.duration);
    }

    // --- Pick and encode output ----------------------------------------------
    let output_frame = pick_output(&config, &result.outputs)?;

    let out_fmt = fmt.unwrap_or(EncoderFormat::Png);
    let bytes = backend
        .encode_to_bytes(&output_frame, out_fmt)
        .with_context(|| format!("Failed to encode output: {output_path}"))?;
    std::fs::write(&output_path, &bytes)
        .with_context(|| format!("Failed to write output: {output_path}"))?;

    eprintln!("Pipeline output written to {output_path}");
    Ok(())
}
