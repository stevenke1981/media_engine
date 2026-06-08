//! CLI subcommand implementations.

use anyhow::{Context, Result};
use media_compute::{ComputeEffect, EffectDesc, EffectKind, ImageCodec};
use media_compute_cpu::{
    BrightnessEffect, ContrastEffect, CpuBackend, GrayscaleEffect, InvertEffect,
};
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
