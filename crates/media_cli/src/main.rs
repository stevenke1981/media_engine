//! CLI entry point for the Rust Media Engine.

use clap::{Parser, Subcommand, ValueEnum};
use media_image::EncoderFormat;

mod commands;

#[derive(Parser)]
#[command(name = "media", about = "Rust Media Engine CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show image metadata (format, dimensions).
    ImageInfo {
        /// Path to the input image.
        input: String,
    },
    /// Apply an image effect with the given backend.
    Apply {
        /// Input image path.
        input: String,
        /// Output image path.
        output: String,
        /// Effect name (brightness, contrast, grayscale, invert, blur,
        /// sharpen, sepia, edge_detect, threshold, box_blur, emboss,
        /// pixelate).
        #[arg(long)]
        effect: String,
        /// Effect parameter(s) in key=value form (e.g. factor=0.5).
        /// May be repeated for effects with multiple parameters.
        #[arg(long)]
        param: Vec<String>,
        /// Compute backend (cpu, wgpu, cuda).
        #[arg(long, default_value = "cpu")]
        backend: BackendKind,
        /// Output format.
        #[arg(long, value_enum, default_value = "png")]
        format: OutputFormat,
    },
    /// Run a pipeline defined in a JSON or TOML config file.
    PipelineRun {
        /// Path to the pipeline config file (.json or .toml).
        config: String,
        /// Override input path from the config.
        #[arg(long)]
        input: Option<String>,
        /// Override output path from the config.
        #[arg(long)]
        output: Option<String>,
        /// Compute backend for individual effects (cpu, wgpu, cuda).
        #[arg(long, default_value = "cpu")]
        backend: BackendKind,
        /// Output format.
        #[arg(long, value_enum)]
        format: Option<OutputFormat>,
    },
}

#[derive(ValueEnum, Clone, Debug)]
enum BackendKind {
    Cpu,
    #[cfg(feature = "wgpu-backend")]
    Wgpu,
    #[cfg(feature = "cuda-backend")]
    Cuda,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum OutputFormat {
    Png,
    Jpeg,
    WebP,
}

impl From<OutputFormat> for EncoderFormat {
    fn from(f: OutputFormat) -> Self {
        match f {
            OutputFormat::Png => EncoderFormat::Png,
            OutputFormat::Jpeg => EncoderFormat::Jpeg { quality: 85 },
            OutputFormat::WebP => EncoderFormat::WebP { quality: 80.0 },
        }
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::ImageInfo { input } => commands::cmd_image_info(&input)?,
        Commands::Apply {
            input,
            output,
            effect,
            param,
            backend,
            format,
        } => commands::cmd_apply(&input, &output, &effect, &param, &backend, format.into())?,
        Commands::PipelineRun {
            config,
            input,
            output,
            backend,
            format,
        } => commands::cmd_pipeline_run(&config, input, output, &backend, format.map(Into::into))?,
    }

    Ok(())
}
