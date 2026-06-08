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
    /// Apply brightness using the CPU backend.
    CpuBrightness {
        /// Input image path.
        input: String,
        /// Output image path.
        output: String,
        /// Brightness factor (-1.0 to 1.0).
        #[arg(long, default_value = "0.1")]
        factor: f32,
        /// Output format.
        #[arg(long, value_enum, default_value = "png")]
        format: OutputFormat,
    },
    /// Apply contrast using the CPU backend.
    CpuContrast {
        /// Input image path.
        input: String,
        /// Output image path.
        output: String,
        /// Contrast factor.
        #[arg(long, default_value = "1.5")]
        factor: f32,
        /// Output format.
        #[arg(long, value_enum, default_value = "png")]
        format: OutputFormat,
    },
    /// Convert to grayscale using the CPU backend.
    CpuGrayscale {
        /// Input image path.
        input: String,
        /// Output image path.
        output: String,
        /// Output format.
        #[arg(long, value_enum, default_value = "png")]
        format: OutputFormat,
    },
    /// Invert colors using the CPU backend.
    CpuInvert {
        /// Input image path.
        input: String,
        /// Output image path.
        output: String,
        /// Output format.
        #[arg(long, value_enum, default_value = "png")]
        format: OutputFormat,
    },
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
        Commands::CpuBrightness {
            input,
            output,
            factor,
            format,
        } => commands::cmd_cpu_brightness(&input, &output, factor, format.into())?,
        Commands::CpuContrast {
            input,
            output,
            factor,
            format,
        } => commands::cmd_cpu_contrast(&input, &output, factor, format.into())?,
        Commands::CpuGrayscale {
            input,
            output,
            format,
        } => commands::cmd_cpu_grayscale(&input, &output, format.into())?,
        Commands::CpuInvert {
            input,
            output,
            format,
        } => commands::cmd_cpu_invert(&input, &output, format.into())?,
    }

    Ok(())
}
