//! Build script for the CUDA compute backend.
//!
//! - Locates the CUDA toolkit via CUDA_PATH or a well-known default path.
//! - Adds the CUDA import library directory to the linker search path.
//! - Links against nvcuda.lib (CUDA Driver API).
//! - Compiles all .cu kernel sources to PTX and embeds them via include_bytes!.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let cuda_path = resolve_cuda_path();
    println!("cuda_path={}", cuda_path.display());

    // ── Linker configuration ──────────────────────────────────────────
    let lib_dir = cuda_path.join("lib").join("x64");
    if lib_dir.exists() {
        println!("cargo:rustc-link-search={}", lib_dir.display());
    } else {
        println!(
            "cargo:warning=CUDA lib directory not found: {}",
            lib_dir.display()
        );
    }
    println!("cargo:rustc-link-lib=nvcuda");

    // ── Compile CUDA kernels to PTX ───────────────────────────────────
    let kernel_dir = Path::new("kernels");
    if !kernel_dir.exists() {
        return;
    }

    let nvcc = cuda_path.join("bin").join("nvcc.exe");
    let out_dir = std::env::var("OUT_DIR").unwrap();

    // Set up MSVC environment for nvcc (PATH, INCLUDE, LIB).
    if let Some(msvc_bin) = find_msvc_cl() {
        let msvc_root = msvc_bin
            .parent()
            .unwrap() // bin/Hostx64/x64
            .parent()
            .unwrap() // bin/Hostx64
            .parent()
            .unwrap(); // bin -> MSVC root (<root>/bin/Hostx64/x64/cl.exe)
        let msvc_dir = msvc_bin.parent().unwrap();

        // Prepend MSVC compiler directory to PATH so nvcc finds cl.exe.
        let old_path = std::env::var("PATH").unwrap_or_default();
        let new_path = format!("{};{}", msvc_dir.display(), old_path);
        std::env::set_var("PATH", &new_path);
        println!("cargo:warning=MSVC cl.exe at: {}", msvc_bin.display());

        // Set INCLUDE for MSVC and Windows SDK headers.
        let sdk_root = Path::new(r"C:\Program Files (x86)\Windows Kits\10");
        let sdk_ver = "10.0.26100.0";

        let msvc_inc = msvc_root.join("include");
        let sdk_ucrt = sdk_root.join("Include").join(sdk_ver).join("ucrt");
        let sdk_um = sdk_root.join("Include").join(sdk_ver).join("um");
        let sdk_shared = sdk_root.join("Include").join(sdk_ver).join("shared");

        let mut include_dirs = Vec::new();
        if msvc_inc.exists() {
            include_dirs.push(msvc_inc);
        }
        if sdk_ucrt.exists() {
            include_dirs.push(sdk_ucrt);
        }
        if sdk_um.exists() {
            include_dirs.push(sdk_um);
        }
        if sdk_shared.exists() {
            include_dirs.push(sdk_shared);
        }

        let include_str = include_dirs
            .iter()
            .map(|d| d.to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join(";");
        if !include_str.is_empty() {
            std::env::set_var("INCLUDE", &include_str);
            println!("cargo:warning=INCLUDE set with {} dirs", include_dirs.len());
        }

        // Set LIB for MSVC and Windows SDK libraries.
        let msvc_lib = msvc_root.join("lib").join("x64");
        let sdk_ucrt_lib = sdk_root.join("Lib").join(sdk_ver).join("ucrt").join("x64");
        let sdk_um_lib = sdk_root.join("Lib").join(sdk_ver).join("um").join("x64");

        let mut lib_dirs = Vec::new();
        if msvc_lib.exists() {
            lib_dirs.push(msvc_lib);
        }
        if sdk_ucrt_lib.exists() {
            lib_dirs.push(sdk_ucrt_lib);
        }
        if sdk_um_lib.exists() {
            lib_dirs.push(sdk_um_lib);
        }

        let lib_str = lib_dirs
            .iter()
            .map(|d| d.to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join(";");
        if !lib_str.is_empty() {
            std::env::set_var("LIB", &lib_str);
            println!("cargo:warning=LIB set with {} dirs", lib_dirs.len());
        }
    } else {
        println!("cargo:warning=MSVC cl.exe not found — nvcc will fail if cl.exe is not on PATH");
    }

    let mut ptx_paths: Vec<(String, PathBuf)> = Vec::new();

    for entry in std::fs::read_dir(kernel_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "cu") {
            let stem = path.file_stem().unwrap().to_string_lossy().to_string();
            let ptx_path = Path::new(&out_dir).join(&stem).with_extension("ptx");

            if !nvcc.exists() {
                println!(
                    "cargo:warning=nvcc not found at {} — skipping PTX compilation for {}",
                    nvcc.display(),
                    path.display(),
                );
                continue;
            }

            let status = Command::new(&nvcc)
                .arg("--ptx")
                .arg("-allow-unsupported-compiler")
                .arg("-o")
                .arg(&ptx_path)
                .arg(&path)
                .status()
                .expect("failed to execute nvcc");

            assert!(
                status.success(),
                "nvcc compilation failed for: {}",
                path.display(),
            );

            println!("cargo:rerun-if-changed={}", path.display());
            ptx_paths.push((stem.to_uppercase(), ptx_path));
        }
    }

    // ── Generate PTX embed module ────────────────────────────────────
    generate_embed(&out_dir, &ptx_paths);
}

/// Search common Visual Studio installation paths for cl.exe.
fn find_msvc_cl() -> Option<PathBuf> {
    let candidates = [
        r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC",
        r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\17\Community\VC\Tools\MSVC",
        r"C:\Program Files (x86)\Microsoft Visual Studio\17\BuildTools\VC\Tools\MSVC",
    ];

    let mut best: Option<(u64, u64, u64, PathBuf)> = None;

    for base in candidates {
        let base_path = Path::new(base);
        if !base_path.exists() {
            continue;
        }
        let entries = match std::fs::read_dir(base_path) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let ver_path = entry.path();
            let ver_name = match ver_path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };
            // Parse version like "14.44.35207"
            let parts: Vec<&str> = ver_name.split('.').collect();
            if parts.len() < 3 {
                continue;
            }
            let major: u64 = parts[0].parse().unwrap_or(0);
            let minor: u64 = parts[1].parse().unwrap_or(0);
            let patch: u64 = parts[2].parse().unwrap_or(0);

            let cl_path = ver_path
                .join("bin")
                .join("Hostx64")
                .join("x64")
                .join("cl.exe");
            if cl_path.exists() {
                let is_better = match &best {
                    None => true,
                    Some((bmaj, bmin, bpat, _)) => {
                        major > *bmaj
                            || (major == *bmaj && minor > *bmin)
                            || (major == *bmaj && minor == *bmin && patch > *bpat)
                    }
                };
                if is_better {
                    best = Some((major, minor, patch, cl_path));
                }
            }
        }
    }

    best.map(|(_, _, _, path)| path)
}

/// Generate kernels_embedded.rs in OUT_DIR containing `include_bytes!`
/// for each compiled PTX file, plus exported constants.
fn generate_embed(out_dir: &str, ptx_files: &[(String, std::path::PathBuf)]) {
    let embed_path = Path::new(out_dir).join("kernels_embedded.rs");
    let mut content = String::new();

    content.push_str("// Auto-generated by build.rs — do not edit.\n");
    content.push_str("#[allow(dead_code)]\n");

    // Add a `include_bytes!` for each PTX file.
    // The generated file lives in OUT_DIR, so paths from include_bytes!
    // are relative to OUT_DIR — which is where the .ptx files live.
    for (name, ptx_path) in ptx_files {
        let const_name = format!("{}_PTX", name);
        // Use the filename relative to OUT_DIR
        let relative = ptx_path.file_name().unwrap().to_string_lossy();
        content.push_str(&format!(
            "pub const {}: &[u8] = include_bytes!(r#\"{}\"#);\n",
            const_name, relative,
        ));
    }

    std::fs::write(&embed_path, content).expect("Failed to write kernels_embedded.rs");
    println!(
        "cargo:warning=Generated kernels_embedded.rs with {} PTX entries",
        ptx_files.len()
    );
}

/// Locate the CUDA toolkit installation directory.
fn resolve_cuda_path() -> Box<Path> {
    // Honour the standard environment variable first.
    if let Ok(path) = std::env::var("CUDA_PATH") {
        if Path::new(&path).exists() {
            return Path::new(&path).into();
        }
    }
    // Fall back to the typical default install location (v12.x).
    let default = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6";
    if Path::new(default).exists() {
        return Path::new(default).into();
    }
    let default = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5";
    if Path::new(default).exists() {
        return Path::new(default).into();
    }
    let default = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4";
    if Path::new(default).exists() {
        return Path::new(default).into();
    }
    // Last-resort guess; linking will fail loud and clear if wrong.
    Path::new(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6").into()
}
