//! Integration tests for WGPU compute effects.
//!
//! These tests require a WGPU-capable GPU to run. If no adapter is found,
//! the tests skip gracefully (no panic).
//!
//! The WGSL shaders operate on BGRA8 pixel data packed in `u32`:
//! - byte 0 (bits 0-7):   Blue
//! - byte 1 (bits 8-15):  Green
//! - byte 2 (bits 16-23): Red
//! - byte 3 (bits 24-31): Alpha

use media_compute::{ComputeBackend, EffectDesc, EffectKind, EffectParam};
use media_compute_wgpu::WgpuBackend;
use media_core::pixel_format::PixelFormat;
use media_core::Frame;

// ── Helpers ────────────────────────────────────────────────────────────────

/// Try to create a WGPU backend.  Returns `None` if no GPU adapter is
/// available (test will be skipped).
fn create_backend() -> Option<WgpuBackend> {
    match WgpuBackend::new(None) {
        Ok(backend) => Some(backend),
        Err(e) => {
            eprintln!("⚠️  WGPU backend unavailable, skipping test: {e}");
            None
        }
    }
}

/// Extract BGRA channels for pixel (x, y) from a flat BGRA8 byte slice.
fn pixel_at(data: &[u8], x: u32, y: u32, width: u32) -> (u8, u8, u8, u8) {
    let idx = (y * width + x) as usize * 4;
    (data[idx], data[idx + 1], data[idx + 2], data[idx + 3]) // B, G, R, A
}

/// Create a 4×4 BGRA8 test frame with a diverse, known pixel pattern.
fn create_test_frame4x4() -> Frame {
    // Format: B, G, R, A per pixel.
    // clang-format off
    let pixels: Vec<u8> = vec![
        // row 0
        100, 150, 200, 255, // (0,0) — mid brightness
        50, 100, 150, 255, // (1,0)
        200, 100, 50, 255, // (2,0)
        0, 128, 255, 255, // (3,0)
        // row 1
        255, 255, 255, 255, // (0,1) — white
        128, 128, 128, 255, // (1,1) — mid-gray
        0, 0, 0, 255, // (2,1) — black
        100, 200, 50, 255, // (3,1)
        // row 2
        100, 100, 100, 255, // (0,2) — gray
        200, 200, 200, 255, // (1,2) — light gray
        50, 50, 50, 255, // (2,2) — dark gray
        150, 150, 150, 255, // (3,2) — gray
        // row 3
        30, 60, 90, 255, // (0,3)
        120, 180, 240, 255, // (1,3)
        70, 140, 210, 255, // (2,3)
        180, 90, 0, 255, // (3,3)
    ];
    // clang-format on

    let mut frame = Frame::new_cpu(4, 4, PixelFormat::Bgra8).unwrap();
    frame.as_cpu_mut().unwrap().data.copy_from_slice(&pixels);
    frame
}

// ── Brightness ─────────────────────────────────────────────────────────────

#[test]
fn test_brightness_identity() {
    let backend = match create_backend() {
        Some(b) => b,
        None => return,
    };
    let effect = backend.get_effect(&EffectKind::Brightness).unwrap();
    let frame = create_test_frame4x4();
    let desc = EffectDesc::new(EffectKind::Brightness).with(EffectParam::Float(1.0));
    let result = effect.apply(&frame, &desc).unwrap();

    let original = frame.as_cpu().unwrap();
    let output = result.as_cpu().unwrap();
    assert_eq!(
        output.data, original.data,
        "Brightness 1.0 should not change pixels"
    );
}

#[test]
fn test_brightness_double() {
    let backend = match create_backend() {
        Some(b) => b,
        None => return,
    };
    let effect = backend.get_effect(&EffectKind::Brightness).unwrap();
    let frame = create_test_frame4x4();
    let desc = EffectDesc::new(EffectKind::Brightness).with(EffectParam::Float(2.0));
    let result = effect.apply(&frame, &desc).unwrap();

    let output = result.as_cpu().unwrap();

    // Pixel (0,0): B=100*2=200, G=150*2=255(clamped), R=200*2=255(clamped)
    let (b, g, r, a) = pixel_at(&output.data, 0, 0, 4);
    assert_eq!(b, 200, "B at (0,0)");
    assert_eq!(g, 255, "G at (0,0) — should clamp");
    assert_eq!(r, 255, "R at (0,0) — should clamp");
    assert_eq!(a, 255, "A at (0,0)");

    // Pixel (2,1): B=0*2=0, G=0*2=0, R=0*2=0, A=255
    let (b, g, r, a) = pixel_at(&output.data, 2, 1, 4);
    assert_eq!(b, 0, "B at (2,1) — black");
    assert_eq!(g, 0, "G at (2,1) — black");
    assert_eq!(r, 0, "R at (2,1) — black");
    assert_eq!(a, 255, "A at (2,1)");
}

#[test]
fn test_brightness_half() {
    let backend = match create_backend() {
        Some(b) => b,
        None => return,
    };
    let effect = backend.get_effect(&EffectKind::Brightness).unwrap();
    let frame = create_test_frame4x4();
    let desc = EffectDesc::new(EffectKind::Brightness).with(EffectParam::Float(0.5));
    let result = effect.apply(&frame, &desc).unwrap();

    let output = result.as_cpu().unwrap();

    // Pixel (0,0): B=100*0.5=50, G=150*0.5=75, R=200*0.5=100
    let (b, g, r, a) = pixel_at(&output.data, 0, 0, 4);
    assert_eq!(b, 50, "B at (0,0)");
    assert_eq!(g, 75, "G at (0,0)");
    assert_eq!(r, 100, "R at (0,0)");
    assert_eq!(a, 255, "A at (0,0)");
}

// ── Contrast ───────────────────────────────────────────────────────────────

#[test]
fn test_contrast_identity() {
    let backend = match create_backend() {
        Some(b) => b,
        None => return,
    };
    let effect = backend.get_effect(&EffectKind::Contrast).unwrap();
    let frame = create_test_frame4x4();
    let desc = EffectDesc::new(EffectKind::Contrast).with(EffectParam::Float(1.0));
    let result = effect.apply(&frame, &desc).unwrap();

    let original = frame.as_cpu().unwrap();
    let output = result.as_cpu().unwrap();

    // Float round-trip: ((v/255 - 0.5) * 1.0 + 0.5) * 255 can have ±1 error
    // (e.g. 30/255=0.117647... → *255 = 29.9999 → u32 truncates to 29).
    for i in 0..original.data.len() / 4 {
        let base = i * 4;
        for ch in 0..3 {
            let diff = (output.data[base + ch] as i16 - original.data[base + ch] as i16).abs();
            assert!(
                diff <= 1,
                "Contrast 1.0 channel {ch} at pixel {i}: expected ~{}, got {}",
                original.data[base + ch],
                output.data[base + ch]
            );
        }
        assert_eq!(
            output.data[base + 3],
            original.data[base + 3],
            "Alpha at pixel {i}"
        );
    }
}

#[test]
fn test_contrast_zero() {
    let backend = match create_backend() {
        Some(b) => b,
        None => return,
    };
    let effect = backend.get_effect(&EffectKind::Contrast).unwrap();
    let frame = create_test_frame4x4();
    let desc = EffectDesc::new(EffectKind::Contrast).with(EffectParam::Float(0.0));
    let result = effect.apply(&frame, &desc).unwrap();

    let output = result.as_cpu().unwrap();

    // Every pixel should be flat mid-gray (128).  Formula with param0=0:
    //   nb = ((c/255 - 0.5) * 0 + 0.5) * 255 = 0.5 * 255 = 127.5 → 127
    for y in 0..4u32 {
        for x in 0..4u32 {
            let (b, g, r, a) = pixel_at(&output.data, x, y, 4);
            assert_eq!(b, 127, "B at ({x},{y})");
            assert_eq!(g, 127, "G at ({x},{y})");
            assert_eq!(r, 127, "R at ({x},{y})");
            assert_eq!(a, 255, "A at ({x},{y})");
        }
    }
}

#[test]
fn test_contrast_increase() {
    let backend = match create_backend() {
        Some(b) => b,
        None => return,
    };
    let effect = backend.get_effect(&EffectKind::Contrast).unwrap();
    let frame = create_test_frame4x4();
    let desc = EffectDesc::new(EffectKind::Contrast).with(EffectParam::Float(2.0));
    let result = effect.apply(&frame, &desc).unwrap();

    let original = frame.as_cpu().unwrap();
    let output = result.as_cpu().unwrap();

    // Contrast > 1 pushes values away from mid-gray: values > 128 → brighter,
    // values < 128 → darker.  At least some pixels should change.
    let mut any_change = false;
    for i in 0..output.data.len() {
        if output.data[i] != original.data[i] {
            any_change = true;
            break;
        }
    }
    assert!(any_change, "Contrast 2.0 should change at least one pixel");
}

// ── Grayscale ──────────────────────────────────────────────────────────────

#[test]
fn test_grayscale_all_equal() {
    let backend = match create_backend() {
        Some(b) => b,
        None => return,
    };
    let effect = backend.get_effect(&EffectKind::Grayscale).unwrap();
    let frame = create_test_frame4x4();
    let desc = EffectDesc::new(EffectKind::Grayscale);
    let result = effect.apply(&frame, &desc).unwrap();

    let output = result.as_cpu().unwrap();

    // Each pixel must have R = G = B (gray).
    for y in 0..4u32 {
        for x in 0..4u32 {
            let (b, g, r, a) = pixel_at(&output.data, x, y, 4);
            assert_eq!(r, g, "R ≠ G at ({x},{y})");
            assert_eq!(g, b, "G ≠ B at ({x},{y})");
            assert_eq!(a, 255, "Alpha changed at ({x},{y})");
        }
    }
}

#[test]
fn test_grayscale_known_value() {
    let backend = match create_backend() {
        Some(b) => b,
        None => return,
    };
    let effect = backend.get_effect(&EffectKind::Grayscale).unwrap();
    let frame = create_test_frame4x4();
    let desc = EffectDesc::new(EffectKind::Grayscale);
    let result = effect.apply(&frame, &desc).unwrap();

    let output = result.as_cpu().unwrap();

    // Pixel (0,0): B=100, G=150, R=200.
    // Gray = 0.299*R + 0.587*G + 0.114*B
    //      = 0.299*200 + 0.587*150 + 0.114*100
    //      = 59.8 + 88.05 + 11.4 = 159.25 → 159
    let (b, g, r, a) = pixel_at(&output.data, 0, 0, 4);
    assert_eq!(r, 159, "R gray at (0,0)");
    assert_eq!(g, 159, "G gray at (0,0)");
    assert_eq!(b, 159, "B gray at (0,0)");
    assert_eq!(a, 255, "A at (0,0)");

    // Pixel (0,1): white → gray(255)
    let (b, g, r, a) = pixel_at(&output.data, 0, 1, 4);
    assert_eq!(r, 255, "R gray at (0,1) — white");
    assert_eq!(g, 255, "G gray at (0,1) — white");
    assert_eq!(b, 255, "B gray at (0,1) — white");
    assert_eq!(a, 255);

    // Pixel (2,1): black → gray(0)
    let (b, g, r, a) = pixel_at(&output.data, 2, 1, 4);
    assert_eq!(r, 0, "R gray at (2,1) — black");
    assert_eq!(g, 0, "G gray at (2,1) — black");
    assert_eq!(b, 0, "B gray at (2,1) — black");
    assert_eq!(a, 255);
}

// ── Invert ─────────────────────────────────────────────────────────────────

#[test]
fn test_invert_known_value() {
    let backend = match create_backend() {
        Some(b) => b,
        None => return,
    };
    let effect = backend.get_effect(&EffectKind::Invert).unwrap();
    let frame = create_test_frame4x4();
    let desc = EffectDesc::new(EffectKind::Invert);
    let result = effect.apply(&frame, &desc).unwrap();

    let original = frame.as_cpu().unwrap();
    let output = result.as_cpu().unwrap();

    // Every color channel: out = 255 - in.  Alpha unchanged.
    for i in 0..original.data.len() / 4 {
        let base = i * 4;
        assert_eq!(
            output.data[base],
            255 - original.data[base],
            "B inv at pixel {i}"
        );
        assert_eq!(
            output.data[base + 1],
            255 - original.data[base + 1],
            "G inv at pixel {i}"
        );
        assert_eq!(
            output.data[base + 2],
            255 - original.data[base + 2],
            "R inv at pixel {i}"
        );
        assert_eq!(
            output.data[base + 3],
            original.data[base + 3],
            "A changed at pixel {i}"
        );
    }
}

#[test]
fn test_invert_twice_is_identity() {
    let backend = match create_backend() {
        Some(b) => b,
        None => return,
    };
    let effect = backend.get_effect(&EffectKind::Invert).unwrap();
    let frame = create_test_frame4x4();
    let desc = EffectDesc::new(EffectKind::Invert);

    let frame1 = effect.apply(&frame, &desc).unwrap();
    let frame2 = effect.apply(&frame1, &desc).unwrap();

    let original = frame.as_cpu().unwrap();
    let result = frame2.as_cpu().unwrap();
    assert_eq!(
        original.data, result.data,
        "Double invert should restore original"
    );
}

// ── Blur ───────────────────────────────────────────────────────────────────

#[test]
fn test_blur_changes_image() {
    let backend = match create_backend() {
        Some(b) => b,
        None => return,
    };
    let effect = backend.get_effect(&EffectKind::Blur).unwrap();
    let frame = create_test_frame4x4();
    let desc = EffectDesc::new(EffectKind::Blur);
    let result = effect.apply(&frame, &desc).unwrap();

    let original = frame.as_cpu().unwrap();
    let output = result.as_cpu().unwrap();

    // The 3×3 box blur should change at least some pixel values.
    let changed: usize = output
        .data
        .iter()
        .zip(original.data.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        changed > 0,
        "Blur should change at least one pixel byte (changed={changed})"
    );
}

#[test]
fn test_blur_uniform_unchanged() {
    let backend = match create_backend() {
        Some(b) => b,
        None => return,
    };
    let effect = backend.get_effect(&EffectKind::Blur).unwrap();

    // A uniform-color frame should remain identical after blur.
    let mut frame = Frame::new_cpu(4, 4, PixelFormat::Bgra8).unwrap();
    let cpu = frame.as_cpu_mut().unwrap();
    cpu.data.fill(128);

    let desc = EffectDesc::new(EffectKind::Blur);
    let result = effect.apply(&frame, &desc).unwrap();

    let output = result.as_cpu().unwrap();
    assert!(
        output.data.iter().all(|&v| v == 128),
        "Blur of uniform field should remain uniform"
    );
}

#[test]
fn test_blur_edge_pixels_smoothed() {
    let backend = match create_backend() {
        Some(b) => b,
        None => return,
    };
    let effect = backend.get_effect(&EffectKind::Blur).unwrap();

    // Sharp horizontal edge: top half black, bottom half white.
    let mut frame = Frame::new_cpu(4, 4, PixelFormat::Bgra8).unwrap();
    let cpu = frame.as_cpu_mut().unwrap();
    for y in 0..4u32 {
        for x in 0..4u32 {
            let idx = (y * 4 + x) as usize * 4;
            if y < 2 {
                cpu.data[idx..=idx + 2].fill(0); // black
            } else {
                cpu.data[idx..=idx + 2].fill(255); // white
            }
            cpu.data[idx + 3] = 255; // alpha
        }
    }

    let desc = EffectDesc::new(EffectKind::Blur);
    let result = effect.apply(&frame, &desc).unwrap();
    let output = result.as_cpu().unwrap();

    // Pixels at the edge (row 1 and 2) should be smoothed (≈127-128).
    let (b1, g1, r1, _) = pixel_at(&output.data, 0, 1, 4);
    assert!(
        b1 > 0 && b1 < 255,
        "Edge pixel (0,1) should be blurred (b={b1})"
    );
    assert_eq!(r1, g1, "All channels should blur equally");

    let (b2, _, _, _) = pixel_at(&output.data, 0, 2, 4);
    assert!(
        b2 > 0 && b2 < 255,
        "Edge pixel (0,2) should be blurred (b={b2})"
    );
}

// ── Sharpen ────────────────────────────────────────────────────────────────

#[test]
fn test_sharpen_changes_image() {
    let backend = match create_backend() {
        Some(b) => b,
        None => return,
    };
    let effect = backend.get_effect(&EffectKind::Sharpen).unwrap();
    let frame = create_test_frame4x4();
    let desc = EffectDesc::new(EffectKind::Sharpen).with(EffectParam::Float(1.0));
    let result = effect.apply(&frame, &desc).unwrap();

    let original = frame.as_cpu().unwrap();
    let output = result.as_cpu().unwrap();

    let changed: usize = output
        .data
        .iter()
        .zip(original.data.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        changed > 0,
        "Sharpen should change at least one pixel byte (changed={changed})"
    );
}

#[test]
fn test_sharpen_zero_strength_identity() {
    let backend = match create_backend() {
        Some(b) => b,
        None => return,
    };
    let effect = backend.get_effect(&EffectKind::Sharpen).unwrap();
    let frame = create_test_frame4x4();
    // Strength = 0: sharpened = center + (center - blur) * 0 = center
    let desc = EffectDesc::new(EffectKind::Sharpen).with(EffectParam::Float(0.0));
    let result = effect.apply(&frame, &desc).unwrap();

    let original = frame.as_cpu().unwrap();
    let output = result.as_cpu().unwrap();
    assert_eq!(
        output.data, original.data,
        "Sharpen strength=0 should be identity"
    );
}

#[test]
fn test_sharpen_uniform_unchanged() {
    let backend = match create_backend() {
        Some(b) => b,
        None => return,
    };
    let effect = backend.get_effect(&EffectKind::Sharpen).unwrap();

    // Uniform frame: center - blur = 0, so sharpen should be identity.
    let mut frame = Frame::new_cpu(4, 4, PixelFormat::Bgra8).unwrap();
    frame.as_cpu_mut().unwrap().data.fill(128);

    let desc = EffectDesc::new(EffectKind::Sharpen).with(EffectParam::Float(1.0));
    let result = effect.apply(&frame, &desc).unwrap();

    let output = result.as_cpu().unwrap();
    assert!(
        output.data.iter().all(|&v| v == 128),
        "Sharpen of uniform field should remain unchanged"
    );
}

// ── Effect availability ────────────────────────────────────────────────────

#[test]
fn test_all_effects_available() {
    let backend = match create_backend() {
        Some(b) => b,
        None => return,
    };

    assert!(backend.has_effect(&EffectKind::Brightness));
    assert!(backend.has_effect(&EffectKind::Contrast));
    assert!(backend.has_effect(&EffectKind::Grayscale));
    assert!(backend.has_effect(&EffectKind::Invert));
    assert!(backend.has_effect(&EffectKind::Blur));
    assert!(backend.has_effect(&EffectKind::Sharpen));
    assert!(!backend.has_effect(&EffectKind::Custom("nonexistent".into())));

    let available = backend.available_effects();
    assert_eq!(available.len(), 12);
}

#[test]
fn test_effect_names() {
    let backend = match create_backend() {
        Some(b) => b,
        None => return,
    };

    assert_eq!(
        backend.get_effect(&EffectKind::Brightness).unwrap().name(),
        "brightness"
    );
    assert_eq!(
        backend.get_effect(&EffectKind::Contrast).unwrap().name(),
        "contrast"
    );
    assert_eq!(
        backend.get_effect(&EffectKind::Grayscale).unwrap().name(),
        "grayscale"
    );
    assert_eq!(
        backend.get_effect(&EffectKind::Invert).unwrap().name(),
        "invert"
    );
    assert_eq!(
        backend.get_effect(&EffectKind::Blur).unwrap().name(),
        "blur"
    );
    assert_eq!(
        backend.get_effect(&EffectKind::Sharpen).unwrap().name(),
        "sharpen"
    );
}

// ── Pipeline / chaining ────────────────────────────────────────────────────

#[test]
fn test_invert_then_grayscale_chain() {
    let backend = match create_backend() {
        Some(b) => b,
        None => return,
    };
    let frame = create_test_frame4x4();

    let invert = backend.get_effect(&EffectKind::Invert).unwrap();
    let gray = backend.get_effect(&EffectKind::Grayscale).unwrap();

    let frame1 = invert
        .apply(&frame, &EffectDesc::new(EffectKind::Invert))
        .unwrap();
    let frame2 = gray
        .apply(&frame1, &EffectDesc::new(EffectKind::Grayscale))
        .unwrap();

    // Result should be fully grayscale.
    let output = frame2.as_cpu().unwrap();
    for y in 0..4u32 {
        for x in 0..4u32 {
            let (b, g, r, a) = pixel_at(&output.data, x, y, 4);
            assert_eq!(r, g, "R ≠ G at ({x},{y}) after chain");
            assert_eq!(g, b, "G ≠ B at ({x},{y}) after chain");
            assert_eq!(a, 255, "Alpha at ({x},{y})");
        }
    }
}

#[test]
fn test_brightness_then_invert_chain() {
    let backend = match create_backend() {
        Some(b) => b,
        None => return,
    };
    let frame = create_test_frame4x4();

    let bright = backend.get_effect(&EffectKind::Brightness).unwrap();
    let invert = backend.get_effect(&EffectKind::Invert).unwrap();

    // Half brightness, then invert.
    let half = bright
        .apply(
            &frame,
            &EffectDesc::new(EffectKind::Brightness).with(0.5f32),
        )
        .unwrap();
    let result = invert
        .apply(&half, &EffectDesc::new(EffectKind::Invert))
        .unwrap();

    let output = result.as_cpu().unwrap();

    // Pixel (0,0): original [100,150,200,255] → half → [50,75,100,255] → invert → [205,180,155,255]
    let (b, g, r, a) = pixel_at(&output.data, 0, 0, 4);
    assert_eq!(b, 205, "B at (0,0) after chain");
    assert_eq!(g, 180, "G at (0,0) after chain");
    assert_eq!(r, 155, "R at (0,0) after chain");
    assert_eq!(a, 255, "A at (0,0) after chain");
}

// ── In-place application ───────────────────────────────────────────────────

#[test]
fn test_supports_in_place() {
    let backend = match create_backend() {
        Some(b) => b,
        None => return,
    };
    for kind in [
        EffectKind::Brightness,
        EffectKind::Contrast,
        EffectKind::Grayscale,
        EffectKind::Invert,
        EffectKind::Blur,
        EffectKind::Sharpen,
    ] {
        let effect = backend.get_effect(&kind).unwrap();
        assert!(
            effect.supports_in_place(),
            "{} should support in-place",
            effect.name()
        );
    }
}

#[test]
fn test_invert_in_place() {
    let backend = match create_backend() {
        Some(b) => b,
        None => return,
    };
    let effect = backend.get_effect(&EffectKind::Invert).unwrap();
    let original = create_test_frame4x4();
    let original_data = original.as_cpu().unwrap().data.clone();

    let mut frame = create_test_frame4x4();
    effect
        .apply_in_place(&mut frame, &EffectDesc::new(EffectKind::Invert))
        .unwrap();

    let cpu = frame.as_cpu().unwrap();
    for i in 0..original_data.len() / 4 {
        let base = i * 4;
        assert_eq!(cpu.data[base], 255 - original_data[base], "B inv at {i}");
        assert_eq!(
            cpu.data[base + 1],
            255 - original_data[base + 1],
            "G inv at {i}"
        );
        assert_eq!(
            cpu.data[base + 2],
            255 - original_data[base + 2],
            "R inv at {i}"
        );
        assert_eq!(cpu.data[base + 3], original_data[base + 3], "A at {i}");
    }
}

#[test]
fn test_brightness_in_place_matches_apply() {
    let backend = match create_backend() {
        Some(b) => b,
        None => return,
    };
    let effect = backend.get_effect(&EffectKind::Brightness).unwrap();
    let frame = create_test_frame4x4();
    let desc = EffectDesc::new(EffectKind::Brightness).with(EffectParam::Float(1.5));

    // Out-of-place.
    let result = effect.apply(&frame, &desc).unwrap();

    // In-place.
    let mut frame2 = create_test_frame4x4();
    effect.apply_in_place(&mut frame2, &desc).unwrap();

    assert_eq!(
        result.as_cpu().unwrap().data,
        frame2.as_cpu().unwrap().data,
        "in-place and out-of-place should produce identical results"
    );
}

// ── Error handling ─────────────────────────────────────────────────────────

#[test]
fn test_non_cpu_frame_errors() {
    // WGPU effects require a CPU-backed frame.  Creating a non-CPU frame
    // directly is impossible with public API, but we test the error path
    // by checking that a zero-size frame is rejected.
    //
    // Note: wgpu 23 panics on validation errors (zero-size buffer binding),
    // so we catch_unwind here.
    let backend = match create_backend() {
        Some(b) => b,
        None => return,
    };
    let effect = backend.get_effect(&EffectKind::Invert).unwrap();

    let small = Frame::new_cpu(0, 0, PixelFormat::Bgra8).unwrap();
    let desc = EffectDesc::new(EffectKind::Invert);
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| effect.apply(&small, &desc)));

    assert!(
        result.is_err(),
        "Zero-size frame should produce a panic (wgpu validation error) or Err"
    );
}

#[test]
fn test_unknown_effect_not_found() {
    let backend = match create_backend() {
        Some(b) => b,
        None => return,
    };
    assert!(backend
        .get_effect(&EffectKind::Custom("missing".into()))
        .is_none());
}
