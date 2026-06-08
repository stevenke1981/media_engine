//! SIMD-accelerated pixel format conversions with runtime CPU feature dispatch.
//!
//! Supported conversions:
//! - `bgra_to_rgba`  — BGRA  → RGBA  (alpha-preserving)
//! - `rgba_to_bgra`  — RGBA  → BGRA  (swizzle only, no channel arithmetic)
//! - `rgba_to_rgb`   — RGBA  → RGB   (drop alpha channel)
//! - `rgb_to_rgba`   — RGB   → RGBA  (fill alpha = 255)
//!
//! Each function uses runtime dispatch:
//!   1. AVX2  (256-bit vectors, 8–10 pixels per iteration)
//!   2. SSSE3 (128-bit vectors, 4 pixels per iteration)
//!   3. Scalar fallback (portable, safe)
//!
//! # Safety
//! All `*_simd` functions are `unsafe` because they require `target_feature` enable.
//! The public wrappers are safe and handle dispatch internally.

// ---------------------------------------------------------------------------
// BGRA ↔ RGBA  (swizzle channels 0↔2)
// ---------------------------------------------------------------------------

/// Convert BGRA bytes to RGBA bytes (safe dispatch).
#[inline]
pub fn bgra_to_rgba(pixels: &mut [u8]) {
    assert!(
        pixels.len() % 4 == 0,
        "bgra_to_rgba: buffer length must be a multiple of 4"
    );

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { bgra_to_rgba_avx2(pixels) };
        }
        if is_x86_feature_detected!("ssse3") {
            return unsafe { bgra_to_rgba_ssse3(pixels) };
        }
    }

    bgra_to_rgba_scalar(pixels);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bgra_to_rgba_avx2(pixels: &mut [u8]) {
    use std::arch::x86_64::*;

    let len = pixels.len();
    let mut i = 0;

    // Shuffle mask: for each 16-byte lane, map BGRABGRA...→RGBARGBA...
    // Byte:   0(B)→2, 1(G)→1, 2(R)→0, 3(A)→3, then repeat pixels 1,2,3.
    let shuf_mask = _mm256_setr_epi8(
        2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15, 2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11,
        14, 13, 12, 15,
    );

    // Process 32 bytes (8 pixels) per iteration.
    while i + 32 <= len {
        let v = _mm256_loadu_si256(pixels.as_ptr().add(i) as *const __m256i);
        let shuffled = _mm256_shuffle_epi8(v, shuf_mask);
        _mm256_storeu_si256(pixels.as_mut_ptr().add(i) as *mut __m256i, shuffled);
        i += 32;
    }

    // Tail pixels (0 to 31 remaining bytes — at most 7 full pixels + up to 3 stray bytes).
    if i < len {
        bgra_to_rgba_scalar(&mut pixels[i..]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn bgra_to_rgba_ssse3(pixels: &mut [u8]) {
    use std::arch::x86_64::*;

    let len = pixels.len();
    let mut i = 0;

    // Shuffle mask: byte 0(B)→2, 1(G)→1, 2(R)→0, 3(A)→3, repeat.
    let shuf_mask = _mm_setr_epi8(2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15);

    // Process 16 bytes (4 pixels) per iteration.
    while i + 16 <= len {
        let v = _mm_loadu_si128(pixels.as_ptr().add(i) as *const __m128i);
        let shuffled = _mm_shuffle_epi8(v, shuf_mask);
        _mm_storeu_si128(pixels.as_mut_ptr().add(i) as *mut __m128i, shuffled);
        i += 16;
    }

    if i < len {
        bgra_to_rgba_scalar(&mut pixels[i..]);
    }
}

fn bgra_to_rgba_scalar(pixels: &mut [u8]) {
    for chunk in pixels.chunks_exact_mut(4) {
        chunk.swap(0, 2);
    }
}

/// Convert RGBA bytes to BGRA bytes (safe dispatch).
#[inline]
pub fn rgba_to_bgra(pixels: &mut [u8]) {
    // The operation is identical to bgra_to_rgba — swap channels 0↔2.
    bgra_to_rgba(pixels);
}

// ---------------------------------------------------------------------------
// RGBA → RGB  (drop alpha)
// ---------------------------------------------------------------------------

/// Convert RGBA bytes to RGB bytes (packed, in-place safe dispatch).
///
/// `pixels` must have length divisible by 4.  The output is written back into the
/// same buffer and will be 3/4 of the original length.
///
/// # Panics
/// Panics if `pixels.len()` is not a multiple of 4.
pub fn rgba_to_rgb(pixels: &mut [u8]) {
    let n = pixels.len();
    assert!(
        n % 4 == 0,
        "rgba_to_rgb: buffer length must be a multiple of 4"
    );
    let pixels_out = n * 3 / 4; // each output pixel is 3 bytes

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { rgba_to_rgb_avx2(pixels, pixels_out) };
        }
        if is_x86_feature_detected!("ssse3") {
            return unsafe { rgba_to_rgb_ssse3(pixels, pixels_out) };
        }
    }

    rgba_to_rgb_scalar(pixels, pixels_out);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rgba_to_rgb_avx2(pixels: &mut [u8], out_len: usize) {
    use std::arch::x86_64::*;

    let len = pixels.len();
    let mut rd = 0usize;
    let mut wr = 0usize;

    // Process 8 pixels (32 bytes → 24 bytes) per iteration using 2× loads.
    while rd + 32 <= len && wr + 24 <= out_len {
        // Load 8 RGBA pixels = 32 bytes.
        let v0 = _mm256_loadu_si256(pixels.as_ptr().add(rd) as *const __m256i);

        // We want to extract the 3-byte RGB components from each of the 8 pixels.
        //
        // v0 = [B0,G0,R0,A0 | B1,G1,R1,A1 | ... | B7,G7,R7,A7]
        //
        // We use a combination of shuffles and permutes.
        //
        // Approach: partition into low 4 pixels (bytes 0-15) and high 4 pixels (bytes 16-31).
        let lo = _mm256_castsi256_si128(v0); // bytes  0-15 (pixels 0-3)
        let hi = _mm256_extracti128_si256(v0, 1); // bytes 16-31 (pixels 4-7)

        // Shuffle each 128-bit lane to gather RGB bytes.
        // For 4 pixels [B0 R0 G0 A0 B1 R1 G1 A1 ...], we want [B0,R0,G0, B1,R1,G1, B2,R2,G2, B3,R3,G3, ...]
        //
        // _mm_shuffle_epi8 uses a control mask. For each of 16 result bytes, the control byte:
        //   bit 7   = 0 (zero out if 1, but we don't want that)
        //   bits 3-0 = source byte index (with saturation within each 16-byte lane)
        //
        // Source lane bytes: 0(B0),1(G0),2(R0),3(A0),4(B1),5(G1),6(R1),7(A1),
        //                    8(B2),9(G2),10(R2),11(A2),12(B3),13(G3),14(R3),15(A3)
        //
        // Desired: 0(B0),1(G0),2(R0), 4(B1),5(G1),6(R1), 8(B2),9(G2),10(R2), 12(B3),13(G3),14(R3)
        //          leaving 4 bytes padding or we compress fully.
        //
        // Actually we want packed: 12 bytes (4×3) stored contiguously.
        // So bytes 0-11 = [B0,G0,R0, B1,G1,R1, B2,G2,R2, B3,G3,R3]
        //
        // Control mask indices: 0,1,2, 4,5,6, 8,9,10, 12,13,14,  (3,7,11,15 don't matter)
        // Use 0x80 for "don't care" / zero bytes to fill remaining positions.
        //
        // Better: use two shuffles per lane to produce two 8-byte halves, then mash together.
        // Or use compress/VPERM approach.
        //
        // Simplest reliable path for AVX2:
        //   lo_shuf = shuffle lo to get [B0,G0,R0, B1,G1,R1, 0,0,  B2,G2,R2, B3,G3,R3, 0,0]
        //   hi_shuf = same for hi pixels
        // Then compact with a 16-bit permute?  Too messy in pure SIMD for edge case.
        //
        // Fall back to two-128-bit approach: process 4 pixels at a time with SSSE3-style shuffle.
        // For 4 RGBA pixels → 12 RGB bytes:
        //   shuffle with mask [0,1,2, 4,5,6, 8,9,10, 12,13,14, 0..0] won't compress.
        //
        // Let's just use the 2-register approach: one shuffle produces 16 bytes containing
        // 12 useful + 4 padding.  Then we store only the first 12 bytes.
        //
        // Better: use _mm_shuffle_epi8 with a mask that places the 12 RGB bytes first,
        // then store 12 bytes.

        // Mask for low 4 pixels (bytes 0-15): extract B,G,R from each pixel byte positions
        // 0(B0),1(G0),2(R0),3(A0), 4(B1),5(G1),6(R1),7(A1), 8(B2),9(G2),10(R2),11(A2), 12(B3),13(G3),14(R3),15(A3)
        // We want: B0,G0,R0, B1,G1,R1, B2,G2,R2, B3,G3,R3 = 12 bytes
        // Put them at positions 0-11, rest don't matter:
        let mask = _mm_setr_epi8(
            0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -128i8, -128i8, -128i8, -128i8,
        );

        let lo_shuf = _mm_shuffle_epi8(lo, mask);
        let hi_shuf = _mm_shuffle_epi8(hi, mask);

        // Store 12 bytes from lo_shuf, then 12 from hi_shuf.
        std::ptr::copy_nonoverlapping(
            &lo_shuf as *const _ as *const u8,
            pixels.as_mut_ptr().add(wr),
            12,
        );
        std::ptr::copy_nonoverlapping(
            &hi_shuf as *const _ as *const u8,
            pixels.as_mut_ptr().add(wr + 12),
            12,
        );

        rd += 32;
        wr += 24;
    }

    if rd < len {
        let rem = len - rd;
        pixels.copy_within(rd..len, wr);
        rgba_to_rgb_scalar(&mut pixels[wr..wr + rem], out_len - wr);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn rgba_to_rgb_ssse3(pixels: &mut [u8], out_len: usize) {
    use std::arch::x86_64::*;

    let len = pixels.len();
    let mut rd = 0usize;
    let mut wr = 0usize;

    // Process 4 pixels (16 bytes → 12 bytes) per iteration.
    let mask = _mm_setr_epi8(
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -128i8, -128i8, -128i8, -128i8,
    );

    while rd + 16 <= len && wr + 12 <= out_len {
        let v = _mm_loadu_si128(pixels.as_ptr().add(rd) as *const __m128i);
        let shuffled = _mm_shuffle_epi8(v, mask);
        std::ptr::copy_nonoverlapping(
            &shuffled as *const _ as *const u8,
            pixels.as_mut_ptr().add(wr),
            12,
        );
        rd += 16;
        wr += 12;
    }

    if rd < len {
        let rem = len - rd;
        pixels.copy_within(rd..len, wr);
        rgba_to_rgb_scalar(&mut pixels[wr..wr + rem], out_len - wr);
    }
}

#[allow(unused_variables)]
fn rgba_to_rgb_scalar(pixels: &mut [u8], out_len: usize) {
    // out_len is unused — we compute from the slice length.
    let n = pixels.len();
    assert!(
        n % 4 == 0,
        "rgba_to_rgb_scalar: length must be multiple of 4"
    );
    let mut rd = 0usize;
    let mut wr = 0usize;
    while rd + 4 <= n {
        pixels[wr] = pixels[rd];
        pixels[wr + 1] = pixels[rd + 1];
        pixels[wr + 2] = pixels[rd + 2];
        rd += 4;
        wr += 3;
    }
}

// ---------------------------------------------------------------------------
// RGB → RGBA  (fill alpha = 255)
// ---------------------------------------------------------------------------

/// Convert RGB bytes to RGBA bytes (in-place safe dispatch).
///
/// The buffer must be pre-sized to hold the RGBA output.
/// `rgb_len` is the number of RGB bytes (must be multiple of 3).
///
/// This works **back-to-front** to avoid clobbering unread source data.
pub fn rgb_to_rgba(pixels: &mut [u8], rgb_len: usize) {
    assert!(
        rgb_len % 3 == 0,
        "rgb_to_rgba: rgb_len must be a multiple of 3"
    );
    let rgba_len = pixels.len();
    assert_eq!(
        rgba_len,
        rgb_len * 4 / 3,
        "rgb_to_rgba: rgba buffer must be exactly rgb_len * 4/3 bytes"
    );

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { rgb_to_rgba_avx2(pixels, rgb_len) };
        }
        if is_x86_feature_detected!("ssse3") {
            return unsafe { rgb_to_rgba_ssse3(pixels, rgb_len) };
        }
    }

    rgb_to_rgba_scalar(pixels, rgb_len);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rgb_to_rgba_avx2(pixels: &mut [u8], rgb_len: usize) {
    use std::arch::x86_64::*;

    // Convert 8 RGB pixels (24 bytes) → 8 RGBA pixels (32 bytes) per iteration.
    // Process back-to-front to avoid clobbering R/G/B before they are read.
    //
    // RGB input (24 bytes):  [B0,G0,R0, B1,G1,R1, ..., B7,G7,R7]
    // RGBA output (32 bytes): [B0,G0,R0,255, B1,G1,R1,255, ..., B7,G7,R7,255]
    //
    // We use two 128-bit loads of 12 RGB bytes each (4 pixels), expanded via shuffles.
    // Using a single 256-bit load split into 128-bit lanes would misalign because
    // 24 RGB bytes (8 pixels) straddle the 16-byte lane boundary: bytes 0-15 contain
    // pixels 0-4 (B4 at byte 12, G4 at 13, R4 at 14, B5 at 15), and bytes 16-23 contain
    // pixels 5-7 (G5 at 16, R5 at 17). Splitting at byte 16 splits pixel 5 mid-pixel.

    let mut rd = rgb_len;
    let mut wr = pixels.len();

    while rd >= 24 {
        rd -= 24;
        wr -= 32;

        // Load 12 bytes for pixels 0-3 (starting at rd).
        // _mm_loadu_si128 reads 16 bytes; only positions 0-11 are used by the shuffle.
        let lo = _mm_loadu_si128(pixels.as_ptr().add(rd) as *const __m128i);
        // Load 12 bytes for pixels 4-7 (starting at rd + 12).
        // rd + 12 + 16 = rd + 28. Since rd >= 0 usually and rd + 24 <= rgb_len,
        // we need rd + 28 <= rgba_len. At worst, rd = rgb_len - 24, so
        // rd + 28 = rgb_len + 4 <= rgba_len = rgb_len * 4/3 (true when rgb_len >= 12).
        let hi = _mm_loadu_si128(pixels.as_ptr().add(rd + 12) as *const __m128i);

        // For each 4-pixel group, expand 12 bytes → 16 bytes with alpha.
        // Shuffle maps: positions 0-2 → B,G,R; position 3 → 0 (placeholder for alpha).
        // Then OR with alpha mask to set placeholder bytes to 0xFF.
        let expand_mask = _mm_setr_epi8(
            0, 1, 2, -128i8, // B0,G0,R0, placeholder
            3, 4, 5, -128i8, // B1,G1,R1, placeholder
            6, 7, 8, -128i8, // B2,G2,R2, placeholder
            9, 10, 11, -128i8, // B3,G3,R3, placeholder
        );
        let alpha_mask = _mm_setr_epi8(0, 0, 0, -1i8, 0, 0, 0, -1i8, 0, 0, 0, -1i8, 0, 0, 0, -1i8);

        let lo_exp = _mm_shuffle_epi8(lo, expand_mask);
        let lo_exp = _mm_or_si128(lo_exp, alpha_mask);
        let hi_exp = _mm_shuffle_epi8(hi, expand_mask);
        let hi_exp = _mm_or_si128(hi_exp, alpha_mask);

        _mm_storeu_si128(pixels.as_mut_ptr().add(wr) as *mut __m128i, lo_exp);
        _mm_storeu_si128(pixels.as_mut_ptr().add(wr + 16) as *mut __m128i, hi_exp);
    }

    // Handle remaining pixels (< 24) — in-place scalar conversion back-to-front.
    if rd > 0 {
        for i in (0..rd / 3).rev() {
            let src = i * 3;
            let dst = i * 4;
            let b = pixels[src];
            let g = pixels[src + 1];
            let r = pixels[src + 2];
            pixels[dst] = b;
            pixels[dst + 1] = g;
            pixels[dst + 2] = r;
            pixels[dst + 3] = 255;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn rgb_to_rgba_ssse3(pixels: &mut [u8], rgb_len: usize) {
    use std::arch::x86_64::*;

    let expand_mask = _mm_setr_epi8(
        0, 1, 2, -128i8, 3, 4, 5, -128i8, 6, 7, 8, -128i8, 9, 10, 11, -128i8,
    );
    let alpha_mask = _mm_setr_epi8(0, 0, 0, -1i8, 0, 0, 0, -1i8, 0, 0, 0, -1i8, 0, 0, 0, -1i8);

    let mut rd = rgb_len;
    let mut wr = pixels.len();

    while rd >= 12 {
        rd -= 12;
        wr -= 16;

        let v = _mm_loadu_si128(pixels.as_ptr().add(rd) as *const __m128i);
        let exp = _mm_shuffle_epi8(v, expand_mask);
        let exp = _mm_or_si128(exp, alpha_mask);
        _mm_storeu_si128(pixels.as_mut_ptr().add(wr) as *mut __m128i, exp);
    }

    if rd > 0 {
        // Remaining pixels (< 4) — scalar back-to-front.
        for i in (0..rd / 3).rev() {
            let src = i * 3;
            let dst = i * 4;
            let b = pixels[src];
            let g = pixels[src + 1];
            let r = pixels[src + 2];
            pixels[dst] = b;
            pixels[dst + 1] = g;
            pixels[dst + 2] = r;
            pixels[dst + 3] = 255;
        }
    }
}

fn rgb_to_rgba_scalar(pixels: &mut [u8], rgb_len: usize) {
    // Process back-to-front. Use temps to avoid overlap when rd+3 > wr.
    let mut rd = rgb_len;
    let mut wr = pixels.len();
    while rd >= 3 {
        rd -= 3;
        wr -= 4;
        let b = pixels[rd];
        let g = pixels[rd + 1];
        let r = pixels[rd + 2];
        pixels[wr] = b;
        pixels[wr + 1] = g;
        pixels[wr + 2] = r;
        pixels[wr + 3] = 255;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bgra_rgba_roundtrip() {
        let mut data: Vec<u8> = (0..128).map(|i| i as u8).collect();
        let original = data.clone();
        bgra_to_rgba(&mut data);
        bgra_to_rgba(&mut data); // swap back
        assert_eq!(data, original, "BGRA↔RGBA round-trip failed");
    }

    #[test]
    fn test_bgra_rgba_simple() {
        let mut pixels = vec![
            0x10, 0x20, 0x30, 0xFF, // pixel 0: B=0x10, G=0x20, R=0x30, A=0xFF
            0x40, 0x50, 0x60, 0xFF, // pixel 1: B=0x40, G=0x50, R=0x60, A=0xFF
        ];
        bgra_to_rgba(&mut pixels);
        assert_eq!(
            &pixels,
            &[
                0x30, 0x20, 0x10, 0xFF, // pixel 0: R=0x30, G=0x20, B=0x10, A=0xFF
                0x60, 0x50, 0x40, 0xFF, // pixel 1: R=0x60, G=0x50, B=0x40, A=0xFF
            ]
        );
    }

    #[test]
    fn test_rgba_to_rgb() {
        let mut pixels = vec![10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255];
        rgba_to_rgb(&mut pixels);
        // Length is unchanged (takes &mut [u8]); RGBA data at front is compacted.
        assert_eq!(&pixels[0..3], &[10, 20, 30]);
        assert_eq!(&pixels[3..6], &[40, 50, 60]);
        assert_eq!(&pixels[6..9], &[70, 80, 90]);
    }

    #[test]
    fn test_rgb_to_rgba() {
        let rgb = vec![10u8, 20, 30, 40, 50, 60];
        let mut rgba = vec![0u8; 8];
        rgba[..6].copy_from_slice(&rgb);
        rgb_to_rgba(&mut rgba, 6);

        assert_eq!(rgba.len(), 8);
        assert_eq!(&rgba[0..4], &[10, 20, 30, 255]);
        assert_eq!(&rgba[4..8], &[40, 50, 60, 255]);
    }

    #[test]
    fn test_rgba_rgb_roundtrip() {
        // Use data where alpha=255 (rgb_to_rgba fills alpha=255).
        let original_rgba: Vec<u8> = (0..120)
            .map(|i| {
                if i % 4 == 3 {
                    255
                } else {
                    i % 4 * 64 + i / 4 % 64
                }
            })
            .collect();
        let mut rgba = original_rgba.clone();
        let rgb_len = rgba.len() * 3 / 4;

        // RGBA → RGB (in-place compaction, length unchanged)
        rgba_to_rgb(&mut rgba);

        // RGB → RGBA — extract compacted RGB prefix, expand back
        let mut out = vec![0u8; original_rgba.len()];
        out[..rgb_len].copy_from_slice(&rgba[..rgb_len]);
        rgb_to_rgba(&mut out, rgb_len);
        assert_eq!(out, original_rgba, "RGBA→RGB→RGBA roundtrip failed");
    }
}
