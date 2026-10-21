// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Two stills compared pixel by pixel, and a piece of one enlarged to look at.
//!
//! `cargo run --release -p fieldform-gpu --example png_diff -- diff <a.png> <b.png> <out.png>`
//! counts the pixels that differ by more than a few levels, says where they lie, and writes
//! the difference: grey where the two agree, red by how much where they do not.
//!
//! `... -- crop <in.png> <out.png> <x> <y> <w> <h> <scale>` cuts a piece out and enlarges it
//! with no smoothing, so a pixel stays a pixel.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("diff") => diff(&args[1], &args[2], &args[3]),
        Some("crop") => crop(&args[1], &args[2], &args[3..]),
        _ => eprintln!("png_diff diff <a> <b> <out> | crop <in> <out> <x> <y> <w> <h> <scale>"),
    }
}

fn diff(a: &str, b: &str, out: &str) {
    let a = image::open(a).expect("a opens").to_rgb8();
    let b = image::open(b).expect("b opens").to_rgb8();
    assert_eq!(a.dimensions(), b.dimensions(), "the two are not the same size");
    let (w, h) = a.dimensions();
    let mut d = image::RgbImage::new(w, h);
    let (mut over, mut big, mut most) = (0u64, 0u64, 0u8);
    let (mut lo, mut hi) = ((w, h), (0u32, 0u32));
    for y in 0..h {
        for x in 0..w {
            let (p, q) = (a.get_pixel(x, y).0, b.get_pixel(x, y).0);
            let delta = (0..3).map(|c| p[c].abs_diff(q[c])).max().unwrap();
            most = most.max(delta);
            if delta > 8 {
                over += 1;
                big += u64::from(delta > 40);
                lo = (lo.0.min(x), lo.1.min(y));
                hi = (hi.0.max(x), hi.1.max(y));
                d.put_pixel(x, y, image::Rgb([delta.saturating_mul(4), 0, 0]));
            } else {
                let g = ((u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2])) / 6) as u8;
                d.put_pixel(x, y, image::Rgb([g, g, g]));
            }
        }
    }
    d.save(out).expect("writable");
    println!(
        "{over} of {} pixels differ by more than 8 levels ({big} by more than 40; the largest {most}), within ({}, {})-({}, {}); wrote {out}",
        u64::from(w) * u64::from(h),
        lo.0,
        lo.1,
        hi.0,
        hi.1
    );
}

fn crop(input: &str, out: &str, rest: &[String]) {
    let n: Vec<u32> = rest.iter().map(|s| s.parse().expect("a number")).collect();
    let (x, y, w, h, scale) = (n[0], n[1], n[2], n[3], n[4]);
    let src = image::open(input).expect("opens").to_rgb8();
    let piece = image::imageops::crop_imm(&src, x, y, w, h).to_image();
    let big = image::imageops::resize(&piece, w * scale, h * scale, image::imageops::FilterType::Nearest);
    big.save(out).expect("writable");
    println!("wrote {out}, {}x{} of {}x{} at {scale}x", w, h, src.width(), src.height());
}
