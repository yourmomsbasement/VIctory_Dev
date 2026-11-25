//! Explicit mip-zero PCF for known D16_UNORM views on a linear clamp sampler.
use super::{Attr, Bin, Cast, Lower, Source, V, pred};
use crate::emit::StaticSampler;

pub(super) fn linear_clamp_less_equal(source: &Source) -> bool {
    let Source::Constant(words) = source else {
        return false;
    };
    let Some(w) = words.get(..4).and_then(|w| w.try_into().ok()) else {
        return false;
    };
    let s = StaticSampler::from_words(w);
    s.address == [3; 3] && s.comparison == 4 && matches!(s.filter, 0x14 | 0x15) && s.min_lod == 0.0
}

impl Lower<'_> {
    // Return the dimensions, fraction along each axis, and the four red values
    // in top-left, top-right, bottom-left, bottom-right order.
    pub(super) fn clamped_texels(&mut self, h: V, u: V, v: V) -> ([V; 2], [V; 2], [V; 4]) {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let f32t = self.s.t.f32;
        let zero = self.s.m.i32(0);
        let one = self.s.m.i32(1);
        let undef = self.s.m.undef(i32t);
        let ty = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let dims = self
            .s
            .op(c, "getDimensions", Attr::ReadOnly, ty, 72, &[h, zero]);
        let [width, height] = [0, 1].map(|k| self.s.m.extract(c, dims, k));
        let half = self.s.m.f32(0.5f32.to_bits());
        let [(xs, fx), (ys, fy)] = [(u, width), (v, height)].map(|(coord, size)| {
            let bound = self.s.op(c, "unary.f32", Attr::ReadNone, f32t, 7, &[coord]);
            let dim = self.s.m.cast(c, Cast::UiToFp, size, f32t);
            let at = self.s.m.bin(c, Bin::Mul, bound, dim);
            let center = self.s.m.bin(c, Bin::Sub, at, half);
            let floor = self
                .s
                .op(c, "unary.f32", Attr::ReadNone, f32t, 27, &[center]);
            let fraction = self.s.m.bin(c, Bin::Sub, center, floor);
            let low = self.s.m.cast(c, Cast::FpToSi, floor, i32t);
            let high = self.s.m.bin(c, Bin::Add, low, one);
            let last = self.s.m.bin(c, Bin::Sub, size, one);
            let pair = [low, high].map(|p| {
                let p = self
                    .s
                    .op(c, "binary.i32", Attr::ReadNone, i32t, 37, &[p, zero]);
                self.s
                    .op(c, "binary.i32", Attr::ReadNone, i32t, 40, &[p, last])
            });
            (pair, fraction)
        });
        let taps = [(0, 0), (1, 0), (0, 1), (1, 1)].map(|(x, y)| {
            let sample = self.s.op(
                c,
                "textureLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                66,
                &[h, zero, xs[x], ys[y], undef, undef, undef, undef],
            );
            self.s.m.extract(c, sample, 0)
        });
        ([width, height], [fx, fy], taps)
    }

    pub(super) fn d16_compare(&mut self, h: V, u: V, v: V, reference: V) -> V {
        let (_, [fx, fy], taps) = self.clamped_texels(h, u, v);
        let c = self.cur;
        let f = self.s.t.f32;
        let scale = self.s.m.f32(65535.0f32.to_bits());
        let code = |lower: &mut Self, value| {
            let scaled = lower.s.m.bin(c, Bin::Mul, value, scale);
            lower.s.op(c, "unary.f32", Attr::ReadNone, f, 26, &[scaled]) // Round_ne
        };
        let saturated = self
            .s
            .op(c, "unary.f32", Attr::ReadNone, f, 7, &[reference]);
        let reference = code(self, saturated);
        let [tl, tr, bl, br] = taps.map(|depth| {
            let depth = code(self, depth);
            let pass = self.s.m.cmp(c, pred::OLE, reference, depth);
            self.s.m.cast(c, Cast::UiToFp, pass, f)
        });
        let mut lerp = |a, b, weight| {
            let delta = self.s.m.bin(c, Bin::Sub, b, a);
            let weighted = self.s.m.bin(c, Bin::Mul, delta, weight);
            self.s.m.bin(c, Bin::Add, a, weighted)
        };
        let top = lerp(tl, tr, fx);
        let bottom = lerp(bl, br, fx);
        lerp(top, bottom, fy)
    }
}
