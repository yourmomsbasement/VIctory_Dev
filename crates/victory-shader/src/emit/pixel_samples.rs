use super::{Attr, Bin, Cast, Input, Lower, V, Val, Why, pred};
use crate::emit::{MAX_PIXEL_SAMPLES, PIXEL_SAMPLE_WORDS, SAMPLE_SPAN, SLOT_PIXEL_SAMPLES};

impl Lower<'_> {
    /// Append after the sample, so the original derivative and color calculation
    /// execute in their original control flow. Depth distinguishes overdraw.
    pub(super) fn trace_sample(
        &mut self,
        pc: usize,
        handle: V,
        table: Option<[u32; 2]>,
        uv_lod: [V; 3],
        result: V,
        guard: Option<V>,
    ) -> Result<(), Why> {
        let Some([first, last, y]) = SAMPLE_SPAN.get() else {
            return Ok(());
        };
        let c = self.cur;
        let pos = [0, 1, 2].map(|i| self.input(Input::Position(i)));
        let [Val::I32(px), Val::I32(py), Val::I32(depth)] = pos else {
            return Err("a sample trace without a pixel position");
        };
        let [xpos, ypos] = [px, py].map(|v| {
            let f = self.as_float(v);
            self.s.m.cast(c, Cast::FpToUi, f, self.s.t.i32)
        });
        let [first, last, y] = [first, last, y].map(|n| self.s.m.i32(n));
        let left = self.s.m.cmp(c, pred::IUGE, xpos, first);
        let right = self.s.m.cmp(c, pred::IULE, xpos, last);
        let row = self.s.m.cmp(c, pred::EQ, ypos, y);
        let inside = self.s.m.bin(c, Bin::And, left, right);
        let mut inside = self.s.m.bin(c, Bin::And, inside, row);
        if let Some(active) = guard {
            inside = self.s.m.bin(c, Bin::And, inside, active);
        }
        let pixel = self.begin_guarded(Some(inside));
        let buffer = self.s.root_uav_handle(SLOT_PIXEL_SAMPLES);
        let undef = self.s.m.undef(self.s.t.i32);
        let [zero, one, four, eight, sixteen, stride] =
            [0, 1, 4, 8, 16, PIXEL_SAMPLE_WORDS * 4].map(|v| self.s.m.i32(v));
        let mask = self.s.m.i8(1);
        let header = self.s.op(
            self.cur,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[buffer, four, undef, mask, four],
        );
        let header = self.s.m.extract(self.cur, header, 0);
        let header = self.as_int(header);
        let enabled = self.s.m.cmp(self.cur, pred::NE, header, zero);
        let capture = self.begin_guarded(Some(enabled));
        let index = self.s.op(
            self.cur,
            "atomicBinOp.i32",
            Attr::NoUnwind,
            self.s.t.i32,
            78,
            &[buffer, zero, zero, undef, undef, one],
        );
        let limit = self.s.m.i32(MAX_PIXEL_SAMPLES);
        let fits = self.s.m.cmp(self.cur, pred::IULT, index, limit);
        let append = self.begin_guarded(Some(fits));
        let offset = self.s.m.bin(self.cur, Bin::Mul, index, stride);
        let mut address = self.s.m.bin(self.cur, Bin::Add, offset, eight);
        let rgba = [0, 1, 2, 3].map(|i| {
            let v = self.s.m.extract(self.cur, result, i);
            self.as_int(v)
        });
        let pc = self.s.m.i32(pc as u32);
        let unknown = self.s.m.i32(u32::MAX);
        let heap = self
            .s
            .sample_indices
            .get(&handle)
            .copied()
            .unwrap_or(unknown);
        let [slot, dword] = table.unwrap_or([u32::MAX; 2]).map(|n| self.s.m.i32(n));
        let [u, v, lod] = uv_lod.map(|f| self.as_int(f));
        let dims = self
            .s
            .m
            .named_struct("dx.types.Dimensions", vec![self.s.t.i32; 4]);
        let dimensions = self.s.op(
            self.cur,
            "getDimensions",
            Attr::ReadOnly,
            dims,
            72,
            &[handle, zero],
        );
        let [width, height, levels] = [0, 1, 3].map(|k| self.s.m.extract(self.cur, dimensions, k));
        let mask = self.s.m.i8(15);
        for [a, b, d, e] in [
            [pc, xpos, depth, heap],
            [u, v, slot, dword],
            rgba,
            [width, height, levels, lod],
        ] {
            self.s.op(
                self.cur,
                "rawBufferStore.i32",
                Attr::NoUnwind,
                self.s.t.void,
                140,
                &[buffer, address, undef, a, b, d, e, mask, four],
            );
            address = self.s.m.bin(self.cur, Bin::Add, address, sixteen);
        }
        self.end_guarded(append, &[]);
        self.end_guarded(capture, &[]);
        self.end_guarded(pixel, &[]);
        Ok(())
    }
}
