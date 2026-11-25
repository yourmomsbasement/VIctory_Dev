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
        self.append_trace(inside, |lower| {
            let zero = lower.s.m.i32(0);
            let rgba = [0, 1, 2, 3].map(|i| {
                let v = lower.s.m.extract(lower.cur, result, i);
                lower.as_int(v)
            });
            let pc = lower.s.m.i32(pc as u32);
            let unknown = lower.s.m.i32(u32::MAX);
            let heap = lower
                .s
                .sample_indices
                .get(&handle)
                .copied()
                .unwrap_or(unknown);
            let [slot, dword] = table.unwrap_or([u32::MAX; 2]).map(|n| lower.s.m.i32(n));
            let [u, v, lod] = uv_lod.map(|f| lower.as_int(f));
            let dims = lower
                .s
                .m
                .named_struct("dx.types.Dimensions", vec![lower.s.t.i32; 4]);
            let dimensions = lower.s.op(
                lower.cur,
                "getDimensions",
                Attr::ReadOnly,
                dims,
                72,
                &[handle, zero],
            );
            let [width, height, levels] = [0, 1, 3].map(|k| lower.s.m.extract(lower.cur, dimensions, k));
            [[pc, xpos, depth, heap], [u, v, slot, dword], rgba, [width, height, levels, lod]]
        });
        Ok(())
    }

    /// The CPU supplies the same bounded, fence-backed record buffer for both probes.
    fn append_trace(&mut self, inside: V, fields: impl FnOnce(&mut Self) -> [[V; 4]; 4]) {
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
        let mask = self.s.m.i8(15);
        for [a, b, d, e] in fields(self) {
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
    }

    pub(super) fn trace_compute(
        &mut self, inst: &crate::ir::Inst, probe: super::super::ComputeProbe,
        operands: [V; 4], guard: Option<V>, texture: Option<(V, u32)>,
    ) -> Result<(), Why> {
        use crate::ir::Reg;
        let [x, y] = probe.coord_regs.map(|r| inst.read(Reg::V(r)));
        let x = self.i32_of(x.ok_or("a compute trace without its X coordinate")?)?;
        let y = self.i32_of(y.ok_or("a compute trace without its Y coordinate")?)?;
        let [want_x, want_y] = probe.pixel.map(|n| self.s.m.i32(n));
        let on_x = self.s.m.cmp(self.cur, pred::EQ, x, want_x);
        let on_y = self.s.m.cmp(self.cur, pred::EQ, y, want_y);
        let mut inside = self.s.m.bin(self.cur, Bin::And, on_x, on_y);
        if let Some(active) = guard {
            inside = self.s.m.bin(self.cur, Bin::And, inside, active);
        }
        self.append_trace(inside, |lower| {
            let pc = lower.s.m.i32(probe.pc as u32);
            let zero = lower.s.m.i32(0);
            let op = lower.s.m.i32(texture.map_or(0, |(_, op)| op));
            let extra = texture.map_or([zero; 4], |(h, _)| lower.probe_texel(h, operands[0], operands[1]));
            [[pc, x, y, op], operands.map(|v| lower.as_int(v)), extra, [zero; 4]]
        });
        Ok(())
    }

    /// Read the nearest clamped texel at mip zero independently of the sampler.
    fn probe_texel(&mut self, h: V, u: V, v: V) -> [V; 4] {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let f32t = self.s.t.f32;
        let zero = self.s.m.i32(0);
        let one = self.s.m.i32(1);
        let undef = self.s.m.undef(i32t);
        let ty = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let dims = self.s.op(c, "getDimensions", Attr::ReadOnly, ty, 72, &[h, zero]);
        let [width, height] = [0, 1].map(|k| self.s.m.extract(c, dims, k));
        let [x, y] = [(u, width), (v, height)].map(|(coord, size)| {
            let bound = self.s.op(c, "unary.f32", Attr::ReadNone, f32t, 7, &[coord]);
            let dim = self.s.m.cast(c, Cast::UiToFp, size, f32t);
            let at = self.s.m.bin(c, Bin::Mul, bound, dim);
            let at = self.s.m.cast(c, Cast::FpToUi, at, i32t);
            let last = self.s.m.bin(c, Bin::Sub, size, one);
            self.s.op(c, "binary.i32", Attr::ReadNone, i32t, 40, &[at, last])
        });
        let sample = self.s.op(c, "textureLoad.f32", Attr::ReadOnly, self.s.t.res_ret, 66,
            &[h, zero, x, y, undef, undef, undef, undef]);
        let first = self.s.m.extract(c, sample, 0);
        let unknown = self.s.m.i32(u32::MAX);
        let heap = self.s.sample_indices.get(&h).copied().unwrap_or(unknown);
        [width, height, self.as_int(first), heap]
    }
}
