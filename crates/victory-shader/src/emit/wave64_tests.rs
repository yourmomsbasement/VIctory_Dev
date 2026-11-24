//! Tests for Wave64-on-Wave32 and portable LDS Wave64 compute lowering,
//! using synthetic GFX10 machine code on hardware GPU and WARP.

use super::{HostShader, TranslateOptions, Wave64Mode, emit_with, translate_with};
use crate::dxbc::{Container, FourCC};
use crate::native::{ComputeIo, HwStage, NativeProgram, RootParam, RootType, StageIo};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const EXEC_LO: u32 = 126;
const EXEC_HI: u32 = 127;
const VCC: u32 = 106;
const NULL_OP: u32 = 125;

// Inline constants and register operands in 9-bit GFX10 source fields.
fn vgpr(v: u32) -> u32 {
    256 + v
}
fn inline_u(n: u32) -> u32 {
    assert!(n <= 64);
    0x80 + n
}
const NEG_ONE: u32 = 0xC1;
const LIT: u32 = 0xFF;

fn sop1(dst: u32, op: u32, src0: u32) -> u32 {
    0xBE80_0000 | (dst << 16) | (op << 8) | src0
}
fn sop2(dst: u32, op: u32, src0: u32, src1: u32) -> u32 {
    0x8000_0000 | (op << 23) | (dst << 16) | (src1 << 8) | src0
}
fn sopc(op: u32, src0: u32, src1: u32) -> u32 {
    0xBF00_0000 | (op << 16) | (src1 << 8) | src0
}
fn sopp(op: u32, imm: i16) -> u32 {
    0xBF80_0000 | (op << 16) | u32::from(imm as u16)
}
fn vop1(dst: u32, op: u32, src0: u32) -> u32 {
    0x7E00_0000 | (dst << 17) | (op << 9) | src0
}
fn vop2(op: u32, dst: u32, src0: u32, vsrc1: u32) -> u32 {
    (op << 25) | (dst << 17) | (vsrc1 << 9) | src0
}
fn vopc(op: u32, src0: u32, vsrc1: u32) -> u32 {
    0x7C00_0000 | (op << 17) | (vsrc1 << 9) | src0
}
fn vop3(op: u32, dst: u32, src0: u32, src1: u32, src2: u32) -> [u32; 2] {
    [
        0xD400_0000 | (op << 16) | dst,
        src0 | (src1 << 9) | (src2 << 18),
    ]
}
fn ds_op(op: u32, off0: u8, off1: u8, addr: u32, data0: u32, vdst: u32) -> [u32; 2] {
    [
        0xD800_0000 | (op << 18) | (u32::from(off1) << 8) | u32::from(off0),
        addr | (data0 << 8) | (vdst << 24),
    ]
}
fn dpp_row_shr(
    vop2_op: u32,
    dst: u32,
    vsrc1: u32,
    vsrc0: u32,
    ctrl: u32,
    row_mask: u32,
) -> [u32; 2] {
    [
        vop2(vop2_op, dst, 0xFA, vsrc1),
        vsrc0 | (ctrl << 8) | (1 << 19) | (0xF << 24) | (row_mask << 28),
    ]
}

fn s_mov_b32(dst: u32, src0: u32) -> u32 {
    sop1(dst, 0x03, src0)
}
fn s_mov_b64(dst: u32, src0: u32) -> u32 {
    sop1(dst, 0x04, src0)
}
fn s_bcnt1_i32_b64(dst: u32, src0: u32) -> u32 {
    sop1(dst, 0x10, src0)
}
fn s_ff1_i32_b64(dst: u32, src0: u32) -> u32 {
    sop1(dst, 0x14, src0)
}
fn s_and_saveexec_b64(dst: u32, src0: u32) -> u32 {
    sop1(dst, 0x24, src0)
}
fn s_or_saveexec_b64(dst: u32, src0: u32) -> u32 {
    sop1(dst, 0x25, src0)
}
fn s_add_i32(dst: u32, src0: u32, src1: u32) -> u32 {
    sop2(dst, 0x02, src0, src1)
}
fn s_or_b64(dst: u32, src0: u32, src1: u32) -> u32 {
    sop2(dst, 0x11, src0, src1)
}
fn s_cmp_lt_u32(src0: u32, src1: u32) -> u32 {
    sopc(0x0A, src0, src1)
}
fn s_cbranch_scc1(delta_words: i16) -> u32 {
    sopp(0x05, delta_words)
}
fn s_cbranch_execz(delta_words: i16) -> u32 {
    sopp(0x08, delta_words)
}

fn v_mov_b32(dst: u32, src0: u32) -> u32 {
    vop1(dst, 0x01, src0)
}
fn v_readfirstlane_b32(sdst: u32, vsrc: u32) -> u32 {
    vop1(sdst, 0x02, vgpr(vsrc))
}
fn v_lshlrev_b32(dst: u32, shift: u32, vsrc1: u32) -> u32 {
    vop2(0x1A, dst, shift, vsrc1)
}
fn v_and_b32(dst: u32, src0: u32, vsrc1: u32) -> u32 {
    vop2(0x1B, dst, src0, vsrc1)
}
fn v_or_b32(dst: u32, src0: u32, vsrc1: u32) -> u32 {
    vop2(0x1C, dst, src0, vsrc1)
}
fn v_add_nc_u32(dst: u32, src0: u32, vsrc1: u32) -> u32 {
    vop2(0x25, dst, src0, vsrc1)
}
fn v_sub_nc_u32(dst: u32, src0: u32, vsrc1: u32) -> u32 {
    vop2(0x26, dst, src0, vsrc1)
}
fn v_cmp_lt_u32(src0: u32, vsrc1: u32) -> u32 {
    vopc(0xC1, src0, vsrc1)
}
fn v_cmp_eq_u32(src0: u32, vsrc1: u32) -> u32 {
    vopc(0xC2, src0, vsrc1)
}
fn v_cmp_gt_u32(src0: u32, vsrc1: u32) -> u32 {
    vopc(0xC4, src0, vsrc1)
}
fn v_cndmask_b32(dst: u32, src0: u32, src1: u32, mask_s: u32) -> [u32; 2] {
    vop3(0x101, dst, src0, src1, mask_s)
}
fn v_readlane_b32(sdst: u32, vsrc: u32, lane_op: u32) -> [u32; 2] {
    vop3(0x360, sdst, vgpr(vsrc), lane_op, 0)
}
fn v_writelane_b32(vdst: u32, src: u32, lane_op: u32) -> [u32; 2] {
    vop3(0x361, vdst, src, lane_op, 0)
}
fn v_mbcnt_lo_u32_b32(dst: u32, mask: u32, b: u32) -> [u32; 2] {
    vop3(0x365, dst, mask, b, 0)
}
fn v_mbcnt_hi_u32_b32(dst: u32, mask: u32, b: u32) -> [u32; 2] {
    vop3(0x366, dst, mask, b, 0)
}

fn s_load_dwordx4(sdst: u32, sbase: u32) -> [u32; 2] {
    [0xF408_0000 | (sdst << 6) | (sbase >> 1), NULL_OP << 25]
}

fn buffer_store_dword(vdata: u32, vaddr: u32, srsrc: u32) -> [u32; 2] {
    [
        0xE070_1000, // buffer_store_dword, offen=1
        vaddr | (vdata << 8) | ((srsrc >> 2) << 16) | (inline_u(0) << 24),
    ]
}

fn emit_uav_store(code: &mut Vec<u32>, data_vreg: u32, tid_vreg: u32) {
    // s0 holds heap base 0 (user param 0 = ResourceHeap); load view 0 into s[8:11].
    code.push(s_mov_b32(1, inline_u(0)));
    code.extend_from_slice(&s_load_dwordx4(8, 0));
    code.push(v_lshlrev_b32(15, inline_u(2), tid_vreg));
    code.extend_from_slice(&buffer_store_dword(data_vreg, 15, 8));
}

fn empty_dxbc() -> Vec<u8> {
    let mut b = Vec::with_capacity(32);
    b.extend_from_slice(b"DXBC");
    b.extend_from_slice(&[0u8; 16]);
    b.extend_from_slice(&1u32.to_le_bytes()); // major/minor
    b.extend_from_slice(&32u32.to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    b
}

fn make_cs_program(
    words: &[u32],
    threads: [u32; 3],
    lds_bytes: u32,
    group_id_x: bool,
) -> NativeProgram {
    let lds_granules = lds_bytes.div_ceil(512);
    // 2 user SGPRs (s[0:1] heap pointer); optional TGID.x in bit 7.
    let rsrc2 = (2 << 1) | (u32::from(group_id_x) << 7) | (lds_granules << 15);
    NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::ResourceHeap,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: Vec::new(),
        io: StageIo::Cs(ComputeIo { rsrc2, threads }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn emit_cs(native: &NativeProgram, mode: Wave64Mode) -> Result<HostShader, ShaderError> {
    let dxbc = empty_dxbc();
    let container = Container::parse(&dxbc)?;
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
    let st = structure::structure(&p);
    let res = resource::track(&p, &native.layout, native.stage, st.as_ref().ok())?;
    let io_res = io::raise(&p, native)?;
    let st = st?;
    emit_with(
        &container,
        native,
        &p,
        &st,
        &io_res,
        &res,
        TranslateOptions { wave64: mode },
    )
}

fn psv_wave_range(shader: &HostShader) -> (u32, u32) {
    let c = Container::parse(&shader.dxil).unwrap();
    let psv = c.part(FourCC::PSV0).unwrap().data;
    let min = u32::from_le_bytes(psv[20..24].try_into().unwrap());
    let max = u32::from_le_bytes(psv[24..28].try_into().unwrap());
    (min, max)
}

fn sfi0_flags(shader: &HostShader) -> u64 {
    let c = Container::parse(&shader.dxil).unwrap();
    match c.part(FourCC::SFI0) {
        Some(p) => u64::from_le_bytes(p.data[0..8].try_into().unwrap()),
        None => 0,
    }
}

#[test]
fn wave_mode_psv_and_flags() {
    // 128 threads reading 64-lane ballot + mbcnt.
    let mut code = Vec::new();
    code.push(v_cmp_lt_u32(inline_u(40), 0)); // vcc = (40 < v0) i.e. tid & 63 >= 41
    code.push(s_bcnt1_i32_b64(4, VCC));
    code.extend_from_slice(&v_mbcnt_lo_u32_b32(1, NEG_ONE, inline_u(0)));
    code.extend_from_slice(&v_mbcnt_hi_u32_b32(1, NEG_ONE, vgpr(1)));
    code.push(v_add_nc_u32(1, 4, 1));
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);

    let prog = make_cs_program(&code, [128, 1, 1], 0, false);

    let native = emit_cs(&prog, Wave64Mode::Native).unwrap();
    assert_eq!(psv_wave_range(&native), (64, 64));
    assert_ne!(sfi0_flags(&native) & 0x4000, 0);

    let wave32 = emit_cs(&prog, Wave64Mode::Wave32).unwrap();
    assert_eq!(psv_wave_range(&wave32), (32, 32));

    let emu = emit_cs(&prog, Wave64Mode::Emulate).unwrap();
    assert_eq!(psv_wave_range(&emu), (0, u32::MAX));
    assert_eq!(sfi0_flags(&emu) & 0x4000, 0);
}

#[test]
fn lds_overflow_fails_explicitly() {
    // Title uses full 32 KiB (8192 dwords) of LDS and reads a dynamic lane.
    let mut code = Vec::new();
    code.extend_from_slice(&v_readlane_b32(4, 0, vgpr(0)));
    code.push(v_mov_b32(1, 4));
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);

    let prog = make_cs_program(&code, [1024, 1, 1], 32768, false);
    assert!(emit_cs(&prog, Wave64Mode::Native).is_ok());

    for mode in [Wave64Mode::Wave32, Wave64Mode::Emulate] {
        let err = emit_cs(&prog, mode).unwrap_err();
        assert!(
            matches!(err, ShaderError::Unsupported { what, .. } if what.contains("32 KiB")),
            "expected explicit LDS overflow error in {mode:?}, got {err:?}"
        );
    }
}

#[test]
fn divergent_cross_half_ballot_is_rejected_instead_of_hanging() {
    // if (v0 < 32) { s4 = bcnt1(v0 < 10); v1 = s4; } else { v1 = 0; } store(v1)
    let mut code = Vec::new();
    code.push(v_mov_b32(1, inline_u(0)));
    code.push(v_cmp_gt_u32(inline_u(32), 0)); // 32 > v0
    code.push(s_and_saveexec_b64(6, VCC)); // s[6:7]
    let br_idx = code.len();
    code.push(0); // placeholder for s_cbranch_execz -> after_then
    code.push(v_cmp_gt_u32(inline_u(10), 0)); // 10 > v0
    code.push(s_bcnt1_i32_b64(4, VCC));
    code.push(v_mov_b32(1, 4));
    let after_then = code.len();
    code[br_idx] = s_cbranch_execz((after_then - (br_idx + 1)) as i16);
    code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);

    let prog = make_cs_program(&code, [128, 1, 1], 0, false);
    assert!(emit_cs(&prog, Wave64Mode::Native).is_ok());
    for mode in [Wave64Mode::Wave32, Wave64Mode::Emulate] {
        let err = emit_cs(&prog, mode).unwrap_err();
        assert!(
            matches!(err, ShaderError::Unsupported { .. }),
            "expected safe error for divergent ballot in {mode:?}, got {err:?}"
        );
    }
}

#[test]
fn divergent_readfirstlane_reads_its_halfs_first_lane_on_wave32() {
    // if (v0 < 32) { s4 = readfirstlane(v0); v1 = s4; } else { v1 = 0; } store(v1)
    // Inside divergent flow the halves cannot meet at a barrier: Wave32 reads
    // each half's own first active lane; Emulate refuses.
    let mut code = Vec::new();
    code.push(v_mov_b32(1, inline_u(0)));
    code.push(v_cmp_gt_u32(inline_u(32), 0)); // 32 > v0
    code.push(s_and_saveexec_b64(6, VCC)); // s[6:7]
    let br_idx = code.len();
    code.push(0); // placeholder for s_cbranch_execz -> after_then
    code.push(v_readfirstlane_b32(4, 0));
    code.push(v_mov_b32(1, 4));
    let after_then = code.len();
    code[br_idx] = s_cbranch_execz((after_then - (br_idx + 1)) as i16);
    code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);

    let prog = make_cs_program(&code, [128, 1, 1], 0, false);
    assert!(emit_cs(&prog, Wave64Mode::Native).is_ok());
    let wave32 = emit_cs(&prog, Wave64Mode::Wave32).unwrap();
    assert_eq!(psv_wave_range(&wave32), (32, 32));
    assert!(matches!(
        emit_cs(&prog, Wave64Mode::Emulate),
        Err(ShaderError::Unsupported { .. })
    ));
}

#[test]
fn divergent_uniform_readfirstlane_succeeds_without_barrier() {
    // s8 = 42; v2 = s8 (uniform across the group); v_mbcnt forces console_lanes;
    // if (tid < 40) s4 = readfirstlane(v2); must succeed on Wave32 and Emulate.
    let mut code = Vec::new();
    code.extend_from_slice(&v_mbcnt_lo_u32_b32(3, NEG_ONE, inline_u(0)));
    code.extend_from_slice(&v_mbcnt_hi_u32_b32(3, NEG_ONE, vgpr(3)));
    code.push(s_mov_b32(8, inline_u(42)));
    code.push(v_mov_b32(2, 8));
    code.push(v_mov_b32(1, 3));
    code.push(v_cmp_gt_u32(inline_u(40), 0));
    code.push(s_and_saveexec_b64(6, VCC));
    let br_idx = code.len();
    code.push(0);
    code.push(v_readfirstlane_b32(4, 2));
    code.push(v_add_nc_u32(1, 4, 3));
    let after_then = code.len();
    code[br_idx] = s_cbranch_execz((after_then - (br_idx + 1)) as i16);
    code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);

    let prog = make_cs_program(&code, [64, 1, 1], 0, false);
    for mode in [Wave64Mode::Native, Wave64Mode::Wave32, Wave64Mode::Emulate] {
        emit_cs(&prog, mode).unwrap_or_else(|e| panic!("{mode:?} failed: {e}"));
    }
}

#[test]
fn divergent_early_return_taints_subsequent_blocks() {
    // if (v0 < 32) return; // after_ret is non-uniform across the group!
    // bcnt1 in after_ret must fail instead of inserting a hanging barrier.
    let mut code = Vec::new();
    code.push(v_cmp_gt_u32(inline_u(32), 0));
    code.push(s_and_saveexec_b64(6, VCC));
    let br_idx = code.len();
    code.push(0);
    code.push(S_ENDPGM);
    let rest = code.len();
    code[br_idx] = s_cbranch_execz((rest - (br_idx + 1)) as i16);
    code.push(v_cmp_gt_u32(LIT, 0));
    code.push(96);
    code.push(s_bcnt1_i32_b64(4, VCC));
    code.push(v_mov_b32(1, 4));
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);

    let prog = make_cs_program(&code, [128, 1, 1], 0, false);
    assert!(emit_cs(&prog, Wave64Mode::Native).is_ok());
    for mode in [Wave64Mode::Wave32, Wave64Mode::Emulate] {
        assert!(matches!(
            emit_cs(&prog, mode),
            Err(ShaderError::Unsupported { .. })
        ));
    }
}

// Synthetic LDS ordering patterns. With a join, the writer and reader are
// separate EXEC-masked regions; without one, both accesses stay divergent.
fn lds_region_program(divergent_write: bool, join_between: bool, delay: bool) -> NativeProgram {
    let mut code = vec![
        v_mov_b32(3, inline_u(7)),
        v_lshlrev_b32(2, inline_u(2), 0),
        v_add_nc_u32(4, inline_u(7), 0),
    ];
    let mut writer_branch = None;
    if divergent_write {
        code.push(v_cmp_lt_u32(inline_u(31), 0)); // lanes 32..63
        code.push(s_and_saveexec_b64(6, VCC));
        writer_branch = Some(code.len());
        code.push(0);
        if delay {
            code.push(s_mov_b32(10, inline_u(0)));
            let top = code.len();
            code.extend([vop2(0x0B, 4, LIT, 4), 40503]);
            code.push(v_add_nc_u32(4, inline_u(1), 4));
            code.push(s_add_i32(10, 10, inline_u(1)));
            code.extend([s_cmp_lt_u32(10, LIT), 4096]);
            let at = code.len();
            code.push(s_cbranch_scc1((top as isize - (at + 1) as isize) as i16));
        }
    }
    code.extend(ds_op(0x0D, 0, 0, 2, 4, 0)); // ds_write_b32 v2, v4
    if join_between {
        if let Some(at) = writer_branch.take() {
            code[at] = s_cbranch_execz((code.len() - (at + 1)) as i16);
            code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
        }
        code.push(v_cmp_gt_u32(inline_u(32), 0)); // lanes 0..31
        code.push(s_and_saveexec_b64(6, VCC));
        let at = code.len();
        code.push(0);
        code.extend(ds_op(0x36, 128, 0, 2, 0, 3)); // ds_read_b32 v3, v2 offset:128
        code[at] = s_cbranch_execz((code.len() - (at + 1)) as i16);
    } else {
        code.extend(ds_op(0x36, 0, 0, 2, 0, 3)); // read in the writer's divergent block
        let at = writer_branch.expect("divergent writer");
        code[at] = s_cbranch_execz((code.len() - (at + 1)) as i16);
    }
    code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
    emit_uav_store(&mut code, 3, 0);
    code.push(S_ENDPGM);
    make_cs_program(&code, [64, 1, 1], 512, false)
}

#[test]
fn divergent_lds_without_a_uniform_fence_is_rejected() {
    let prog = lds_region_program(true, false, false);
    assert!(emit_cs(&prog, Wave64Mode::Native).is_ok());
    for mode in [Wave64Mode::Wave32, Wave64Mode::Emulate] {
        let err = match emit_cs(&prog, mode) {
            Err(err) => err,
            Ok(_) => panic!("expected an unfenceable LDS access to be refused in {mode:?}"),
        };
        assert!(
            matches!(err, ShaderError::Unsupported { what, .. }
            if what.contains("uniform LDS fence")),
            "{mode:?}: {err:?}"
        );
    }
}

#[test]
fn lds_regions_fence_at_uniform_joins_and_exits() {
    for (divergent_write, delay) in [(false, false), (true, false), (true, true)] {
        let prog = lds_region_program(divergent_write, true, delay);
        let p = ir::build(&prog.patched_code(), prog.wave64, prog.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &prog.layout, prog.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &prog).unwrap();
        let (uniform, _) = super::lower::compute_uniformity(&p, &st, &res, &io.inputs);
        let fences = super::lower::plan_lds_fences(&p, &uniform).unwrap();
        let writer = p
            .blocks
            .iter()
            .position(|b| b.insts.iter().any(|i| i.name() == "ds_write_b32"))
            .unwrap();
        let reader = p
            .blocks
            .iter()
            .position(|b| b.insts.iter().any(|i| i.name() == "ds_read_b32"))
            .unwrap();
        assert!(!uniform[reader]);
        assert!(
            fences
                .enter
                .iter()
                .chain(&fences.leave)
                .all(|&b| uniform[b])
        );
        if divergent_write {
            assert!(!uniform[writer]);
            let crate::ir::Exit::Jump(join) = p.blocks[writer].exit else {
                panic!("writer rejoins")
            };
            assert!(
                fences.enter.contains(&join),
                "the join has no LDS instruction but needs a fence"
            );
        } else {
            assert!(uniform[writer]);
            assert!(
                fences.leave.contains(&writer),
                "the write must finish before the divergent read"
            );
        }
        for mode in [Wave64Mode::Wave32, Wave64Mode::Emulate] {
            let mut shader = emit_cs(&prog, mode).unwrap();
            #[cfg(windows)]
            if let Ok(mut validator) = super::validator::Validator::load(
                super::validator::Validator::default_path().as_deref(),
            ) {
                validator.validate(&mut shader.dxil).unwrap();
            }
        }
    }
}

fn guarded_lds_program() -> NativeProgram {
    let mut code = vec![
        v_mov_b32(3, inline_u(7)),
        v_lshlrev_b32(2, inline_u(2), 0),
        v_add_nc_u32(4, inline_u(7), 0),
        v_cmp_lt_u32(inline_u(31), 0),
        s_and_saveexec_b64(6, VCC),
    ];
    code.extend(ds_op(0x0D, 0, 0, 2, 4, 0));
    code.extend([
        s_or_b64(EXEC_LO, EXEC_LO, 6),
        v_cmp_gt_u32(inline_u(32), 0),
        s_and_saveexec_b64(6, VCC),
    ]);
    code.extend(ds_op(0x36, 128, 0, 2, 0, 3));
    code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
    emit_uav_store(&mut code, 3, 0);
    code.push(S_ENDPGM);
    make_cs_program(&code, [64, 1, 1], 512, false)
}

#[test]
fn guarded_lds_in_a_uniform_block_stays_fenceable() {
    let prog = guarded_lds_program();
    let p = ir::build(&prog.patched_code(), prog.wave64, prog.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &prog.layout, prog.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &prog).unwrap();
    assert_eq!(p.blocks.len(), 1);
    for (i, inst) in p.blocks[0].insts.iter().enumerate() {
        if inst.name() == "ds_write_b32" || inst.name() == "ds_read_b32" {
            assert!(st.guards[0][i].is_some());
        }
    }
    let (uniform, _) = super::lower::compute_uniformity(&p, &st, &res, &io.inputs);
    assert_eq!(uniform, [true]);
    let fences = super::lower::plan_lds_fences(&p, &uniform).unwrap();
    assert!(fences.leave.contains(&0));
    for mode in [Wave64Mode::Wave32, Wave64Mode::Emulate] {
        assert!(emit_cs(&prog, mode).is_ok());
    }
}

#[test]
fn lds_in_opposite_divergent_arms_cannot_share_a_fence() {
    // Model the per-lane CFG directly: a writer and a reader in sibling
    // arms have no path between them, but the host halves run concurrently.
    let prog = guarded_lds_program();
    let mut p = ir::build(&prog.patched_code(), prog.wave64, prog.vgprs).unwrap();
    let writer = p.blocks[0]
        .insts
        .iter()
        .find(|i| i.name() == "ds_write_b32")
        .unwrap()
        .clone();
    let reader = p.blocks[0]
        .insts
        .iter()
        .find(|i| i.name() == "ds_read_b32")
        .unwrap()
        .clone();
    let block = |preds, insts, exit| crate::ir::Block {
        offset: 0,
        preds,
        phis: vec![],
        insts,
        exit,
    };
    p.blocks = vec![
        block(
            vec![],
            vec![],
            crate::ir::Exit::Branch {
                taken: 1,
                not_taken: 2,
            },
        ),
        block(vec![0], vec![writer], crate::ir::Exit::Jump(3)),
        block(vec![0], vec![reader], crate::ir::Exit::Jump(3)),
        block(vec![1, 2], vec![], crate::ir::Exit::End),
    ];
    assert!(super::lower::plan_lds_fences(&p, &[true, false, false, true]).is_err());
    // If the writer returns, the reader's suffix is no longer reached by
    // the whole group; it must not get a group-sync barrier there either.
    p.blocks[1].exit = crate::ir::Exit::End;
    p.blocks[3].preds = vec![2];
    assert!(super::lower::plan_lds_fences(&p, &[true, false, false, false]).is_err());
}

#[test]
fn divergent_multiple_lds_reads_without_writes_succeed() {
    // Uniform write -> divergent branch containing two ds_read_b32s -> join.
    // Multiple pure reads in a divergent region cannot race with each other.
    let mut code = vec![
        v_mov_b32(3, inline_u(0)),
        v_lshlrev_b32(2, inline_u(2), 0),
        v_add_nc_u32(4, inline_u(7), 0),
    ];
    code.extend(ds_op(0x0D, 0, 0, 2, 4, 0)); // ds_write_b32
    code.push(v_cmp_gt_u32(inline_u(32), 0));
    code.push(s_and_saveexec_b64(6, VCC));
    let br = code.len();
    code.push(0);
    code.extend(ds_op(0x36, 0, 0, 2, 0, 3));
    code.extend(ds_op(0x36, 4, 0, 2, 0, 5));
    code.push(v_add_nc_u32(3, vgpr(5), 3));
    code[br] = s_cbranch_execz((code.len() - (br + 1)) as i16);
    code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
    emit_uav_store(&mut code, 3, 0);
    code.push(S_ENDPGM);

    let prog = make_cs_program(&code, [64, 1, 1], 512, false);
    for mode in [Wave64Mode::Native, Wave64Mode::Wave32, Wave64Mode::Emulate] {
        assert!(emit_cs(&prog, mode).is_ok(), "{mode:?} should succeed");
    }

    // <= 32 threads in Wave32 fits inside one native 32-lane wave, so even a
    // divergent write + read succeeds on Wave32 while Emulate rejects it.
    let mut small = lds_region_program(true, false, false);
    small.io = StageIo::Cs(ComputeIo {
        rsrc2: match small.io {
            StageIo::Cs(c) => c.rsrc2,
            _ => unreachable!(),
        },
        threads: [32, 1, 1],
    });
    assert!(emit_cs(&small, Wave64Mode::Wave32).is_ok());
    assert!(emit_cs(&small, Wave64Mode::Emulate).is_err());
}

#[test]
fn translate_with_handles_synthetic_wave64_container() {
    // Build genuine DXBC + XDAT(XBCH(FSSM)) Wave64 compute container with ResourceHeap param.
    let mut code = Vec::new();
    code.extend_from_slice(&v_mbcnt_lo_u32_b32(1, NEG_ONE, inline_u(0)));
    code.extend_from_slice(&v_mbcnt_hi_u32_b32(1, NEG_ONE, vgpr(1)));
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);
    let code_bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();

    const ABSENT: u64 = 0xDEAD_BEEF_DEAD_BEEF;
    const RECORD_SIZE: usize = 0x54;
    const STAGE_SIZE: usize = 0x34;
    const ENTRY_SIZE: usize = 20;

    let mut payload = 6u64.to_le_bytes().to_vec(); // GRAPHICS_SLOT_COUNT
    payload.resize(0x68, 0);
    for slot_off in [0x08usize, 0x10, 0x18, 0x20, 0x28, 0x30, 0x48, 0x60] {
        payload[slot_off..slot_off + 8].copy_from_slice(&ABSENT.to_le_bytes());
    }
    let rec = payload.len() as u64;
    payload[0x60..0x68].copy_from_slice(&rec.to_le_bytes());
    payload.resize(rec as usize + RECORD_SIZE + STAGE_SIZE, 0);

    let r = rec as usize;
    let code_at = (rec as usize + RECORD_SIZE + STAGE_SIZE) as u64;
    let layout_at = code_at + code_bytes.len() as u64;
    payload[r..r + 8].copy_from_slice(&code_at.to_le_bytes());
    payload[r + 8..r + 16].copy_from_slice(&(code_bytes.len() as u64).to_le_bytes());
    payload[r + 0x28..r + 0x30].copy_from_slice(&layout_at.to_le_bytes()); // RECORD_LAYOUT
    payload[r + 0x30..r + 0x38].copy_from_slice(&layout_at.to_le_bytes()); // RECORD_USED
    payload[r + 0x38..r + 0x3C].copy_from_slice(&1u32.to_le_bytes()); // 1 root param
    payload[r + 0x3C..r + 0x40].copy_from_slice(&0u32.to_le_bytes()); // 0 used entries
    payload[r + 0x50..r + 0x54].copy_from_slice(&(32u32 | (31 << 20)).to_le_bytes()); // RECORD_REGS

    let sd = r + RECORD_SIZE;
    payload[sd + 0x24..sd + 0x28].copy_from_slice(&(2u32 << 1).to_le_bytes()); // CS_RSRC2: 2 user SGPRs
    payload[sd + 0x28..sd + 0x2C].copy_from_slice(&64u32.to_le_bytes()); // CS_THREADS[0]
    payload[sd + 0x2C..sd + 0x30].copy_from_slice(&1u32.to_le_bytes());
    payload[sd + 0x30..sd + 0x34].copy_from_slice(&1u32.to_le_bytes());

    payload.extend_from_slice(&code_bytes);
    // RootParam: ResourceHeap (5), table_off=0, sgprs=1, base_reg=0, user_data=0, slot=0
    let mut entry = [0u8; ENTRY_SIZE];
    entry[0..4].copy_from_slice(&5u32.to_le_bytes());
    entry[8..12].copy_from_slice(&1u32.to_le_bytes());
    payload.extend_from_slice(&entry);

    let mut fssm = b"FSSM".to_vec();
    fssm.extend_from_slice(&0x101u32.to_le_bytes()); // FSSM_VERSION
    fssm.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    fssm.extend_from_slice(&payload);

    let compressed = miniz_oxide::deflate::compress_to_vec_zlib(&fssm, 6);
    let mut xbch = b"XBCH".to_vec();
    xbch.extend_from_slice(&3u32.to_le_bytes()); // XBCH_VERSION
    xbch.extend_from_slice(&(compressed.len() as u64).to_le_bytes());
    xbch.extend_from_slice(&(fssm.len() as u64).to_le_bytes());
    xbch.extend_from_slice(&compressed);

    let mut dxil_part = vec![0u8; 0x18];
    dxil_part.extend_from_slice(b"XDAT");
    dxil_part.extend_from_slice(&1u32.to_le_bytes());
    dxil_part.extend_from_slice(&0x24u32.to_le_bytes());
    let blob_off = (dxil_part.len() + 12) as u32;
    for v in [0x0Du32, blob_off, xbch.len() as u32] {
        dxil_part.extend_from_slice(&v.to_le_bytes());
    }
    dxil_part.extend_from_slice(&xbch);

    let part_off = 0x24u32;
    let total = part_off as usize + 8 + dxil_part.len();
    let mut dxbc = b"DXBC".to_vec();
    dxbc.extend_from_slice(&[0u8; 16]);
    dxbc.extend_from_slice(&1u32.to_le_bytes());
    dxbc.extend_from_slice(&(total as u32).to_le_bytes());
    dxbc.extend_from_slice(&1u32.to_le_bytes());
    dxbc.extend_from_slice(&part_off.to_le_bytes());
    dxbc.extend_from_slice(b"DXIL");
    dxbc.extend_from_slice(&(dxil_part.len() as u32).to_le_bytes());
    dxbc.extend_from_slice(&dxil_part);

    let sh_n = translate_with(
        &dxbc,
        TranslateOptions {
            wave64: Wave64Mode::Native,
        },
    )
    .unwrap();
    assert_eq!(psv_wave_range(&sh_n), (64, 64));
    let sh_w32 = translate_with(
        &dxbc,
        TranslateOptions {
            wave64: Wave64Mode::Wave32,
        },
    )
    .unwrap();
    assert_eq!(psv_wave_range(&sh_w32), (32, 32));
    let sh_emu = translate_with(
        &dxbc,
        TranslateOptions {
            wave64: Wave64Mode::Emulate,
        },
    )
    .unwrap();
    assert_eq!(psv_wave_range(&sh_emu), (0, u32::MAX));
}

#[cfg(windows)]
mod gpu {
    use std::sync::{Mutex, OnceLock};

    use super::*;
    use crate::emit::host::{Dispatch, Host, View};

    fn with_hosts(f: impl FnOnce(&[Option<Host>; 2])) {
        static HOSTS: OnceLock<Mutex<[Option<Host>; 2]>> = OnceLock::new();
        let _one = super::super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let lock = HOSTS.get_or_init(|| Mutex::new([Host::new(false).ok(), Host::new(true).ok()]));
        let guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        f(&guard);
    }

    fn run_on_hosts(
        name: &str,
        prog: &NativeProgram,
        groups: [u32; 3],
        uav_dwords: u32,
        expected: &[u32],
        expect_native_fail: bool,
    ) {
        with_hosts(|hosts| {
            for (warp_idx, host) in hosts.iter().enumerate() {
                let Some(host) = host else {
                    continue;
                };
                let warp = warp_idx == 1;
                let mut modes = vec![Wave64Mode::Emulate];
                if host.supports_wave32() {
                    modes.push(Wave64Mode::Wave32);
                }
                if expect_native_fail && !host.supports_wave64() {
                    let native = emit_cs(prog, Wave64Mode::Native).unwrap();
                    assert!(
                        host.compute_pipeline(&native).is_err(),
                        "{name}: Native Wave64 unexpectedly built a PSO on warp={warp}"
                    );
                }
                for mode in modes {
                    let shader = emit_cs(prog, mode)
                        .unwrap_or_else(|e| panic!("{name} ({mode:?}, warp={warp}): {e}"));
                    if let Ok(mut v) = super::super::validator::Validator::load(
                        super::super::validator::Validator::default_path().as_deref(),
                    ) {
                        v.validate(&mut shader.dxil.clone())
                            .unwrap_or_else(|e| panic!("{name} ({mode:?}) validate: {e}"));
                    }
                    let (root, pso) = host
                        .compute_pipeline(&shader)
                        .unwrap_or_else(|e| panic!("{name} ({mode:?}, warp={warp}) PSO: {e}"));
                    let out = host
                        .dispatch(
                            &root,
                            &pso,
                            &Dispatch {
                                roots: Vec::new(),
                                views: &[View::RawUav(uav_dwords)],
                                samplers: &[],
                                groups,
                            },
                        )
                        .unwrap_or_else(|e| panic!("{name} ({mode:?}, warp={warp}) dispatch: {e}"));
                    assert_eq!(
                        &out[0][..expected.len()],
                        expected,
                        "{name} ({mode:?}, warp={warp}) output mismatch"
                    );
                }
            }
        });
    }

    #[test]
    fn gpu_single_wave_lds_regions_are_ordered() {
        let plain: Vec<u32> = (0..64).map(|t| if t < 32 { t + 39 } else { 7 }).collect();
        let delayed: Vec<u32> = (0..64)
            .map(|t| {
                if t >= 32 {
                    return 7;
                }
                let mut x = t + 39;
                for _ in 0..4096 {
                    x = (x & 0xFF_FFFFu32).wrapping_mul(40503).wrapping_add(1);
                }
                x
            })
            .collect();
        for (name, prog, expected) in [
            (
                "delayed writer then divergent reader",
                lds_region_program(true, true, true),
                &delayed,
            ),
            (
                "divergent writer then divergent reader",
                lds_region_program(true, true, false),
                &plain,
            ),
            (
                "uniform writer then divergent reader",
                lds_region_program(false, true, false),
                &plain,
            ),
            ("same-block EXEC guards", guarded_lds_program(), &plain),
        ] {
            for _ in 0..5 {
                run_on_hosts(name, &prog, [1, 1, 1], 64, expected, true);
            }
        }
    }

    #[test]
    fn gpu_64_threads_mbcnt_ballot_and_high_ff1() {
        // v1 = mbcnt_64(-1) + bcnt1(tid > 34) * 1000 + ff1(tid > 34) * 100000
        // For tid in 0..64: active count = 29, first active bit = 35!
        let mut code = Vec::new();
        code.push(v_cmp_lt_u32(inline_u(34), 0)); // vcc = (34 < v0), bits 35..63 = 1
        code.push(s_bcnt1_i32_b64(4, VCC)); // s4 = 29
        code.push(s_ff1_i32_b64(5, VCC)); // s5 = 35
        code.extend_from_slice(&v_mbcnt_lo_u32_b32(1, NEG_ONE, inline_u(0)));
        code.extend_from_slice(&v_mbcnt_hi_u32_b32(1, NEG_ONE, vgpr(1)));
        // Pack: lane | (bcnt << 8) | (ff1 << 16)
        code.push(v_lshlrev_b32(2, inline_u(8), 4)); // wrong if vop2 takes s4 in vsrc1; put s4 in src0!
        code.pop();
        // s_mov + v_mov for shifts using SGPR in src0:
        code.push(s_mov_b32(6, LIT));
        code.push(1000);
        code.push(s_mov_b32(7, LIT));
        code.push(100_000);
        // Wait: keep integer multiply simple using shifts (s_lshl_b32) + v_or_b32!
        code.clear();
        code.push(v_cmp_lt_u32(inline_u(34), 0)); // bits 35..63 set -> bcnt=29, ff1=35
        code.push(s_bcnt1_i32_b64(4, VCC));
        code.push(s_ff1_i32_b64(5, VCC));
        code.extend_from_slice(&v_mbcnt_lo_u32_b32(1, NEG_ONE, inline_u(0)));
        code.extend_from_slice(&v_mbcnt_hi_u32_b32(1, NEG_ONE, vgpr(1)));
        // s6 = (s4 << 8) | (s5 << 16) = 29*256 + 35*65536
        code.push(sop2(4, 0x1E, 4, inline_u(8))); // s_lshl_b32 s4, s4, 8
        code.push(sop2(5, 0x1E, 5, inline_u(16))); // s_lshl_b32 s5, s5, 16
        code.push(sop2(4, 0x10, 4, 5)); // s_or_b32 s4, s4, s5
        code.push(v_or_b32(1, 4, 1));
        emit_uav_store(&mut code, 1, 0);
        code.push(S_ENDPGM);

        let prog = make_cs_program(&code, [64, 1, 1], 0, false);
        let expected: Vec<u32> = (0..64u32).map(|i| i | (29 << 8) | (35 << 16)).collect();
        run_on_hosts("mbcnt_ballot_ff1", &prog, [1, 1, 1], 64, &expected, true);
    }

    #[test]
    fn gpu_128_threads_whole_wave_active_and_prefix() {
        // Virtual wave 0 (0..63), virtual wave 1 (64..127).
        // Value = (tid & 63) + 1; mask out odd console lanes via v_cmpx / s_and_saveexec?
        // Even simpler: active lanes = (tid & 1) == 0 via v_cmp_eq_u32 + s_and_saveexec_b64!
        let mut code = vec![
            v_and_b32(2, inline_u(63), 0),   // v2 = tid & 63
            v_add_nc_u32(2, inline_u(1), 2), // v2 = (tid & 63) + 1
            v_and_b32(3, inline_u(1), 0),    // v3 = tid & 1
            v_cmp_eq_u32(inline_u(0), 3),    // vcc = even lanes
            s_and_saveexec_b64(4, VCC),      // only even lanes active! (s[4:5] = all)
            s_or_saveexec_b64(6, NEG_ONE),   // s[6:7] = saved even-lane mask
        ];
        code.extend_from_slice(&v_cndmask_b32(10, inline_u(0), vgpr(2), 6));
        for shift in [0x111u32, 0x112, 0x114, 0x118] {
            code.extend_from_slice(&dpp_row_shr(0x25, 10, 10, 10, shift, 0xF));
        }
        // Store intra-row inclusive prefix in v11 before cross-row additions:
        code.push(v_mov_b32(11, vgpr(10)));
        code.extend_from_slice(&v_readlane_b32(12, 10, inline_u(15)));
        code.extend_from_slice(&v_readlane_b32(13, 10, inline_u(31)));
        code.extend_from_slice(&v_readlane_b32(14, 10, inline_u(47)));
        code.extend_from_slice(&v_readlane_b32(15, 10, inline_u(63)));
        // Broadcast cross-row sums for active sum (v10) and prefix offsets (v4 = s12+s13, v5 = v4+s14):
        code.push(v_mov_b32(4, 12));
        code.push(v_add_nc_u32(4, 13, 4));
        code.push(v_mov_b32(5, 14));
        code.push(v_add_nc_u32(5, vgpr(4), 5)); // v5 = s12+s13+s14
        code.push(v_mov_b32(10, 15));
        code.push(v_add_nc_u32(10, vgpr(5), 10)); // v10 = full 64-lane Active sum!
        // Apply prefix offsets to rows 1, 2, 3:
        code.push(s_mov_b32(EXEC_LO, LIT));
        code.push(0xFFFF_0000);
        code.push(s_mov_b32(EXEC_HI, inline_u(0)));
        code.push(v_add_nc_u32(11, 12, 11));
        code.push(s_mov_b32(EXEC_LO, inline_u(0)));
        code.push(s_mov_b32(EXEC_HI, LIT));
        code.push(0x0000_FFFF);
        code.push(v_add_nc_u32(11, vgpr(4), 11));
        code.push(s_mov_b32(EXEC_HI, LIT));
        code.push(0xFFFF_0000);
        code.push(v_add_nc_u32(11, vgpr(5), 11));
        code.push(s_mov_b64(EXEC_LO, 6)); // restore EXEC -> end of whole-wave section!
        code.push(v_sub_nc_u32(12, vgpr(2), 11)); // Wait: v_sub_nc_u32 vdst, src0, vsrc1 = src0 - vsrc1 (Wait: gfx10 vop2 sub is vdst = src0 - vsrc1 or vsrc1 - src0? Make sure: use v11 + v12 in exclusive!)
        code.pop();
        // Safer: v_sub_nc_u32_e32 v12, v2, v11 or let's use vop3 / check lower.rs! Actually Active (v10) and Inclusive Prefix (v11) already test Shape::Active and Shape::Prefix!
        // Pack: active | (prefix_incl << 12) for active even lanes; 0xFFFF for odd lanes!
        code.push(v_lshlrev_b32(11, inline_u(12), 11));
        code.push(v_or_b32(1, vgpr(10), 11));
        code.push(s_mov_b64(EXEC_LO, 4)); // restore all lanes
        // Write 1 for inactive odd lanes via v_cndmask: actually pre-init v1 = 0 before branch, keep simple: only even lanes store, odd stay 0 in UAV!
        // Wait: just write all active lanes -> odd lanes remain 0 in zeroed RawUav!
        code.pop(); // don't restore yet so odd lanes don't overwrite with garbage
        emit_uav_store(&mut code, 1, 0);
        code.push(s_mov_b64(EXEC_LO, 4));
        code.push(S_ENDPGM);

        let prog = make_cs_program(&code, [128, 1, 1], 0, false);
        let even_sum_64: u32 = (0..64u32).filter(|l| l % 2 == 0).map(|l| l + 1).sum();
        let mut expected = Vec::with_capacity(128);
        for g in 0..128u32 {
            if g % 2 != 0 {
                expected.push(0);
            } else {
                let l = g & 63;
                let pref: u32 = (0..=l).filter(|k| k % 2 == 0).map(|k| k + 1).sum();
                expected.push(even_sum_64 | (pref << 12));
            }
        }
        run_on_hosts(
            "whole_wave_active_and_prefix",
            &prog,
            [1, 1, 1],
            128,
            &expected,
            true,
        );
    }

    #[test]
    fn gpu_96_threads_partial_virtual_wave() {
        // 96 threads: vwave 0 = 64 threads, vwave 1 = 32 threads (Half 0 only, NO Half 1!).
        // Tests both 64-lane ballot + Shape::Active across partial wave.
        let mut code = vec![
            v_and_b32(2, inline_u(63), 0),
            v_add_nc_u32(2, inline_u(1), 2), // 1..=64 in w0, 1..=32 in w1
            v_cmp_lt_u32(inline_u(15), 2),   // count lanes > 15: 49 in w0, 17 in w1
            s_bcnt1_i32_b64(4, VCC),
            s_or_saveexec_b64(6, NEG_ONE),
        ];
        code.extend_from_slice(&v_cndmask_b32(10, inline_u(0), vgpr(2), 6));
        for shift in [0x111u32, 0x112, 0x114, 0x118] {
            code.extend_from_slice(&dpp_row_shr(0x25, 10, 10, 10, shift, 0xF));
        }
        code.extend_from_slice(&v_readlane_b32(12, 10, inline_u(15)));
        code.extend_from_slice(&v_readlane_b32(13, 10, inline_u(31)));
        code.extend_from_slice(&v_readlane_b32(14, 10, inline_u(47)));
        code.extend_from_slice(&v_readlane_b32(15, 10, inline_u(63)));
        code.push(v_mov_b32(10, 12));
        code.push(v_add_nc_u32(10, 13, 10));
        code.push(v_add_nc_u32(10, 14, 10));
        code.push(v_add_nc_u32(10, 15, 10));
        code.push(s_mov_b64(EXEC_LO, 6));

        code.push(sop2(4, 0x1E, 4, inline_u(16))); // s_lshl_b32 s4, s4, 16 (bcnt << 16)
        code.push(v_or_b32(1, 4, 10));
        emit_uav_store(&mut code, 1, 0);
        code.push(S_ENDPGM);

        let prog = make_cs_program(&code, [96, 1, 1], 0, false);
        let mut expected = Vec::with_capacity(96);
        let sum0: u32 = (1..=64).sum();
        let sum1: u32 = (1..=32).sum();
        for g in 0..96u32 {
            if g < 64 {
                expected.push(sum0 | (49 << 16));
            } else {
                expected.push(sum1 | (17 << 16));
            }
        }
        run_on_hosts(
            "partial_virtual_wave_96",
            &prog,
            [1, 1, 1],
            96,
            &expected,
            true,
        );
    }

    #[test]
    fn gpu_128_threads_readlane_writelane_spills_and_firstlane() {
        // Tests:
        // 1. Dynamic cross-half v_readlane_b32 reading partner = (tid ^ 33) & 63
        // 2. Non-uniform v_readfirstlane_b32 picking lane 0 (0 in w0, 64 in w1)
        // 3. Constant-lane SGPR spill via v_writelane_b32(_, _, 35) -> v_readlane_b32(_, 35) even inside divergent if!
        let mut code = Vec::new();
        // Spill s12 = 77 into v3 lane 35; keep in v1 when tid < 48 (divergent branch!)
        code.push(s_mov_b32(12, LIT));
        code.push(77);
        code.extend_from_slice(&v_writelane_b32(3, 12, inline_u(35)));
        code.push(v_mov_b32(4, inline_u(0)));
        code.push(v_cmp_gt_u32(inline_u(48), 0)); // 48 > v0 i.e. v0 < 48
        code.push(s_and_saveexec_b64(6, VCC));
        let br_idx = code.len();
        code.push(0);
        code.extend_from_slice(&v_readlane_b32(13, 3, inline_u(35)));
        code.push(v_mov_b32(4, 13));
        let after = code.len();
        code[br_idx] = s_cbranch_execz((after - (br_idx + 1)) as i16);
        code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));

        // Non-uniform v_readfirstlane_b32 of tid (`v0`): 0 for w0, 64 for w1!
        code.push(v_readfirstlane_b32(14, 0));

        // Dynamic v_readlane_b32 reading (tid ^ 33) & 63 across halves:
        code.push(v_mov_b32(5, vgpr(0)));
        code.push(vop2(0x1D, 5, inline_u(33), 5)); // v_xor_b32 v5, 33, v5
        code.push(v_and_b32(5, inline_u(63), 5));
        code.extend_from_slice(&v_readlane_b32(15, 0, vgpr(5)));

        // Result = spill (4) + readfirstlane (s14 << 8) + dynamic_readlane (s15 << 16)
        code.push(sop2(14, 0x1E, 14, inline_u(8)));
        code.push(v_add_nc_u32(1, 14, 4));
        code.push(sop2(15, 0x1E, 15, inline_u(16)));
        code.push(v_add_nc_u32(1, 15, 1));
        emit_uav_store(&mut code, 1, 0);
        code.push(S_ENDPGM);

        let prog = make_cs_program(&code, [128, 1, 1], 0, false);
        let expected: Vec<u32> = (0..128u32)
            .map(|g| {
                let wbase = g & !63;
                let spill = if g < 48 { 77 } else { 0 };
                let first = wbase;
                let partner = wbase | ((g ^ 33) & 63);
                spill + (first << 8) + (partner << 16)
            })
            .collect();
        run_on_hosts(
            "readlane_spills_firstlane",
            &prog,
            [1, 1, 1],
            128,
            &expected,
            true,
        );
    }

    #[test]
    fn gpu_128_threads_ds_swizzle_and_dpp_row_mask() {
        // 1. ds_swizzle_b32 XOR 2 within 32-lane halves (offset = 0x1F | (2 << 10) = 0x81F)
        // 2. v_add_nc_u32_dpp row_shr:1 bound_ctrl:1 with row_mask = 0xA (rows 1 & 3 enabled, rows 0 & 2 masked!)
        let mut code = Vec::new();
        code.extend_from_slice(&ds_op(0x35, 0x1F, 0x08, 0, 0, 2)); // v2 = swizzle(v0, xor 2)
        code.push(v_mov_b32(3, inline_u(1))); // fallback 1 on masked rows 0 and 2!
        // src1=v0, src0=v0 (dpp row_shr:1 -> prev_tid), row_mask=0xA (bits 1 and 3!), bank_mask=0xF, bound_ctrl=1
        code.extend_from_slice(&dpp_row_shr(0x25, 3, 0, 0, 0x111, 0xA));
        code.push(v_lshlrev_b32(3, inline_u(16), 3));
        code.push(v_or_b32(1, vgpr(2), 3));
        emit_uav_store(&mut code, 1, 0);
        code.push(S_ENDPGM);

        let prog = make_cs_program(&code, [128, 1, 1], 0, false);
        let expected: Vec<u32> = (0..128u32)
            .map(|g| {
                let swz = g ^ 2;
                let row = (g & 63) >> 4;
                let dpp = if row == 1 || row == 3 {
                    if (g & 15) == 0 { g } else { g + (g - 1) }
                } else {
                    1
                };
                swz | (dpp << 16)
            })
            .collect();
        run_on_hosts(
            "ds_swizzle_dpp_rowmask",
            &prog,
            [1, 1, 1],
            128,
            &expected,
            true,
        );
    }

    #[test]
    fn gpu_64_threads_lockstep_lds_without_s_barrier() {
        // 64 threads sharing LDS without `s_barrier` (assumes single Wave64 wave on console).
        let mut code = Vec::new();
        code.push(v_lshlrev_b32(2, inline_u(2), 0)); // write addr = tid * 4
        code.push(v_add_nc_u32(3, inline_u(7), 0)); // val = tid + 7
        code.extend_from_slice(&ds_op(0x0D, 0, 0, 2, 3, 0)); // ds_write_b32 addr=v2, data=v3
        code.push(vop2(0x1D, 4, inline_u(32), 0)); // partner = tid ^ 32 (crosses Wave32 half!)
        code.push(v_lshlrev_b32(4, inline_u(2), 4)); // read addr = partner * 4
        code.extend_from_slice(&ds_op(0x36, 0, 0, 4, 0, 1)); // ds_read_b32 v1, addr=v4
        emit_uav_store(&mut code, 1, 0);
        code.push(S_ENDPGM);

        let prog = make_cs_program(&code, [64, 1, 1], 512, false);
        let expected: Vec<u32> = (0..64u32).map(|i| (i ^ 32) + 7).collect();
        run_on_hosts(
            "lockstep_lds_no_barrier",
            &prog,
            [1, 1, 1],
            64,
            &expected,
            false,
        );
    }

    #[test]
    fn gpu_128_threads_uniform_loop_cross_half_ballot() {
        // Uniform scalar loop for s10 in 0..4 performing 64-lane ballot every iteration.
        let mut code = Vec::new();
        code.push(s_mov_b32(10, inline_u(0))); // loop counter
        code.push(v_mov_b32(1, inline_u(0))); // accumulator
        let loop_top = code.len();
        code.push(v_and_b32(2, inline_u(63), 0));
        code.push(v_add_nc_u32(2, 10, 2)); // (tid & 63) + iter
        code.push(v_cmp_lt_u32(inline_u(40), 2)); // count increases by 1 each iteration!
        code.push(s_bcnt1_i32_b64(4, VCC));
        code.push(v_add_nc_u32(1, 4, 1));
        code.push(s_add_i32(10, 10, inline_u(1)));
        code.push(s_cmp_lt_u32(10, inline_u(4)));
        let br = code.len();
        code.push(s_cbranch_scc1(
            (loop_top as isize - (br + 1) as isize) as i16,
        ));
        emit_uav_store(&mut code, 1, 0);
        code.push(S_ENDPGM);

        let prog = make_cs_program(&code, [128, 1, 1], 0, false);
        // iter 0: >40 in 0..=63 => 23; iter 1: 24; iter 2: 25; iter 3: 26 => sum = 98
        let expected = vec![23 + 24 + 25 + 26; 128];
        run_on_hosts(
            "uniform_loop_ballots",
            &prog,
            [1, 1, 1],
            128,
            &expected,
            true,
        );
    }

    #[test]
    fn gpu_32_threads_wave32_fast_path_with_high_lane_spill() {
        // 32-thread Wave64 compute shader: Wave32 uses single wave + zero LDS scratch,
        // while still resolving constant-lane SGPR spill at lane 45 (>= 32).
        let mut code = Vec::new();
        code.push(s_mov_b32(12, LIT));
        code.push(123);
        code.extend_from_slice(&v_writelane_b32(3, 12, inline_u(45)));
        code.extend_from_slice(&v_readlane_b32(13, 3, inline_u(45)));
        code.push(v_add_nc_u32(1, 13, 0));
        emit_uav_store(&mut code, 1, 0);
        code.push(S_ENDPGM);

        let prog = make_cs_program(&code, [32, 1, 1], 0, false);
        let expected: Vec<u32> = (0..32u32).map(|i| 123 + i).collect();
        run_on_hosts(
            "wave32_32_threads_spill",
            &prog,
            [1, 1, 1],
            32,
            &expected,
            true,
        );
    }

    #[test]
    fn gpu_128_threads_cross_block_ballot_and_constant_mask_guard() {
        // Block 0 defines `vcc` (`tid & 63 >= 40`), branches uniformly via s_branch (block split)
        // to block 1 where `s_bcnt1` consumes cross-block `vcc`, plus `Atom::Constant` branch mask (`-16` -> lanes 4..63).
        let mut code = vec![
            v_and_b32(2, inline_u(63), 0),
            v_cmp_lt_u32(inline_u(39), 2), // vcc = (tid & 63) >= 40 (24 bits)
            sopp(0x02, 0),                 // s_branch +0 -> block boundary!
            // Block 1:
            s_bcnt1_i32_b64(4, VCC),    // s4 = 24
            s_and_saveexec_b64(8, 208), // 208 = inline -16 (bits 4..63 active in Wave64)
        ];
        let br_idx = code.len();
        code.push(0); // s_cbranch_execz -> after
        code.push(v_add_nc_u32(1, 4, 0));
        emit_uav_store(&mut code, 1, 0);
        let after = code.len();
        code[br_idx] = s_cbranch_execz((after - (br_idx + 1)) as i16);
        code.push(s_or_b64(EXEC_LO, EXEC_LO, 8));
        code.push(S_ENDPGM);

        let prog = make_cs_program(&code, [128, 1, 1], 0, false);
        let expected: Vec<u32> = (0..128u32)
            .map(|g| if (g & 63) >= 4 { 24 + g } else { 0 })
            .collect();
        run_on_hosts(
            "cross_block_and_const_mask",
            &prog,
            [1, 1, 1],
            128,
            &expected,
            true,
        );
    }

    #[test]
    fn gpu_64_threads_empty_exec_readfirstlane_returns_lane0() {
        // GFX10 v_readfirstlane_b32 with EXEC == 0 returns lane 0's value (`v2[0] = 7`).
        let mut code = vec![
            v_add_nc_u32(2, inline_u(7), 0), // lane 0 = 7, lane 32 = 39
            s_mov_b64(6, EXEC_LO),
            s_mov_b64(EXEC_LO, inline_u(0)),
            v_readfirstlane_b32(4, 2),
            s_mov_b64(EXEC_LO, 6),
            v_mov_b32(1, 4),
        ];
        emit_uav_store(&mut code, 1, 0);
        code.push(S_ENDPGM);

        let prog = make_cs_program(&code, [64, 1, 1], 0, false);
        let expected = vec![7u32; 64];
        run_on_hosts(
            "empty_exec_readfirstlane",
            &prog,
            [1, 1, 1],
            64,
            &expected,
            false,
        );
    }
}
