//! End-to-end tests of waterfall loops over per-lane descriptors: the
//! synthetic GFX10 programs of `resource::tests` lowered to DXIL, validated
//! by `dxil.dll` when present, and run on the host GPU and WARP, each lane
//! reading the resource its own descriptor names.

use windows::Win32::Graphics::Direct3D12::*;

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

#[test]
fn dword_loads_cross_character_records_in_raw_and_structured_views() {
    for record_bytes in [100, 1580] {
        check_cross_record_reads(record_bytes);
    }
}

fn check_cross_record_reads(record_bytes: u32) {
    let last = record_bytes - 4;
    let second_last = 2 * record_bytes - 4;
    let words: &[u32] = &[
        0xbe84_0300, 0xbe85_0380, // input table at s[4:5]
        0xbe86_0301, 0xbe87_0380, // output table at s[6:7]
        0xf408_0502, 0xfa00_0000, // input descriptor in s[20:23]
        0xf408_0a03, 0xfa00_0000, // output descriptor in s[40:43]
        0xbf8c_c07f,
        0xf428_060a, 0xfa00_0000 | last, // four dwords spanning the first boundary
        0xbf8c_c07f,
        0x7e08_0218, 0x7e0a_0219, 0x7e0c_021a, 0x7e0e_021b,
        0xe078_0000, 0x800a_0400, // buffer_store_dwordx4 v[4:7]
        0x7e04_02ff, second_last, // v2 = final dword of record 1
        0xe034_1000, 0x8005_0802, // buffer_load_dwordx2 v[8:9], v2, s[20:23], 0 offen
        0xbf8c_3f70,
        0xe074_0010, 0x800a_0800, // buffer_store_dwordx2 v[8:9], offset:16
        0xbf81_0000,
    ];
    let param = |slot| RootParam {
        root: RootType::Table, slot, table_offset_dwords: 0, sgprs: 1, base_register: 0, user_data: slot,
    };
    let used = |slot, kind, stride| UsedResource {
        kind, root: RootType::Table, slot, user_data: slot, stride, dwords: 4, lower_bound: 0, table_offset_dwords: 0,
    };
    let native = NativeProgram {
        stage: HwStage::Cs, wave64: false, vgprs: 10,
        layout: vec![param(0), param(1)],
        used: vec![used(0, ResourceType::Srv, record_bytes), used(1, ResourceType::Uav, 0)],
        io: StageIo::Cs(ComputeIo { rsrc2: 2 << 1, threads: [1, 1, 1] }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let shader = host_shader(lower_native(&native));
    assert!(shader.user_data.iter().any(|u| u.root == RootType::Other(super::USER_TABLE_STRUCTURE)));
    let record_words = record_bytes / 4;
    let data: Vec<u32> = (0..3 * record_words).map(|i| 0x5555_0000 + i).collect();
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        // Both record sizes put the structured view 8 bytes past a 16-byte boundary.
        let padded = [vec![0xdead_beef; 2 * record_words as usize], data.clone()].concat();
        for (stride, actual, input) in [
            (Stride::Raw, 0, &data),
            (Stride::Structured { bytes: record_bytes, first: 2 }, record_bytes, &padded),
        ] {
            let mut constants = vec![0; shader.user_data_dwords() as usize];
            for u in &shader.user_data {
                constants[u.index as usize] = match u.root {
                    RootType::Table => 32 * u32::from(u.slot),
                    RootType::Other(super::USER_TABLE_STRUCTURE) => actual,
                    _ => panic!("unexpected user data {u:?}"),
                };
            }
            let out = host.dispatch(&root, &pso, &Dispatch {
                roots: vec![(0, Root::Constants(constants))],
                views: &[View::Buffer { data: input, stride }, View::RawUav(6)],
                samplers: &[], groups: [1; 3],
            }).expect("dispatch");
            let first = (last / 4) as usize;
            let second = (second_last / 4) as usize;
            assert_eq!(out[0][..4], data[first..first + 4], "{label}, stride {actual}");
            assert_eq!(out[0][4..], data[second..second + 2], "{label}, stride {actual}");
        }
    }
}

#[test]
fn heap_texture_dimensions_are_available_without_sampling() {
    let words: &[u32] = &[
        0xbe84_0300, 0xbe85_0380, // resource heap at s[4:5]
        0xbe86_0301, 0xbe87_0380, // output table at s[6:7]
        0xf40c_0502, 0xfa00_0020, // s_load_dwordx8 s[20:27], s[4:5], 32
        0xf408_0a03, 0xfa00_0000, // s_load_dwordx4 s[40:43], s[6:7], 0
        0xbf8c_c07f,
        0x900c_9e15, // s_lshr_b32 s12, s21, 30
        0x8f14_8216, // s_lshl_b32 s20, s22, 2
        0x938f_ff16, 0x0010_000e, // height minus one from s22
        0x8814_140c, 0x8714_ff14, 0x0000_ffff, // width minus one
        0x938c_ff17, 0x0004_000c, // first mip from s23
        0x7e08_0214, 0x7e0a_020f, 0x7e0c_020c, // v4..6 = width-1, height-1, mip
        0x7e0e_0280, // v_mov_b32 v7, 0
        0xe078_1000, 0x800a_0400, // buffer_store_dwordx4 v[4:7], v0, s[40:43], 0 offen
        0xbf81_0000,
    ];
    let param = |root, slot| RootParam {
        root, slot, table_offset_dwords: 0, sgprs: 1, base_register: 0, user_data: slot,
    };
    let native = NativeProgram {
        stage: HwStage::Cs, wave64: false, vgprs: 8,
        layout: vec![param(RootType::ResourceHeap, 0), param(RootType::Table, 1)],
        used: vec![UsedResource {
            kind: ResourceType::Uav, root: RootType::Table, table_offset_dwords: 0,
            lower_bound: 0, stride: 0, dwords: 4, user_data: 1, slot: 1,
        }],
        io: StageIo::Cs(ComputeIo { rsrc2: 2 << 1, threads: [1, 1, 1] }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let shader = host_shader(lower_native(&native));
    assert_eq!(shader.root_srvs, [super::SLOT_STRIDES]);
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else { continue };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        let mut fields = vec![0; super::SLOT_STRIDES_COUNT as usize + 2];
        // Odd dimensions with nonzero high and low bits, and a nonzero mip.
        fields[super::SLOT_STRIDES_COUNT as usize + 1] = 12346 | (9122 << 14) | (3 << 28);
        let out = host.dispatch(&root, &pso, &Dispatch {
            roots: vec![(0, Root::Constants(vec![0; shader.user_data_dwords() as usize])),
                        (1, Root::View { srv: true, data: fields })],
            views: &[View::RawUav(4)], samplers: &[], groups: [1; 3],
        }).expect("dispatch");
        assert_eq!(out[0][..3], [12346, 9122, 3], "{label}");
    }
}

/// Lowers a wave32 compute program of 32 threads with the table of views at
/// s0 (textures, or with `buffers` structured buffers of dwords, at entries
/// 0 and 1, a raw UAV at entry 2) and the table of samplers at s1.
fn build(words: &[u32], buffers: bool) -> HostShader {
    let table = |user_data: u8| RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: user_data,
    };
    let range = |kind, table_offset_dwords, dwords, stride, user_data| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords,
        user_data,
        slot: user_data,
    };
    let views = if buffers {
        range(ResourceType::Srv, 0, 4, 4, 0)
    } else {
        range(ResourceType::Srv, 0, 8, 0, 0)
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![table(0), table(1)],
        used: vec![
            views,
            range(ResourceType::Uav, 16, 4, 0, 0),
            range(ResourceType::Sampler, 0, 4, 0, 1),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let lowered = lower_native(&native);
    // One access with each lane's own heap index, no wave operation.
    assert_eq!(lowered.nonuniform_handles, 1);
    assert!(!lowered.wave_ops);
    host_shader(lowered)
}

fn lower_native(native: &NativeProgram) -> lower::Lowered {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, native).expect("io::raise");
    lower::lower(&lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower")
}

fn host_shader(lowered: lower::Lowered) -> HostShader {
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: Vec::new(),
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

/// Runs `shader` with `views` at heap indices 0.. (the UAV last) on the
/// hardware GPU and on WARP (each skipped without a device) and returns,
/// per one that ran, the UAV's 32 dwords.
fn run(shader: &HostShader, views: &[View]) -> Vec<(&'static str, Vec<u32>)> {
    let samplers = [D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        MaxLOD: f32::MAX,
        ..Default::default()
    }];
    let mut out = Vec::new();
    // One GPU test at a time, with the VALU tests: in parallel on the
    // hardware adapter and WARP the test process sometimes died of an
    // access violation (7 of 24 runs of these four tests).
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = match host.compute_pipeline(shader) {
            Ok(p) => p,
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        // Both tables start at heap index 0.
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            assert_eq!(u.root, RootType::Table, "{u:?}");
            ud[u.index as usize] = 0;
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views,
                    samplers: &samplers,
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, words[0].clone()));
    }
    out
}

/// The red channel each lane sampled from two 1x1 textures, red 1.0 at
/// heap index 0 and 2.0 at 1.
fn sampled(shader: &HostShader) -> Vec<(&'static str, Vec<f32>)> {
    let first = [[1.0f32, 0.0, 0.0, 1.0]];
    let second = [[2.0f32, 0.0, 0.0, 1.0]];
    let views = [
        View::Texture(1, 1, &first),
        View::Texture(1, 1, &second),
        View::RawUav(32),
    ];
    run(shader, &views)
        .into_iter()
        .map(|(label, words)| (label, words.into_iter().map(f32::from_bits).collect()))
        .collect()
}

#[test]
fn lanes_sample_the_texture_their_if_else_arm_chose() {
    use crate::resource::tests::WATERFALL_JOINED_WAVE32;
    let shader = build(WATERFALL_JOINED_WAVE32, false);
    for (label, red) in sampled(&shader) {
        // Lanes below 16 took the first texture, the others the second.
        let want: Vec<f32> = (0..32).map(|l| if l < 16 { 1.0 } else { 2.0 }).collect();
        assert_eq!(red, want, "{label}");
    }
}

#[test]
fn lanes_sample_the_texture_their_arm_loaded_or_picked() {
    use crate::resource::tests::WATERFALL_JOINED_MIXED_WAVE32;
    let shader = build(WATERFALL_JOINED_MIXED_WAVE32, false);
    for (label, red) in sampled(&shader) {
        // Lanes below 16 loaded entry (lane & 1); of the others, lanes
        // below 24 picked the first, the rest the second.
        let want: Vec<f32> = (0..32)
            .map(|l| match l {
                0..16 => [1.0, 2.0][l & 1],
                16..24 => 1.0,
                _ => 2.0,
            })
            .collect();
        assert_eq!(red, want, "{label}");
    }
}

#[test]
fn lanes_load_from_the_buffer_their_nested_arms_chose() {
    use crate::resource::tests::WATERFALL_JOINED_BUFFER_WAVE32;
    let shader = build(WATERFALL_JOINED_BUFFER_WAVE32, true);
    let first: Vec<u32> = (0..32).map(|i| 100 + i).collect();
    let second: Vec<u32> = (0..32).map(|i| 200 + i).collect();
    let views = [
        View::Buffer {
            data: &first,
            stride: Stride::Structured { bytes: 4, first: 0 },
        },
        View::Buffer {
            data: &second,
            stride: Stride::Structured { bytes: 4, first: 0 },
        },
        View::RawUav(32),
    ];
    for (label, got) in run(&shader, &views) {
        // Dword `lane` of the first buffer below 16, of the second from 16
        // to 23; lanes from 24 on load and store nothing.
        let want: Vec<u32> = (0..32)
            .map(|l| match l {
                0..16 => 100 + l,
                16..24 => 200 + l,
                _ => 0,
            })
            .collect();
        assert_eq!(got, want, "{label}");
    }
}

#[test]
fn lanes_sample_the_entry_their_offset_names_in_the_console_waterfall() {
    use crate::resource::tests::WATERFALL_INDEXED_WAVE32;
    let shader = build(WATERFALL_INDEXED_WAVE32, false);
    let results = sampled(&shader);
    assert!(!results.is_empty());
    for (label, red) in results {
        // Even lanes hold entry 0's offset, odd lanes entry 1's.
        let want: Vec<f32> = (0..32).map(|l| [1.0, 2.0][l & 1]).collect();
        assert_eq!(red, want, "{label}");
    }
}

#[test]
fn lanes_sample_their_own_texture_in_a_waterfall_narrowed_by_v_cmpx() {
    use crate::resource::tests::cmpx_waterfall;
    // Lanes below 16 hold the first descriptor, the others the second.
    let want: Vec<f32> = (0..32).map(|l| if l < 16 { 1.0 } else { 2.0 }).collect();
    for folded in [false, true] {
        let shader = build(&cmpx_waterfall(false, folded), false);
        let results = sampled(&shader);
        assert!(!results.is_empty());
        for (label, red) in results {
            assert_eq!(red, want, "{label}, folded {folded}");
        }
    }
}
