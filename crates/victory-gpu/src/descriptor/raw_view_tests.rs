//! The raw views a host view heap keeps for the runtime stride table
//! ([`victory_shader::emit::StrideEntry`]), read by a translated program:
//! the heap's own slots, entries and raw views, run on the host GPU.

use victory_shader::emit::{HostShader, STRIDE_SPACE, StrideEntry, USER_DATA_SPACE};
use victory_shader::native::{ComputeIo, StageIo};
use victory_shader::{HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource};
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use super::View;
use super::host::HostHeap;
use crate::host::host;

const R32_TYPELESS: u32 = 39;
const R32G32B32A32_FLOAT: u32 = 2;
/// `D3D12_BUFFER_SRV_FLAG_RAW` and `D3D12_BUFFER_UAV_FLAG_RAW`.
const RAW: u32 = 1;
/// Bytes of each title buffer below.
const WIDTH: u32 = 0x1000;
/// Title memory outside buffer objects is copied in windows of this size.
const WINDOW: u64 = 4 << 20;

/// A buffer view of memory the title placed itself: `count` elements from
/// element `first` of the buffer at `address`, in `format` (0 with a
/// `stride` for a structured view, typeless with [`RAW`] for a raw one).
fn placed(
    unordered: bool,
    address: u64,
    (format, stride, flags): (u32, u32, u32),
    (first, count): (u32, u32),
) -> View {
    // The resource description: a buffer of WIDTH bytes.
    let mut resource = [0; 14];
    (resource[0], resource[4]) = (1, WIDTH);
    let mut desc = [0; 16];
    // Format, then the buffer dimension (1 for both kinds of view).
    (desc[0], desc[1]) = (format, 1);
    if unordered {
        // First element (64 bits), count, stride, counter offset, flags.
        (desc[2], desc[4], desc[5], desc[8]) = (first, count, stride, flags);
    } else {
        // The default component mapping, then first element (64 bits),
        // count, stride, flags.
        desc[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        (desc[4], desc[6], desc[7], desc[8]) = (first, count, stride, flags);
    }
    (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
    View::Placed(unordered, resource, desc)
}

/// The compute program of `victory-shader`'s
/// `unknown_stride_views_chosen_by_a_loaded_heap_index`: each of two thread
/// groups loads its view's heap slot from the buffer in table slot 0, and
/// its four lanes read `[lane, 4]` of that view and store it at
/// `[lane, 16 * group]` of the view in table slot 1, neither with a stride
/// known at translation.
fn program() -> HostShader {
    let words: &[u32] = &[
        0xBE94_0304, // s_mov_b32 s20, s4
        0xBE95_0380, // s_mov_b32 s21, 0
        0xF408_040A,
        0xFA00_0000, // s_load_dwordx4 s[16:19], s[20:21], 0x0
        0x8F07_8205, // s_lshl_b32 s7, s5, 2
        0xF420_0188,
        0x0E00_0000, // s_buffer_load_dword s6, s[16:19], s7
        0x8F06_8506, // s_lshl_b32 s6, s6, 5
        0xF408_020A,
        0x0C00_0000, // s_load_dwordx4 s[8:11], s[20:21], s6
        0xF408_030A,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[20:21], 0x20
        0x7E02_0284, // v_mov_b32 v1, 4
        0xE030_3000,
        0x8002_0200, // buffer_load_dword v2, v[0:1], s[8:11], 0 idxen offen
        0x8F07_8405, // s_lshl_b32 s7, s5, 4
        0x7E02_0207, // v_mov_b32 v1, s7
        0xE070_3000,
        0x8003_0200, // buffer_store_dword v2, v[0:1], s[12:15], 0 idxen offen
        0xBF81_0000, // s_endpgm
    ];
    let used = |kind, table_offset_dwords| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    let param = |root, sgprs, user_data, slot| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![
            param(RootType::Constants, 4, 0, 0),
            param(RootType::Table, 1, 4, 1),
        ],
        used: vec![used(ResourceType::Srv, 0), used(ResourceType::Uav, 8)],
        // Four threads (v0: the thread's id), the group's id in s5 after
        // the user SGPRs s0..s4.
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1 | 1 << 7,
            threads: [4, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = victory_shader::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = victory_shader::structure::structure(&p).unwrap();
    let res = victory_shader::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = victory_shader::io::raise(&p, &native).unwrap();
    // An empty container (no signature parts): the tag, a digest, version
    // 1, its size, no parts.
    let mut container = b"DXBC".to_vec();
    container.extend([0; 16]);
    container.extend([1u32, 32, 0].into_iter().flat_map(u32::to_le_bytes));
    let c = victory_shader::Container::parse(&container).unwrap();
    let cs = victory_shader::emit::emit(&c, &native, &p, &st, &io, &res).unwrap();
    assert!(cs.stride_table);
    cs
}

/// The root signature and pipeline of `cs`: its user data as root constants
/// (parameter 0), the runtime stride table as a root view (parameter 1),
/// the heaps indexed directly, as `pipeline::host` makes them.
fn pipeline(
    device: &ID3D12Device,
    cs: &HostShader,
) -> windows::core::Result<(ID3D12RootSignature, ID3D12PipelineState)> {
    let params = [
        D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: USER_DATA_SPACE,
                    Num32BitValues: cs.user_data_dwords(),
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                    ShaderRegister: 0,
                    RegisterSpace: STRIDE_SPACE,
                    Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
    ];
    let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: 0,
                pStaticSamplers: std::ptr::null(),
                Flags: D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
                    | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED,
            },
        },
    };
    // SAFETY: the descriptions point at live data; the blob is read before
    // it is released.
    unsafe {
        let mut blob: Option<ID3DBlob> = None;
        D3D12SerializeVersionedRootSignature(&desc, &mut blob, None)?;
        let blob = blob.ok_or_else(windows::core::Error::empty)?;
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: cs.dxil.as_ptr().cast(),
                BytecodeLength: cs.dxil.len(),
            },
            ..Default::default()
        })?;
        Ok((root, pso))
    }
}

/// Runs two thread groups of the pipeline over `heap` (table at slot 0)
/// and waits for them.
fn dispatch(
    device: &ID3D12Device,
    heap: &HostHeap,
    root: &ID3D12RootSignature,
    pso: &ID3D12PipelineState,
    user_data: u32,
) -> windows::core::Result<()> {
    let zeros = vec![0u32; user_data as usize];
    // SAFETY: an open list with every root argument set; the heap, its
    // table and the buffers its views read outlive the wait.
    unsafe {
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })?;
        let alloc: ID3D12CommandAllocator =
            device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
        let list: ID3D12GraphicsCommandList =
            device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, pso)?;
        list.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
        list.SetComputeRootSignature(root);
        list.SetComputeRoot32BitConstants(0, user_data, zeros.as_ptr().cast(), 0);
        list.SetComputeRootShaderResourceView(1, heap.stride_table_gpu().expect("a table"));
        list.Dispatch(2, 1, 1);
        list.Close()?;
        queue.ExecuteCommandLists(&[Some(list.cast()?)]);
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        queue.Signal(&fence, 1)?;
        while fence.GetCompletedValue() < 1 {
            device.GetDeviceRemovedReason()?;
            std::thread::yield_now();
        }
    }
    Ok(())
}

/// The host heap gives every structured and typed buffer view a raw view
/// of the same bytes and names it in the slot's runtime stride table
/// entry, so a program that reads a buffer without knowing its stride, in a
/// slot it picks at run time, reads the bytes of whatever view the title
/// put there, and zeros past that view's last element. Rebinding a slot
/// rewrites its entry and raw view.
#[test]
fn unknown_stride_reads_reach_the_bytes_of_the_view_in_the_slot() {
    let Some(heap) = HostHeap::new(false, 8) else {
        return; // no host GPU
    };
    let host = host().expect("the heap's host");
    let device = &host.device;
    let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
        HighestShaderModel: D3D_SHADER_MODEL_6_6,
    };
    // SAFETY: a feature query into a struct of its size.
    let queried = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_SHADER_MODEL,
            (&raw mut model).cast(),
            size_of_val(&model) as u32,
        )
    };
    if queried.is_err() || model.HighestShaderModel.0 < D3D_SHADER_MODEL_6_6.0 {
        return; // translated shaders need SM 6.6
    }
    // Title memory in two windows the host copies apart: what the program
    // reads in the first, what it writes in the second. (Nothing here
    // brings the copies up to date; the test fills them itself.)
    let memory = vec![0u8; 2 * WINDOW as usize + 0x1_0000];
    let read = (memory.as_ptr() as u64).next_multiple_of(WINDOW);
    let written = read + WINDOW;
    // (The host makes a structured view only where its elements fall on
    // its copy's: the data starts a multiple of 12, 16 and 20 bytes in.)
    let (slots, data) = (read, read + 0xF000);
    let copy = |address: u64| -> *mut u32 {
        let shadow = crate::shadow::find(address, u64::from(WIDTH)).expect("a host copy");
        let mut p = std::ptr::null_mut();
        // SAFETY: the copy is CPU-visible; it stays mapped while it lives.
        unsafe { shadow.buffer.Map(0, None, Some(&mut p)) }.expect("a mapped copy");
        // SAFETY: the address lies in the copy.
        unsafe { p.cast::<u8>().add(shadow.offset(address) as usize).cast() }
    };
    let (slots_copy, data_copy, out) = (copy(slots), copy(data), copy(written));
    // SAFETY: WIDTH bytes of each copy.
    unsafe {
        for i in 0..WIDTH as usize / 4 {
            data_copy.add(i).write(100 + i as u32);
        }
    }

    // Slot 0: the heap slots the two groups read, a raw view. Slot 1: what
    // they write, elements of 4 bytes from element 5 (20 bytes in).
    let structured = |stride| (0, stride, 0);
    let raw = (R32_TYPELESS, 0, RAW);
    heap.make_at(0, &placed(false, slots, raw, (0, 16)));
    heap.make_at(1, &placed(true, written, structured(4), (5, 8)));
    // Slot 2: three elements of 12 bytes from element 3 (36 bytes in).
    // Slot 3: four RGBA32F elements from element 2 (32 bytes in).
    heap.make_at(2, &placed(false, data, structured(12), (3, 3)));
    heap.make_at(3, &placed(false, data, (R32G32B32A32_FLOAT, 0, 0), (2, 4)));

    // Raw views have their own slots; the others one each in the pool
    // after the heap's two null raw views, 4 bytes early where the view
    // does not start on 16 bytes.
    let pool = heap.alias_base() + 2;
    let entry = |stride, raw_slot, bias| {
        StrideEntry {
            stride,
            raw_slot,
            bias,
        }
        .words()
    };
    assert_eq!(heap.entry_words(0), entry(4, None, 0));
    assert_eq!(heap.entry_words(1), entry(4, Some(pool), 4));
    assert_eq!(heap.entry_words(2), entry(12, Some(pool + 1), 4));
    assert_eq!(heap.entry_words(3), entry(16, Some(pool + 2), 0));

    let cs = program();
    let (root, pso) = pipeline(device, &cs).expect("a pipeline");
    let run = |chosen: [u32; 2]| -> [u32; 8] {
        // SAFETY: the copies' first dwords, and the written view's 8
        // elements from byte 20.
        unsafe {
            slots_copy.cast::<[u32; 2]>().write(chosen);
            out.add(5).cast::<[u32; 8]>().write([0; 8]);
            dispatch(device, &heap, &root, &pso, cs.user_data_dwords()).expect("a dispatch");
            out.add(5).cast::<[u32; 8]>().read()
        }
    };
    // Lane l reads [l, 4]: of slot 2 dwords 10, 13, 16 and (past its three
    // elements) nothing; of slot 3 dwords 9, 13, 17, 21.
    let (twelve, typed) = ([110, 113, 116, 0], [109, 113, 117, 121]);
    assert_eq!(run([2, 3]), [twelve, typed].concat()[..]);
    assert_eq!(run([3, 2]), [typed, twelve].concat()[..]);

    // The title rebinds the slots: slot 2 to four elements of 20 bytes from
    // element 1 (in the raw view slot it had), slot 3 to a raw view from
    // dword 8 (its own slot; its raw view slot goes back to the pool).
    heap.make_at(2, &placed(false, data, structured(20), (1, 4)));
    heap.make_at(3, &placed(false, data, raw, (8, 64)));
    assert_eq!(heap.entry_words(2), entry(20, Some(pool + 1), 4));
    assert_eq!(heap.entry_words(3), entry(4, None, 0));
    let (twenty, plain) = ([106, 111, 116, 121], [109, 110, 111, 112]);
    assert_eq!(run([2, 3]), [twenty, plain].concat()[..]);

    // A slot made null reads nothing through its own slot, and the next
    // view to need a raw view takes the slot given back.
    heap.make_null(2);
    assert_eq!(heap.entry_words(2), [0, 0]);
    assert_eq!(run([2, 3]), [[0; 4], plain].concat()[..]);
    heap.make_at(4, &placed(false, data, structured(16), (1, 2)));
    assert_eq!(heap.entry_words(4), entry(16, Some(pool + 1), 0));
    heap.make_at(5, &placed(false, data, structured(8), (2, 2)));
    assert_eq!(heap.entry_words(5), entry(8, Some(pool + 2), 0));

    // Dynamic slots give their raw views back when their list is done.
    let first = heap.dynamic(2).expect("dynamic slots");
    heap.make_at(first, &placed(false, data, structured(12), (0, 4)));
    assert_eq!(heap.entry_words(first), entry(12, Some(pool + 3), 0));
    heap.release(first, 2);
    assert_eq!(heap.entry_words(first), [0, 0]);
    heap.make_at(6, &placed(false, data, structured(12), (0, 4)));
    assert_eq!(heap.entry_words(6), entry(12, Some(pool + 3), 0));
    drop(memory);
}

/// Give the allocator a turn at the start of cleanup for the last leased
/// slot. This exercises the release ordering without a scheduler race.
#[test]
fn retiring_slot_is_unavailable_until_its_alias_is_cleared() {
    let Some(heap) = HostHeap::new(false, 1) else {
        eprintln!("Skipping descriptor retirement test: no host view heap");
        return;
    };
    let capacity = heap.alias_base() - heap.count();
    let first = heap.dynamic(capacity).expect("empty dynamic pool");
    let last = first + capacity - 1;
    let mut during_cleanup = None;
    heap.release_with(last, 1, |heap, index| {
        assert_eq!(index, last);
        during_cleanup = Some(heap.dynamic(1));
        heap.forget_view(index);
    });

    assert_eq!(during_cleanup, Some(None), "slot reusable before cleanup");
    assert_eq!(heap.dynamic(1), Some(last), "cleared slot was not returned");
    eprintln!("Verified descriptor lease during and after alias cleanup");
}
