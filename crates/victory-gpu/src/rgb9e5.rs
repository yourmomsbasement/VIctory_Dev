//! A host library's output written to a title texture the host GPU cannot
//! write through an unordered access view: Grand Theft Auto VI upscales
//! into an RGB9E5 texture, a format NVIDIA's GPUs sample but cannot store
//! to (no typed UAV store, no render target), and upscalers write their
//! output through such a view. The upscaler writes an FP16 texture of ours
//! instead ([`Bridge::texture`]); its rows are copied to a buffer, a compute
//! shader of ours packs each pixel into RGB9E5 in a second buffer, and that
//! buffer is copied into the title's texture, which a copy may write in any
//! format ([`Bridge::write`]).
//!
//! The resources are kept per output size and never freed: lists run one
//! submission at a time, so a later upscale of the same size reuses them
//! after the earlier one ran, and the title upscales to one size.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::s;

use crate::host::{host, transition};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), SRV(t0), UAV(u0)"
cbuffer Size : register(b0) { uint width; uint height; uint from_pitch; uint to_pitch; };
// FP16 RGBA rows, as DLSS wrote them.
ByteAddressBuffer from : register(t0);
// RGB9E5 rows, for the title's texture.
RWByteAddressBuffer to : register(u0);
// D3D's shared-exponent format: three 9-bit mantissas without an implicit
// one over a 5-bit exponent of bias 15; negative values and NaN store 0.
uint pack(float3 rgb) {
    const float largest = 65408.0; // 511 / 512 * 2^16
    rgb = min(max(rgb, 0.0), largest);
    float m = max(rgb.r, max(rgb.g, rgb.b));
    int e = max(-16, (int)floor(log2(max(m, 1e-30)))) + 16;
    float scale = exp2((float)(e - 24));
    if ((uint)floor(m / scale + 0.5) == 512) {
        e += 1;
        scale *= 2.0;
    }
    uint3 q = (uint3)floor(rgb / scale + 0.5);
    return q.r | q.g << 9 | q.b << 18 | (uint)e << 27;
}
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= width || id.y >= height) return;
    uint2 h = from.Load2(id.y * from_pitch + id.x * 8);
    float3 rgb = float3(f16tof32(h.x), f16tof32(h.x >> 16), f16tof32(h.y));
    to.Store(id.y * to_pitch + id.x * 4, pack(rgb));
}
"#;

/// Whether the host GPU stores to `texture` through a typed unordered
/// access view, as upscalers write their output. NVIDIA creates RGB9E5
/// textures that allow unordered access, so their host copies take the
/// state, but it cannot store to that format: an upscaler writing one
/// removes the device. Formats are asked once.
pub fn stores(texture: &ID3D12Resource) -> bool {
    static KNOWN: Mutex<Vec<(i32, bool)>> = Mutex::new(Vec::new());
    // SAFETY: a live resource.
    let format = unsafe { texture.GetDesc() }.Format;
    let mut known = KNOWN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&(_, ok)) = known.iter().find(|(f, _)| *f == format.0) {
        return ok;
    }
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: format,
        ..Default::default()
    };
    let ok = host().is_some_and(|h| {
        // SAFETY: a plain query into a struct of its size.
        unsafe {
            h.device.CheckFeatureSupport(
                D3D12_FEATURE_FORMAT_SUPPORT,
                (&raw mut support).cast(),
                size_of_val(&support) as u32,
            )
        }
        .is_ok()
            && support.Support1.0 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW.0 != 0
            && support.Support2.0 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE.0 != 0
    });
    tracing::debug!(
        format = format.0,
        stores = ok,
        support1 = support.Support1.0,
        support2 = support.Support2.0,
        "typed unordered stores"
    );
    known.push((format.0, ok));
    ok
}

/// The packing shader.
struct Packer {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Packer {}
// SAFETY: as above.
unsafe impl Sync for Packer {}

fn packer() -> Option<&'static Packer> {
    static PACKER: OnceLock<Option<Packer>> = OnceLock::new();
    PACKER
        .get_or_init(|| match make_packer() {
            Ok(p) => Some(p),
            Err(e) => {
                tracing::warn!(error = %e, "no RGB9E5 packer: outputs the host cannot write stay unwritten");
                None
            }
        })
        .as_ref()
}

fn make_packer() -> windows::core::Result<Packer> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("rgb9e5"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("RGB9E5 packer: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Packer { root, pso })
    }
}

/// An FP16 output of one size, and the buffers it goes through.
pub struct Bridge {
    /// What the library writes, kept in `UNORDERED_ACCESS` between uses.
    texture: ID3D12Resource,
    /// Its rows (kept in `COPY_DEST`) and the packed rows (kept in
    /// `UNORDERED_ACCESS`), each row on 256 bytes.
    rows: ID3D12Resource,
    packed: ID3D12Resource,
    size: [u32; 2],
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Bridge {}
// SAFETY: as above.
unsafe impl Sync for Bridge {}

/// Bytes of a copy row of `width` pixels of `bytes` each.
fn pitch(width: u32, bytes: u32) -> u32 {
    (width * bytes).next_multiple_of(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT)
}

/// The bridge for outputs of `size`, made on first use.
pub fn bridge(size: [u32; 2]) -> Option<&'static Bridge> {
    static BRIDGES: Mutex<Option<HashMap<[u32; 2], &'static Bridge>>> = Mutex::new(None);
    packer()?;
    let mut bridges = BRIDGES.lock().unwrap_or_else(|e| e.into_inner());
    let bridges = bridges.get_or_insert_with(HashMap::new);
    if let Some(&b) = bridges.get(&size) {
        return Some(b);
    }
    let made = match make_bridge(size) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(error = %e, ?size, "no RGB9E5 bridge for this size");
            return None;
        }
    };
    tracing::info!(
        ?size,
        "RGB9E5 output written through an FP16 texture and a packing pass"
    );
    let made: &'static Bridge = Box::leak(Box::new(made));
    bridges.insert(size, made);
    Some(made)
}

fn make_bridge(size: [u32; 2]) -> windows::core::Result<Bridge> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let [width, height] = size;
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let make = |desc: &D3D12_RESOURCE_DESC, state| {
        let mut r: Option<ID3D12Resource> = None;
        // SAFETY: plain creation with valid descriptions.
        unsafe {
            device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, desc, state, None, &mut r)
        }?;
        r.ok_or_else(windows::core::Error::empty)
    };
    let buffer = |bytes: u32, flags| D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: u64::from(bytes) * u64::from(height),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    };
    let texture = make(
        &D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(width),
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        },
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    )?;
    let rows = make(
        &buffer(pitch(width, 8), D3D12_RESOURCE_FLAG_NONE),
        D3D12_RESOURCE_STATE_COPY_DEST,
    )?;
    let packed = make(
        &buffer(pitch(width, 4), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    )?;
    crate::host::name(&texture, || format!("RGB9E5 bridge {width}x{height}"));
    Ok(Bridge {
        texture,
        rows,
        packed,
        size,
    })
}

/// A buffer copy location of rows of `format`.
fn rows(
    buffer: &ID3D12Resource,
    format: DXGI_FORMAT,
    size: [u32; 2],
    pitch: u32,
) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: borrowed; the location never releases it.
        pResource: unsafe { std::mem::transmute_copy(buffer) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: format,
                    Width: size[0],
                    Height: size[1],
                    Depth: 1,
                    RowPitch: pitch,
                },
            },
        },
    }
}

fn subresource(texture: &ID3D12Resource) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: borrowed; the location never releases it.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: 0,
        },
    }
}

impl Bridge {
    /// The FP16 texture (`R16G16B16A16_FLOAT`) the library writes in
    /// place of the title's, in `UNORDERED_ACCESS`.
    pub fn texture(&self) -> &ID3D12Resource {
        &self.texture
    }

    /// Records, after the library wrote [`Bridge::texture`], its pixels packed
    /// into `output`'s top-left corner (`output` in `COPY_DEST`, an RGB9E5
    /// texture at least this size). The bridge's resources end in the
    /// states they started in.
    pub fn write(&self, list: &ID3D12GraphicsCommandList, output: &ID3D12Resource) {
        let Some(packer) = packer() else { return };
        let [width, height] = self.size;
        let (from_pitch, to_pitch) = (pitch(width, 8), pitch(width, 4));
        use D3D12_RESOURCE_STATE_COPY_DEST as COPY_DEST;
        use D3D12_RESOURCE_STATE_COPY_SOURCE as COPY_SOURCE;
        use D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE as READ;
        use D3D12_RESOURCE_STATE_UNORDERED_ACCESS as UNORDERED;
        let from = rows(
            &self.rows,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            self.size,
            from_pitch,
        );
        let to = rows(
            &self.packed,
            DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
            self.size,
            to_pitch,
        );
        // SAFETY: an open list; our resources in the states noted; `output`
        // in COPY_DEST, as the caller says.
        unsafe {
            list.ResourceBarrier(&[transition(&self.texture, UNORDERED, COPY_SOURCE)]);
            list.CopyTextureRegion(&from, 0, 0, 0, &subresource(&self.texture), None);
            list.ResourceBarrier(&[transition(&self.rows, COPY_DEST, READ)]);
            list.SetComputeRootSignature(&packer.root);
            list.SetPipelineState(&packer.pso);
            let size = [width, height, from_pitch, to_pitch];
            list.SetComputeRoot32BitConstants(0, 4, size.as_ptr().cast(), 0);
            list.SetComputeRootShaderResourceView(1, self.rows.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(2, self.packed.GetGPUVirtualAddress());
            list.Dispatch(width.div_ceil(8), height.div_ceil(8), 1);
            list.ResourceBarrier(&[transition(&self.packed, UNORDERED, COPY_SOURCE)]);
            list.CopyTextureRegion(&subresource(output), 0, 0, 0, &to, None);
            list.ResourceBarrier(&[
                transition(&self.texture, COPY_SOURCE, UNORDERED),
                transition(&self.rows, READ, COPY_DEST),
                transition(&self.packed, COPY_SOURCE, UNORDERED),
            ]);
        }
    }
}
