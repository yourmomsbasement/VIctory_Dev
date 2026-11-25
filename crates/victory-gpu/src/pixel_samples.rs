//! GPU values from selected pixel samples and compute instructions. Set
//! VICTORY_TRACE_PIXEL_SAMPLES=container-hash@first-x-last-x,y.
//! VICTORY_DUMP_SAMPLE_PC selects one instruction for placed texture and mip
//! dumps on subsequent material capture frames in the same descriptor heap.
//! VICTORY_TRACE_COMPUTE_VALUES accepts semicolon-separated entries such as
//! hash:7d50:0,1@320,212 for a native instruction, two coordinate VGPRs
//! and a pixel (or a row span such as 290-410,212). The record holds the multiply-accumulate operands or the
//! UV, reference/LOD, sampled red channel and nearest clamped mip-zero texel.
use crate::host::{HostList, host, transition};
use std::collections::BTreeMap;
use std::io::{BufWriter, Write};
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicU64, Ordering},
};
use victory_shader::emit::{MAX_PIXEL_SAMPLES, PIXEL_SAMPLE_WORDS};
use windows::Win32::Graphics::Direct3D12::*;

const BYTES: u64 = 8 + MAX_PIXEL_SAMPLES as u64 * PIXEL_SAMPLE_WORDS as u64 * 4;

/// A GPU-observed slot belongs to one descriptor heap. Keep only the slot
/// numbers across captures, then inspect each one's current view on the
/// next selected frame. The title may rewrite a slot between frames.
pub(crate) type SampledViews = Arc<Mutex<BTreeMap<u32, Option<u64>>>>;

fn dump_pc() -> Option<u32> {
    static PC: OnceLock<Option<u32>> = OnceLock::new();
    *PC.get_or_init(|| {
        let pc = std::env::var("VICTORY_DUMP_SAMPLE_PC").ok()?;
        u32::from_str_radix(pc.trim().trim_start_matches("0x"), 16).ok()
    })
}

/// Streamed mips can be readable while larger mips have no pages. Save
/// the source and a byte per intersecting 4 KiB page to distinguish actual
/// zeros from holes in the snapshot.
fn snapshot_source(address: u64, size: u64) -> Option<(String, String)> {
    if address == 0 || !(1..=8 << 20).contains(&size) {
        return None;
    }
    address.checked_add(size)?;
    let folder = crate::capture::folder()?;
    let mut bytes = vec![0u8; size as usize];
    let mut mapped = Vec::new();
    let mut offset = 0;
    while offset < bytes.len() {
        let at = address + offset as u64;
        let count = (4096 - (at as usize & 4095)).min(bytes.len() - offset);
        let page = &mut bytes[offset..offset + count];
        let ok = crate::host::read_into(at as *const u8, page);
        if !ok {
            page.fill(0);
        }
        mapped.push(u8::from(ok));
        offset += count;
    }
    Some((
        crate::capture::snapshot_material(folder, &bytes).ok()?,
        crate::capture::snapshot_material(folder, &mapped).ok()?,
    ))
}

fn note_views(heap: &crate::descriptor::Heap, frame: u64) -> Option<SampledViews> {
    let pc = dump_pc()?;
    let (shader, _) = selected()?;
    let host = heap.host()?;
    if crate::capture::material_frame(frame) {
        let pending: Vec<u32> = host
            .sampled_views
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter_mut()
            .filter_map(|(&slot, last)| {
                (*last != Some(frame)).then(|| {
                    *last = Some(frame);
                    slot
                })
            })
            .collect();
        for slot in pending {
            let view = heap.handle(slot).and_then(crate::descriptor::view);
            let shape = host.bound_shape(slot);
            let made = host.describe_slot(slot);
            let Some(crate::descriptor::View::Placed(false, resource, srv)) = view else {
                tracing::info!(
                    frame,
                    pc,
                    slot,
                    ?shape,
                    ?made,
                    ?view,
                    "sampled texture view"
                );
                continue;
            };
            let address = u64::from(srv[10]) | u64::from(srv[11]) << 32;
            let size = crate::resource::placed_bytes(&resource);
            let pages = size.and_then(|len| crate::texture::trace_readable_pages(address, len));
            let source = size.and_then(|n| snapshot_source(address, n));
            let copy = crate::resource::placed_copy(address, &resource);
            let matches = copy.as_ref().and_then(|c| host.shows(slot, &c.texture));
            tracing::info!(
                frame,
                pc,
                slot,
                ?shape,
                ?made,
                ?resource,
                ?srv,
                ?pages,
                ?source,
                ?matches,
                "sampled texture view"
            );
            if let Some(copy) = copy.filter(|_| matches == Some(true)) {
                // SAFETY: this reference owns the placed texture across capture.
                let levels = u32::from(unsafe { copy.texture.GetDesc() }.MipLevels);
                let end = srv[4].saturating_add(srv[5]).min(levels);
                let name = format!("{shader}-pc{pc:x}");
                for mip in srv[4]..end {
                    let mut desc = srv;
                    desc[4] = mip;
                    crate::capture::note_material(frame, &name, 0, slot, &desc, copy.clone());
                }
            }
        }
    }
    Some(host.sampled_views.clone())
}

pub(crate) fn selected() -> Option<&'static (String, [u32; 3])> {
    static SPEC: OnceLock<Option<(String, [u32; 3])>> = OnceLock::new();
    SPEC.get_or_init(|| {
        let spec = std::env::var("VICTORY_TRACE_PIXEL_SAMPLES").ok()?;
        let (hash, point) = spec.split_once('@')?;
        let (xs, y) = point.split_once(',')?;
        let (first, last) = xs.split_once('-').unwrap_or((xs, xs));
        let span: [u32; 3] = [first.parse().ok()?, last.parse().ok()?, y.parse().ok()?];
        (hash.len() == 16 && hash.bytes().all(|c| c.is_ascii_hexdigit()) && span[0] <= span[1])
            .then(|| (hash.to_ascii_lowercase(), span))
    })
    .as_ref()
}

/// Semicolon-separated hash:hex-pc:x-vgpr,y-vgpr@pixel-x,pixel-y entries.
pub(crate) fn compute_probes(hash: &str) -> Option<&'static [victory_shader::emit::ComputeProbe]> {
    use victory_shader::emit::ComputeProbe;
    static PROBES: OnceLock<BTreeMap<String, Vec<ComputeProbe>>> = OnceLock::new();
    let probes = PROBES.get_or_init(|| {
        std::env::var("VICTORY_TRACE_COMPUTE_VALUES").unwrap_or_default()
            .split(';').filter_map(|item| {
                let (site, pixel) = item.trim().split_once('@')?;
                let (shader, rest) = site.split_once(':')?;
                let (pc, regs) = rest.split_once(':')?;
                let (rx, ry) = regs.split_once(',')?;
                let (x, y) = pixel.split_once(',')?;
                let (first, last) = x.split_once('-').unwrap_or((x, x));
                let span = [first.parse().ok()?, last.parse().ok()?, y.parse().ok()?];
                if span[0] > span[1] { return None; }
                let probe = ComputeProbe {
                    pc: usize::from_str_radix(pc.trim_start_matches("0x"), 16).ok()?,
                    coord_regs: [rx.parse().ok()?, ry.parse().ok()?],
                    span,
                };
                Some((shader.to_ascii_lowercase(), probe))
            }).fold(BTreeMap::new(), |mut by_shader, (hash, probe)| {
                by_shader.entry(hash).or_insert_with(Vec::new).push(probe);
                by_shader
            })
    });
    probes.get(hash).map(Vec::as_slice)
}

struct Draw {
    id: u64,
    frame: u64,
    buffer: ID3D12Resource,
    readback: Option<ID3D12Resource>,
    views: Option<SampledViews>,
    compute_shader: Option<String>,
}

#[derive(Default)]
pub(crate) struct Samples(Mutex<Vec<Draw>>);

impl Samples {
    pub(crate) fn bind(
        &self,
        list: &HostList,
        frame: u64,
        heap: Option<&crate::descriptor::Heap>,
        compute_shader: Option<&str>,
    ) -> Option<(u64, Option<u64>)> {
        let save = crate::capture::listed(frame) && crate::capture::folder().is_some();
        let mut draws = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if !save && let Some(draw) = draws.iter().find(|d| d.readback.is_none()) {
            // SAFETY: the list owns this zero-enable header through completion.
            return Some((unsafe { draw.buffer.GetGPUVirtualAddress() }, None));
        }
        let host = host()?;
        let buffer = host.uav_buffer(if save { BYTES } else { 8 })?;
        let readback = if save {
            Some(host.read_back_buffer(BYTES)?)
        } else {
            None
        };
        let (header, ptr) = list.upload(host, 8)?;
        // SAFETY: header is a private, mapped 8-byte upload held by the list.
        // The GPU resets the count and enable flag on each execution.
        let address = unsafe {
            ptr.cast::<[u32; 2]>().write([0, u32::from(save)]);
            list.list.ResourceBarrier(&[transition(
                &buffer,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            list.list.CopyBufferRegion(&buffer, 0, &header, 0, 8);
            list.list.ResourceBarrier(&[transition(
                &buffer,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            buffer.GetGPUVirtualAddress()
        };
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        draws.push(Draw {
            id,
            frame,
            buffer,
            readback,
            views: if save && compute_shader.is_none() {
                heap.and_then(|h| note_views(h, frame))
            } else {
                None
            },
            compute_shader: compute_shader.map(str::to_owned),
        });
        Some((address, save.then_some(id)))
    }

    pub(crate) fn resolve(&self, list: &ID3D12GraphicsCommandList) {
        for draw in self.0.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            let mut state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            // SAFETY: all prior writes are on this list, which owns both buffers.
            unsafe {
                if let Some(readback) = &draw.readback {
                    list.ResourceBarrier(&[transition(
                        &draw.buffer,
                        state,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    )]);
                    state = D3D12_RESOURCE_STATE_COPY_SOURCE;
                    list.CopyBufferRegion(readback, 0, &draw.buffer, 0, BYTES);
                }
                list.ResourceBarrier(&[transition(
                    &draw.buffer,
                    state,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
        }
    }

    pub(crate) fn save(&self) {
        let draws = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if draws.is_empty() {
            return;
        }
        let Some(dir) = crate::capture::folder() else {
            return;
        };
        static OUTPUT: Mutex<()> = Mutex::new(());
        let _output = OUTPUT.lock().unwrap_or_else(|e| e.into_inner());
        for draw in draws.iter() {
            let Some(readback) = &draw.readback else {
                continue;
            };
            let kind = if draw.compute_shader.is_some() { "compute-values" } else { "pixel-samples" };
            let path = dir.join(format!("frame-{}-{kind}.tsv", draw.frame));
            let Ok(file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            else {
                continue;
            };
            let header = file.metadata().is_ok_and(|m| m.len() == 0);
            let mut file = BufWriter::new(file);
            if header {
                if draw.compute_shader.is_some() {
                    let _ = writeln!(file, "probe\tshader\tpc\tx\ty\top\ta_or_u\tb_or_v\taccumulator_or_reference_lod\tresult\twidth\theight\tnearest_clamped_mip0\theap_index\tclamp_top_left\tclamp_top_right\tclamp_bottom_left\tclamp_bottom_right");
                } else {
                    let _ = writeln!(file, "probe\tpc\tx\tdepth\tu\tv\tr\tg\tb\ta\theap_index\ttable_slot\ttable_dword\twidth\theight\tmips\timplicit_lod");
                }
            }
            let mut data = std::ptr::null_mut();
            // SAFETY: the submission fence covers the copy. The shader writes
            // one record for each index below MAX_PIXEL_SAMPLES after the header.
            unsafe {
                if readback.Map(0, None, Some(&mut data)).is_err() {
                    continue;
                }
                let words = std::slice::from_raw_parts(data.cast::<u32>(), BYTES as usize / 4);
                if words[0] > MAX_PIXEL_SAMPLES {
                    let _ = writeln!(
                        file,
                        "# probe {} recorded {} of {} samples",
                        draw.id, MAX_PIXEL_SAMPLES, words[0]
                    );
                }
                for row in words[2..]
                    .chunks_exact(PIXEL_SAMPLE_WORDS as usize)
                    .take(words[0] as usize)
                {
                    if let Some(shader) = &draw.compute_shader {
                        let [a, b, acc, result] = [4, 5, 6, 7].map(|i| f32::from_bits(row[i]));
                        let op = match row[3] { 0 => "fmac", 62 => "sample_lz", 65 => "sample_c_lz", _ => "unknown" };
                        let _ = write!(file, "{}\t{shader}\t{:x}\t{}\t{}\t{op}\t{a}\t{b}\t{acc}\t{result}",
                            draw.id, row[0], row[1], row[2]);
                        if row[3] == 0 {
                            let _ = writeln!(file, "\t-\t-\t-\t-\t-\t-\t-\t-");
                        } else {
                            let [tl, tr, bl, br] = [12, 13, 14, 15].map(|i| f32::from_bits(row[i]));
                            let _ = writeln!(file, "\t{}\t{}\t{}\t{}\t{tl}\t{tr}\t{bl}\t{br}", row[8], row[9], f32::from_bits(row[10]), row[11]);
                        }
                        continue;
                    }
                    if row[3] != u32::MAX
                        && Some(row[0]) == dump_pc()
                        && let Some(views) = &draw.views
                    {
                        views
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .entry(row[3])
                            .or_default();
                    }
                    let floats = [2, 4, 5, 8, 9, 10, 11].map(|i| f32::from_bits(row[i]));
                    let [z, u, v, r, g, b, a] = floats;
                    let _ = writeln!(
                        file,
                        "{}\t{:x}\t{}\t{z}\t{u}\t{v}\t{r}\t{g}\t{b}\t{a}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        draw.id,
                        row[0],
                        row[1],
                        row[3],
                        row[6],
                        row[7],
                        row[12],
                        row[13],
                        row[14],
                        f32::from_bits(row[15])
                    );
                }
                readback.Unmap(0, None);
            }
        }
    }

    pub(crate) fn reset(&self) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
}
