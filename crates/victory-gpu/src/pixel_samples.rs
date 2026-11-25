//! GPU values from selected 2D sample instructions. Set
//! VICTORY_TRACE_PIXEL_SAMPLES=container-hash@first-x-last-x,y.
//! VICTORY_DUMP_SAMPLE_PC selects one instruction for placed texture and mip
//! dumps on subsequent material capture frames in the same descriptor heap.
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

struct Draw {
    id: u64,
    frame: u64,
    buffer: ID3D12Resource,
    readback: Option<ID3D12Resource>,
    views: Option<SampledViews>,
}

#[derive(Default)]
pub(crate) struct Samples(Mutex<Vec<Draw>>);

impl Samples {
    pub(crate) fn bind(
        &self,
        list: &HostList,
        frame: u64,
        heap: Option<&crate::descriptor::Heap>,
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
            views: if save {
                heap.and_then(|h| note_views(h, frame))
            } else {
                None
            },
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
            let path = dir.join(format!("frame-{}-pixel-samples.tsv", draw.frame));
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
                let _ = writeln!(
                    file,
                    "probe\tpc\tx\tdepth\tu\tv\tr\tg\tb\ta\theap_index\ttable_slot\ttable_dword\twidth\theight\tmips\timplicit_lod"
                );
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
