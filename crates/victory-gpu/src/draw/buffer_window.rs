//! Compare a small table SRV with the same address in a heap SRV and the
//! bytes preceding it. VICTORY_TRACE_BUFFER_WINDOW=shader:slot+entry:heap:prefix
//! selects the anchor, heap index and prefix size in bytes. The snapshots
//! preserve guest data at submission and both GPU views at the command.
//! An optional final hex mask selects a placed SRV index from the first
//! dword of the window. Following captures read the first GPU word of
//! recently observed slots in the same heap.
use super::{Root, State, View};
use crate::descriptor::host::buffer_copy;
use crate::host::{HostList, host};
use crate::pipeline::host::Stage;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use windows::Win32::Graphics::Direct3D12::{D3D12_RANGE, ID3D12Resource};
use windows::core::Interface;

struct Spec {
    shader: String,
    table: u8,
    entry: u32,
    heap: u32,
    prefix: u64,
    mask: Option<u32>,
    recent: Mutex<Vec<(usize, u64, u32)>>,
}

fn spec() -> Option<&'static Spec> {
    static SPEC: OnceLock<Option<Spec>> = OnceLock::new();
    SPEC.get_or_init(|| {
        let value = std::env::var("VICTORY_TRACE_BUFFER_WINDOW").ok()?;
        let mut fields = value.split(':');
        let shader = fields.next()?.to_owned();
        let (table, entry) = fields.next()?.split_once('+')?;
        let spec = Spec {
            shader,
            table: table.parse().ok()?,
            entry: entry.parse().ok()?,
            heap: fields.next()?.parse().ok()?,
            prefix: fields.next()?.parse().ok()?,
            mask: match fields.next() {
                Some(mask) => Some(u32::from_str_radix(mask, 16).ok()?),
                None => None,
            },
            recent: Mutex::default(),
        };
        (fields.next().is_none() && spec.prefix <= 1 << 20 && spec.entry % 8 == 0).then_some(spec)
    })
    .as_ref()
}

struct Sample {
    id: u64,
    buffer: ID3D12Resource,
    bytes: usize,
    table_bytes: usize,
    slots: Vec<(u32, u64)>,
}

#[derive(Default)]
pub(crate) struct Windows(Mutex<Vec<Sample>>);

impl State {
    pub(super) fn trace_buffer_window(
        &self,
        list: &HostList,
        stage: &Stage,
        made: &[(u8, u32)],
        compute: bool,
    ) {
        let Some(spec) = spec().filter(|s| s.shader == stage.shader.hash) else {
            return;
        };
        let frame = crate::queue::presented() + 1;
        if !crate::capture::listed(frame) {
            return;
        }
        let Some(dir) = crate::capture::folder() else {
            return;
        };
        let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(spec.table)) else {
            return;
        };
        let Some(handle) = table.checked_add(u64::from(spec.entry) * 4) else {
            return;
        };
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let Some(views) = heap.host() else { return };
        let index = made
            .iter()
            .find(|(slot, _)| *slot == spec.table)
            .and_then(|(_, start)| (start / 32).checked_add(spec.entry / 8))
            .or_else(|| heap.index(handle));
        let Some(index) = index else { return };
        let Some(anchor @ (View::Shader(..) | View::Placed(false, ..))) =
            crate::descriptor::view(handle)
        else {
            return;
        };
        let Some((pool, pool_gpu)) = views.bound_buffer_view(spec.heap) else {
            return;
        };
        let Some((a, address, anchor_len)) = buffer_copy(&anchor) else {
            return;
        };
        let Some((p, base, pool_len)) = buffer_copy(&pool) else {
            return;
        };
        if views.shows(index, &a.buffer) != Some(true) || pool_gpu.as_raw() != p.buffer.as_raw() {
            return;
        }
        let Some(start) = address.checked_sub(spec.prefix).filter(|&s| s >= base) else {
            return;
        };
        let Some(left) = pool_len.checked_sub(start - base) else {
            return;
        };
        if p.offset_in(start).is_none() || a.offset_in(address).is_none() {
            return;
        }
        let table_bytes = anchor_len.min(a.left(address)).min(4096);
        let bytes = spec.prefix + table_bytes;
        if table_bytes == 0 || bytes > left || bytes > p.left(start) {
            return;
        }
        let heap_key = std::ptr::from_ref(heap) as usize;
        let heap_base = heap.start;
        let recent = spec
            .recent
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let mut headers = Vec::new();
        for (key, origin, slot) in recent {
            if key != heap_key || origin != heap_base {
                continue;
            }
            let Some((view, resource)) = views.bound_buffer_view(slot) else {
                continue;
            };
            let Some((copy, at, len)) = buffer_copy(&view) else {
                continue;
            };
            if resource.as_raw() == copy.buffer.as_raw()
                && len >= 4
                && copy.offset_in(at).is_some()
                && copy.left(at) >= 4
            {
                headers.push((slot, at, resource, copy));
            }
        }
        let length = bytes + table_bytes + headers.len() as u64 * 4;
        let Some(buffer) = host().and_then(|h| h.read_back_buffer(length)) else {
            return;
        };
        // Both sources are bound buffer SRVs, using implicit read promotion.
        // The explicit bounds above keep each copy inside its actual resource.
        unsafe {
            list.list
                .CopyBufferRegion(&buffer, 0, &pool_gpu, p.offset(start), bytes);
            list.list
                .CopyBufferRegion(&buffer, bytes, &a.buffer, a.offset(address), table_bytes);
            for (i, (_, at, resource, copy)) in headers.iter().enumerate() {
                list.list.CopyBufferRegion(
                    &buffer,
                    bytes + table_bytes + 4 * i as u64,
                    resource,
                    copy.offset(*at),
                    4,
                );
                list.keep(resource);
            }
        }
        list.keep(&a.buffer);
        list.keep(&pool_gpu);
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let slots = headers
            .iter()
            .map(|(slot, at, _, _)| (*slot, *at))
            .collect();
        list.defer_host(move || {
            let mut guest = vec![0; bytes as usize];
            let source = if crate::host::read_into(start as *const u8, &mut guest) {
                if let Some(mask) = spec.mask
                    && let Some(word) = guest.first_chunk::<4>()
                {
                    let slot = u32::from_le_bytes(*word) & mask;
                    if let Some((_, at, _, copy)) = headers.iter().find(|(s, ..)| *s == slot) {
                        copy.trace_header(id, slot, *at);
                    }
                    if let Some((guest, bound)) =
                        crate::descriptor::inspect_placed_slot(heap_key, heap_base, slot)
                    {
                        if bound.is_some() {
                            let found = (heap_key, heap_base, slot);
                            let mut slots = spec.recent.lock().unwrap_or_else(|e| e.into_inner());
                            slots.retain(|key| *key != found);
                            if slots.len() == 32 {
                                slots.remove(0);
                            }
                            slots.push(found);
                        }
                        let describe = |view: View| {
                            let (address, len) = crate::descriptor::host::backed_view_range(&view)?;
                            let mut header = [0; 4];
                            let word = (len >= 4
                                && crate::host::read_into(address as *const u8, &mut header))
                            .then(|| u32::from_le_bytes(header));
                            Some((address, len, word))
                        };
                        let guest = guest.and_then(describe);
                        let bound = bound.and_then(describe);
                        tracing::info!(id, slot, ?guest, ?bound, "indexed shader buffer view");
                    }
                }
                crate::capture::snapshot_material(dir, &guest).ok()
            } else {
                None
            };
            tracing::info!(
                id,
                frame,
                base,
                start,
                address,
                ?source,
                "submitted shader buffer window"
            );
        });
        list.buffer_windows
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(Sample {
                id,
                buffer,
                bytes: bytes as usize,
                table_bytes: table_bytes as usize,
                slots,
            });
    }
}

impl Windows {
    pub(crate) fn save(&self) {
        let samples = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if samples.is_empty() {
            return;
        }
        let Some(dir) = crate::capture::folder() else {
            return;
        };
        for s in samples.iter() {
            let range = D3D12_RANGE {
                Begin: 0,
                End: s.bytes + s.table_bytes + 4 * s.slots.len(),
            };
            let mut data = std::ptr::null_mut();
            // Submission has finished. Map only the bytes copied above.
            unsafe {
                if s.buffer.Map(0, Some(&range), Some(&mut data)).is_err() {
                    continue;
                }
                if !data.is_null() {
                    let all = std::slice::from_raw_parts(data.cast::<u8>(), range.End);
                    let pool = crate::capture::snapshot_material(dir, &all[..s.bytes]).ok();
                    let (table, heads) = all[s.bytes..].split_at(s.table_bytes);
                    let table = crate::capture::snapshot_material(dir, table).ok();
                    tracing::info!(id = s.id, ?pool, ?table, "executed shader buffer window");
                    for ((slot, at), word) in s.slots.iter().zip(heads.chunks_exact(4)) {
                        let word = u32::from_le_bytes(word.try_into().unwrap());
                        tracing::info!(id = s.id, slot, at, word, "indexed GPU buffer header");
                    }
                }
                s.buffer.Unmap(0, Some(&D3D12_RANGE::default()));
            }
        }
    }

    pub(crate) fn reset(&self) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
}
