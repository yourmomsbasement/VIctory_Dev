//! Texture slots actually sampled by one recording of a command list.

use super::host::{HostHeap, TextureQueue};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use windows::Win32::Graphics::Direct3D12::*;

const SLOTS: u32 = victory_shader::emit::SLOT_STRIDES_COUNT;
const BYTES: usize = SLOTS.div_ceil(32) as usize * 4;

pub(crate) static PROMOTED: AtomicU64 = AtomicU64::new(0);

struct Bitmap {
    queue: Option<Arc<Mutex<TextureQueue>>>,
    count: u32,
    buffer: ID3D12Resource,
    readback: ID3D12Resource,
    mapping: usize,
}

#[derive(Default)]
pub(crate) struct TextureReads(Mutex<Vec<Bitmap>>);

impl TextureReads {
    /// A list can switch view heaps. Retain the queue itself so destroying
    /// and reallocating a title heap cannot deliver feedback to its successor.
    pub(crate) fn bind(&self, list: &ID3D12GraphicsCommandList, heap: &HostHeap) -> Option<u64> {
        let mut maps = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let map = if let Some(i) = maps.iter().position(|b| {
            b.queue
                .as_ref()
                .is_some_and(|q| Arc::ptr_eq(q, &heap.textures))
        }) {
            &maps[i]
        } else {
            let device = crate::host::host()?;
            static ZERO: OnceLock<Option<ID3D12Resource>> = OnceLock::new();
            let zero = ZERO
                .get_or_init(|| {
                    let (buffer, data) = device.upload_buffer(BYTES as u64)?;
                    // SAFETY: the upload mapping spans BYTES and is immutable after initialization.
                    unsafe { std::ptr::write_bytes(data, 0, BYTES) };
                    Some(buffer)
                })
                .as_ref()?;
            let map = if let Some(i) = maps.iter().position(|b| b.queue.is_none()) {
                &mut maps[i]
            } else {
                let buffer = device.uav_buffer(BYTES as u64)?;
                let readback = device.read_back_buffer(BYTES as u64)?;
                let mut data = std::ptr::null_mut();
                // SAFETY: the readback resource stays alive throughout all uses of its mapping.
                unsafe { readback.Map(0, None, Some(&mut data)).ok()? };
                maps.push(Bitmap {
                    queue: None,
                    count: 0,
                    buffer,
                    readback,
                    mapping: data as usize,
                });
                maps.last_mut()?
            };
            // Every execution clears its own bits in GPU order, including replay
            // of the same command list. Pixel shader atomics stay in video memory.
            // SAFETY: both buffers cover BYTES and resolve returns the UAV to COMMON.
            unsafe {
                list.ResourceBarrier(&[crate::host::transition(
                    &map.buffer,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )]);
                list.CopyBufferRegion(&map.buffer, 0, zero, 0, BYTES as u64);
                list.ResourceBarrier(&[crate::host::transition(
                    &map.buffer,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
            }
            map.count = heap.count().min(SLOTS);
            map.queue = Some(heap.textures.clone());
            map
        };
        // SAFETY: the recording owns the resource until its final submission completes.
        Some(unsafe { map.buffer.GetGPUVirtualAddress() })
    }

    pub(crate) fn resolve(&self, list: &ID3D12GraphicsCommandList) {
        for map in self
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|b| b.queue.is_some())
        {
            // SAFETY: the list has finished recording all its draws, and owns both resources.
            unsafe {
                list.ResourceBarrier(&[crate::host::transition(
                    &map.buffer,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                list.CopyBufferRegion(
                    &map.readback,
                    0,
                    &map.buffer,
                    0,
                    u64::from(map.count.div_ceil(32)) * 4,
                );
                list.ResourceBarrier(&[crate::host::transition(
                    &map.buffer,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
        }
    }

    pub(crate) fn completed(&self) {
        for map in self.0.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            let Some(queue) = &map.queue else { continue };
            let mut queue = queue.lock().unwrap_or_else(|e| e.into_inner());
            let mut promoted = 0;
            for word in 0..map.count.div_ceil(32) {
                // SAFETY: the fence covers the copy to the retained readback buffer.
                let mut bits = unsafe {
                    (map.mapping as *const u32)
                        .add(word as usize)
                        .read_volatile()
                };
                while bits != 0 {
                    let slot = word * 32 + bits.trailing_zeros();
                    promoted += u32::from(queue.sampled(slot));
                    bits &= bits - 1;
                }
            }
            PROMOTED.fetch_add(u64::from(promoted), Ordering::Relaxed);
        }
    }

    pub(crate) fn reset(&self) {
        for map in self.0.lock().unwrap_or_else(|e| e.into_inner()).iter_mut() {
            map.queue = None;
        }
    }
}
