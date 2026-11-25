//! VICTORY_MESH_INPUT=shader-hash:heap-slot captures the first 256 bytes
//! of a buffer SRV immediately before each selected mesh draw. For indirect
//! draws the TSV also records the first source and submitted XYZ group
//! counts, followed by the command count when a count buffer is bound.

use std::sync::OnceLock;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_RANGE, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE,
    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, ID3D12Resource,
};
use windows::core::Interface;

use crate::host::{HostList, host, transition};
use crate::pipeline::host::HostPipeline;

pub(super) enum Snapshot {
    Unavailable(&'static str),
    Buffer {
        readback: ID3D12Resource,
        address: u64,
        len: usize,
    },
}

/// First indirect dispatch record, as written by the title and submitted by the host.
pub(crate) struct Arguments<'a> {
    pub address: u64,
    pub source: (&'a ID3D12Resource, u64),
    pub submitted: (&'a ID3D12Resource, u64),
    pub count: Option<(&'a ID3D12Resource, u64)>,
    pub max: u32,
    pub reused: bool,
}

pub(super) struct Counts {
    readback: ID3D12Resource,
    counted: bool,
}

impl Arguments<'_> {
    pub(super) fn capture(&self, list: &HostList) -> Option<Counts> {
        if self.max == 0 {
            return None;
        }
        let readback = host()?.read_back_buffer(28)?;
        let (source, offset) = self.source;
        let (scratch, at) = self.submitted;
        // A freshly clamped scratch buffer is in INDIRECT_ARGUMENT. A window
        // reused after indirect::done is in COMMON. Leave it ready to execute.
        let before = if self.reused {
            D3D12_RESOURCE_STATE_COMMON
        } else {
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
        };
        let copy = D3D12_RESOURCE_STATE_COPY_SOURCE;
        // SAFETY: bound() validated these record and count ranges. Source
        // and count are implicitly readable buffers already used by the clamp.
        unsafe {
            list.list.CopyBufferRegion(&readback, 0, source, offset, 12);
            list.list
                .ResourceBarrier(&[transition(scratch, before, copy)]);
            list.list.CopyBufferRegion(&readback, 12, scratch, at, 12);
            list.list.ResourceBarrier(&[transition(
                scratch,
                copy,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            )]);
            if let Some((buffer, offset)) = self.count {
                list.list.CopyBufferRegion(&readback, 24, buffer, offset, 4);
            }
        }
        Some(Counts {
            readback,
            counted: self.count.is_some(),
        })
    }
}

impl Counts {
    pub(super) fn words(&self) -> String {
        let Some(words) = read_words(&self.readback, 6 + usize::from(self.counted)) else {
            return "unmapped".into();
        };
        words
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Called once the capture's submission has finished.
fn read_words(buffer: &ID3D12Resource, count: usize) -> Option<Vec<u32>> {
    let mut data = std::ptr::null_mut();
    let range = D3D12_RANGE {
        Begin: 0,
        End: count * 4,
    };
    // SAFETY: each capture owns a readback containing these dwords.
    unsafe {
        buffer.Map(0, Some(&range), Some(&mut data)).ok()?;
        let result = (!data.is_null())
            .then(|| std::slice::from_raw_parts(data.cast::<u32>(), count).to_vec());
        buffer.Unmap(0, Some(&D3D12_RANGE::default()));
        result
    }
}

pub(super) fn capture(list: &HostList, pipeline: &HostPipeline, heap: usize) -> Option<Snapshot> {
    static SPEC: OnceLock<Option<(String, u32)>> = OnceLock::new();
    let (shader, slot) = SPEC
        .get_or_init(|| {
            let value = std::env::var("VICTORY_MESH_INPUT").ok()?;
            let (shader, slot) = value.split_once(':')?;
            Some((shader.trim().to_owned(), slot.trim().parse().ok()?))
        })
        .as_ref()?;
    if !pipeline
        .stages
        .iter()
        .any(|stage| stage.shader.hash == *shader)
    {
        return None;
    }
    let bound = (heap != 0)
        .then(|| crate::descriptor::heap(heap as crate::object::This))
        .and_then(|h| h.host())
        .and_then(|h| h.bound_buffer_view(*slot));
    let Some((view, resource)) = bound else {
        return Some(Snapshot::Unavailable("unbound"));
    };
    let Some((copy, address, len)) = crate::descriptor::host::buffer_copy(&view) else {
        return Some(Snapshot::Unavailable("no-copy"));
    };
    if resource.as_raw() != copy.buffer.as_raw() {
        return Some(Snapshot::Unavailable("other-copy"));
    }
    let offset = copy.offset_in(address)?;
    let bytes = len.min(copy.left(address)).min(256) & !3;
    if bytes == 0 {
        return Some(Snapshot::Unavailable("empty"));
    }
    let readback = host()?.read_back_buffer(bytes)?;
    // Buffer shadows use implicit read-state promotion. The draw's bindings
    // and preceding UAV barriers are in place. Keep both ends until the query
    // page is read after submission.
    unsafe {
        list.list
            .CopyBufferRegion(&readback, 0, &resource, offset, bytes)
    };
    list.keep(&resource);
    Some(Snapshot::Buffer {
        readback,
        address,
        len: bytes as usize,
    })
}

impl Snapshot {
    pub(super) fn words(&self) -> String {
        match self {
            Self::Unavailable(reason) => (*reason).into(),
            Self::Buffer {
                readback,
                address,
                len,
            } => {
                let Some(words) = read_words(readback, len / 4) else {
                    return "unmapped".into();
                };
                let hex: Vec<_> = words.iter().map(|v| format!("{v:08x}")).collect();
                format!("{address:#x} {}", hex.join(" "))
            }
        }
    }
}
