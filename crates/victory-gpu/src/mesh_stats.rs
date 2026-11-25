//! Per-draw mesh, rasterizer and occlusion counters for selected capture frames.
//! Enable with VICTORY_MESH_STATS=1 and select frames as for the command trace.
//! VICTORY_MATERIAL_ROW=1280x720@160 also records which pixels each material
//! draw changes along row 160 of its first color target.

use std::io::Write;
use std::sync::{Mutex, OnceLock};
use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};

mod row;

const SLOTS: u32 = 64;
type Statistics = D3D12_QUERY_DATA_PIPELINE_STATISTICS1;
const OCCLUSION_OFFSET: u64 = SLOTS as u64 * size_of::<Statistics>() as u64;

struct Draw {
    frame: u64,
    args: Option<u64>,
    target: usize,
    depth: usize,
    pipeline: String,
    sampled_row: bool,
}

struct Page {
    stats: ID3D12QueryHeap,
    occlusion: ID3D12QueryHeap,
    buffer: ID3D12Resource,
    draws: Vec<Draw>,
}

impl Page {
    fn new() -> Option<Self> {
        let host = host()?;
        let heap = |kind| {
            let mut result = None;
            let desc = D3D12_QUERY_HEAP_DESC {
                Type: kind,
                Count: SLOTS,
                NodeMask: 0,
            };
            // SAFETY: the device fills the heap interface on success.
            unsafe { host.device.CreateQueryHeap(&desc, &mut result) }.ok()?;
            result
        };
        Some(Self {
            stats: heap(D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS1)?,
            occlusion: heap(D3D12_QUERY_HEAP_TYPE_OCCLUSION)?,
            buffer: host.read_back_buffer(row::page_bytes())?,
            draws: Vec::new(),
        })
    }
}

#[derive(Default)]
pub(crate) struct Queries(OnceLock<Mutex<Vec<Page>>>);

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        if std::env::var("VICTORY_MESH_STATS").as_deref() != Ok("1")
            || crate::capture::folder().is_none()
        {
            return false;
        }
        let Some(host) = host() else { return false };
        let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS9::default();
        // SAFETY: the output type matches the feature and byte count.
        let ok = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS9,
                (&raw mut opts).cast(),
                size_of_val(&opts) as u32,
            )
        }
        .is_ok()
            && opts.MeshShaderPipelineStatsSupported.as_bool();
        tracing::info!(supported = ok, "mesh draw statistics");
        ok
    })
}

pub(crate) struct Active {
    stats: ID3D12QueryHeap,
    occlusion: ID3D12QueryHeap,
    slot: u32,
    row: Option<row::Probe>,
}

pub(crate) fn begin(
    list: &HostList,
    pipeline: usize,
    args: Option<u64>,
    targets: &[usize],
    depth: usize,
) -> Option<Active> {
    if !enabled() {
        return None;
    }
    let frame = crate::queue::presented() + 1;
    if !crate::capture::listed(frame) {
        return None;
    }
    let p = crate::pipeline::host_pipeline(pipeline as crate::object::This)?;
    let mut pages = list
        .mesh_queries
        .0
        .get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if pages.last().is_none_or(|p| p.draws.len() == SLOTS as usize) {
        pages.push(Page::new()?);
    }
    let page = pages.last_mut()?;
    let index = page.draws.len() as u32;
    let row = if targets.len() >= 5 {
        row::Probe::new(targets[0], &page.buffer, index)
    } else {
        None
    };
    if let Some(probe) = &row {
        probe.copy(list, false);
    }
    page.draws.push(Draw {
        frame,
        args,
        target: targets.first().copied().unwrap_or(0),
        depth,
        pipeline: p.name.clone(),
        sampled_row: row.is_some(),
    });
    // SAFETY: fresh slots on a direct list, paired with end() on this list.
    unsafe {
        list.list
            .BeginQuery(&page.stats, D3D12_QUERY_TYPE_PIPELINE_STATISTICS1, index);
        list.list
            .BeginQuery(&page.occlusion, D3D12_QUERY_TYPE_OCCLUSION, index);
    }
    Some(Active {
        stats: page.stats.clone(),
        occlusion: page.occlusion.clone(),
        slot: index,
        row,
    })
}

impl Active {
    pub(crate) fn end(self, list: &HostList) {
        // SAFETY: these slots began immediately before the mesh command.
        unsafe {
            list.list
                .EndQuery(&self.occlusion, D3D12_QUERY_TYPE_OCCLUSION, self.slot);
            list.list.EndQuery(
                &self.stats,
                D3D12_QUERY_TYPE_PIPELINE_STATISTICS1,
                self.slot,
            );
        }
        if let Some(probe) = self.row {
            probe.copy(list, true);
        }
    }
}

impl Queries {
    pub(crate) fn resolve(&self, list: &ID3D12GraphicsCommandList) {
        let Some(pages) = self.0.get() else { return };
        for p in pages.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            let count = p.draws.len() as u32;
            // SAFETY: all counted slots were ended on this list. The readback
            // buffer has separate, aligned ranges for the two result types.
            unsafe {
                list.ResolveQueryData(
                    &p.stats,
                    D3D12_QUERY_TYPE_PIPELINE_STATISTICS1,
                    0,
                    count,
                    &p.buffer,
                    0,
                );
                list.ResolveQueryData(
                    &p.occlusion,
                    D3D12_QUERY_TYPE_OCCLUSION,
                    0,
                    count,
                    &p.buffer,
                    OCCLUSION_OFFSET,
                );
            }
        }
    }

    pub(crate) fn save(&self) {
        let Some(pages) = self.0.get() else { return };
        let pages = pages.lock().unwrap_or_else(|e| e.into_inner());
        if pages.is_empty() {
            return;
        }
        let Some(dir) = crate::capture::folder() else {
            return;
        };
        static FILE: Mutex<()> = Mutex::new(());
        let _guard = FILE.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("mesh-coverage.tsv"))
        else {
            return;
        };
        let header = file.metadata().is_ok_and(|m| m.len() == 0);
        let mut out = std::io::BufWriter::new(file);
        if header {
            let _ = writeln!(
                out,
                "frame\targuments\ttarget0\tdepth\tpipeline\tmesh_invocations\tmesh_primitives\tclip_in\tclip_out\tpixel_invocations\tpassed_samples\tchanged_x"
            );
        }
        for p in pages.iter() {
            let mut data = std::ptr::null_mut();
            // SAFETY: submission finished before this callback. Both arrays
            // are aligned, sized for SLOTS, and read only at resolved indices.
            unsafe {
                if p.buffer.Map(0, None, Some(&mut data)).is_err() || data.is_null() {
                    continue;
                }
                let samples = data.byte_add(OCCLUSION_OFFSET as usize).cast::<u64>();
                for (i, d) in p.draws.iter().enumerate() {
                    let s = &*data.cast::<Statistics>().add(i);
                    let passed = *samples.add(i);
                    let changed = if d.sampled_row {
                        row::changes(data.cast(), i)
                    } else {
                        "-".to_owned()
                    };
                    let _ = writeln!(
                        out,
                        "{}\t{:#x}\t{:#x}\t{:#x}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        d.frame,
                        d.args.unwrap_or(0),
                        d.target,
                        d.depth,
                        d.pipeline,
                        s.MSInvocations,
                        s.MSPrimitives,
                        s.CInvocations,
                        s.CPrimitives,
                        s.PSInvocations,
                        passed,
                        changed
                    );
                }
                p.buffer.Unmap(0, None);
            }
        }
    }

    pub(crate) fn reset(&self) {
        if let Some(pages) = self.0.get() {
            pages.lock().unwrap_or_else(|e| e.into_inner()).clear();
        }
    }
}
