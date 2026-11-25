//! Per-draw mesh, rasterizer and occlusion counters for selected capture frames.
//! Enable with VICTORY_MESH_STATS=1 and select frames as for the command trace.
//! VICTORY_MATERIAL_ROW=1280x720@160 also records which pixels each material
//! draw changes along row 160 of its first color target, including ordinary
//! vertex and indexed draws such as the game's dynamic decals.
//! VICTORY_RASTER_STATS accepts comma-separated shader hashes to also measure
//! raster draws on other surfaces, including depth-only shadow draws.
//! VICTORY_TRACE_DISPATCH measures the first dispatch of each selected shader
//! as well as its dispatches on listed frames. Some lighting jobs run only at load.
//! VICTORY_COMPUTE_MASK=slot+dword@x also reads one R8_UNORM texel at x
//! on the configured material row before and after those dispatches.
//! With `@first-last` that column holds the two inclusive R8 spans as hex.
//! VICTORY_DEPTH_HISTORY=1280x720@352,364,310 records the two D32 depths
//! at (352,310) and (364,310) around draws and depth clears on listed frames.
//! Use `1280x5120/d16@x0,x1,y` to trace an unsigned 16-bit shadow atlas.

use std::io::Write;
use std::sync::{Mutex, OnceLock};
use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};

mod depth;
mod input;
mod row;
pub(crate) use input::Arguments;

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
    mask_x: Option<[u32; 2]>,
    input: Option<input::Snapshot>,
    dispatch: Option<input::Counts>,
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
pub(crate) struct Queries(OnceLock<Mutex<Vec<Page>>>, depth::History);

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
    depth_probe: Option<depth::Probe>,
}

pub(crate) fn begin_clear(list: &HostList, resource: usize, description: impl FnOnce() -> String) -> Option<depth::Probe> {
    let frame = crate::queue::presented() + 1;
    if !enabled() || !crate::capture::listed(frame) { return None; }
    list.mesh_queries.1.begin(list, depth::source(resource)?, frame, resource, description(), 0)
}

pub(crate) enum Command<'a> {
    Mesh(Option<Arguments<'a>>),
    Raster(Option<u64>),
    Compute(Option<Arguments<'a>>, Option<(usize, [u32; 2])>),
}

pub(crate) fn compute_pixels() -> Option<(u8, u32, [u32; 2])> {
    static SPEC: OnceLock<Option<(u8, u32, [u32; 2])>> = OnceLock::new();
    *SPEC.get_or_init(|| {
        let s = std::env::var("VICTORY_COMPUTE_MASK").ok()?;
        let (view, x) = s.split_once('@')?;
        let (slot, dword) = view.split_once('+')?;
        let (first, last) = x.split_once('-').unwrap_or((x, x));
        let range = [first.parse().ok()?, last.parse().ok()?];
        (range[0] <= range[1]).then_some((slot.parse().ok()?, dword.parse().ok()?, range))
    })
}

pub(crate) fn begin(
    list: &HostList,
    pipeline: usize,
    command: Command<'_>,
    targets: &[usize],
    depth: usize,
    heap: usize,
) -> Option<Active> {
    if !enabled() {
        return None;
    }
    let frame = crate::queue::presented() + 1;
    let listed = crate::capture::listed(frame);
    if !listed && !matches!(command, Command::Compute(..)) {
        return None;
    }
    // The same material target is bound with six attachments for opaque
    // geometry and decals, then with three for other surface passes.
    let row_target = targets.first().copied().filter(|&target| row::requested(target));
    let p = crate::pipeline::host_pipeline(pipeline as crate::object::This)?;
    if matches!(command, Command::Compute(..)) {
        static SEEN: OnceLock<Mutex<std::collections::HashSet<String>>> = OnceLock::new();
        let mut seen = SEEN.get_or_init(Mutex::default).lock().unwrap_or_else(|e| e.into_inner());
        if seen.contains(&p.name) {
            if !listed { return None }
        } else {
            seen.insert(p.name.clone());
        }
    }
    let depth_source = if listed && depth != 0 { depth::source(depth) } else { None };
    if matches!(command, Command::Raster(_)) && row_target.is_none() && depth_source.is_none() {
        static SHADERS: OnceLock<Vec<String>> = OnceLock::new();
        let shaders = SHADERS.get_or_init(|| {
            std::env::var("VICTORY_RASTER_STATS")
                .unwrap_or_default()
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect()
        });
        if !p.stages.iter().any(|stage| shaders.contains(&stage.shader.hash)) {
            return None;
        }
    }
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
    let mask = match &command {
        Command::Compute(_, output) => *output,
        _ => None,
    };
    let mask_x = mask.map(|(_, x)| x);
    let row = row_target.or(mask.map(|(r, _)| r))
        .and_then(|target| row::Probe::new(target, &page.buffer, index, mask_x));
    if let Some(probe) = &row {
        probe.copy(list, false);
    }
    let (args, input, dispatch) = match command {
        Command::Mesh(args) => {
            let input = input::capture(list, p, heap);
            let dispatch = input.as_ref().and(args.as_ref()).and_then(|a| a.capture(list));
            (args.map(|a| a.address), input, dispatch)
        }
        Command::Raster(address) => (address, None, None),
        Command::Compute(args, _) => {
            let dispatch = args.as_ref().and_then(|a| a.capture(list));
            (args.map(|a| a.address), None, dispatch)
        }
    };
    let depth_probe = depth_source.and_then(|src| list.mesh_queries.1.begin(
        list, src, frame, depth, p.name.clone(), args.unwrap_or(0)));
    page.draws.push(Draw {
        frame,
        args,
        target: targets.first().copied().or(mask.map(|(r, _)| r)).unwrap_or(0),
        depth,
        pipeline: p.name.clone(),
        sampled_row: row.is_some() && mask_x.is_none(),
        mask_x: mask_x.filter(|_| row.is_some()),
        input,
        dispatch,
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
        depth_probe,
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
        if let Some(probe) = self.depth_probe {
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
        self.1.save();
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
                "frame\targuments\ttarget0\tdepth\tpipeline\tmesh_invocations\tmesh_primitives\tclip_in\tclip_out\tpixel_invocations\tpassed_samples\tchanged_x\theap_input\tdispatch_source_submitted_count\tcompute_invocations\tmask_before_after"
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
                        "{}\t{:#x}\t{:#x}\t{:#x}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
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
                        changed,
                        d.input.as_ref().map_or_else(|| "-".into(), input::Snapshot::words),
                        d.dispatch.as_ref().map_or_else(|| "-".into(), input::Counts::words),
                        s.CSInvocations,
                        d.mask_x.map_or_else(|| "-".into(), |x| row::mask_pixels(data.cast(), i, x)),
                    );
                }
                p.buffer.Unmap(0, None);
            }
        }
    }

    pub(crate) fn reset(&self) {
        self.1.reset();
        if let Some(pages) = self.0.get() {
            pages.lock().unwrap_or_else(|e| e.into_inner()).clear();
        }
    }
}
