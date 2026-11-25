//! Before/after depth at two texels on the same row. Only the full depth plane
//! can be copied from a DSV. Reuse one device-local staging buffer per list and
//! retain only the two floats per snapshot for readback after the fence.
use crate::host::{HostList, host, transition};
use crate::resource::HostTexture;
use std::io::Write;
use std::sync::{Arc, Mutex, OnceLock};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

struct Spec {
    width: u32,
    height: u32,
    x: [u32; 2],
    y: u32,
}

fn spec() -> Option<&'static Spec> {
    static SPEC: OnceLock<Option<Spec>> = OnceLock::new();
    SPEC.get_or_init(|| {
        let s = std::env::var("VICTORY_DEPTH_HISTORY").ok()?;
        let (size, coords) = s.split_once('@')?;
        let (width, height) = size.split_once('x')?;
        let mut coords = coords.split(',');
        let spec = Spec {
            width: width.parse().ok()?,
            height: height.parse().ok()?,
            x: [coords.next()?.parse().ok()?, coords.next()?.parse().ok()?],
            y: coords.next()?.parse().ok()?,
        };
        (coords.next().is_none() && spec.y < spec.height && spec.x.iter().all(|x| *x < spec.width))
            .then_some(spec)
    })
    .as_ref()
}

pub(super) fn source(resource: usize) -> Option<Arc<HostTexture>> {
    let s = spec()?;
    let copy = crate::resource::host_copy(resource as crate::object::This)?;
    // SAFETY: the Arc holds the resource.
    let d = unsafe { copy.texture.GetDesc() };
    (d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && d.Width == u64::from(s.width)
        && d.Height == s.height
        && d.MipLevels == 1
        && d.DepthOrArraySize == 1
        && d.SampleDesc.Count == 1
        && d.Format == DXGI_FORMAT_R32G8X24_TYPELESS)
        .then_some(copy)
}

#[derive(Clone)]
struct Stage {
    buffer: ID3D12Resource,
    layout: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    at: [u64; 2],
}

impl Stage {
    fn new(texture: &ID3D12Resource) -> Option<Self> {
        let host = host()?;
        let s = spec()?;
        let mut layout = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let (mut row_bytes, mut size) = (0, 0);
        // SAFETY: plane zero of the checked single-slice texture, valid outputs.
        unsafe {
            host.device.GetCopyableFootprints(
                &texture.GetDesc(),
                0,
                1,
                0,
                Some(&mut layout),
                None,
                Some(&mut row_bytes),
                Some(&mut size),
            )
        };
        let stride = row_bytes.checked_div(u64::from(s.width))?;
        if !matches!(stride, 4 | 8) || row_bytes != stride * u64::from(s.width) {
            return None;
        }
        let row = layout.Offset + u64::from(s.y) * u64::from(layout.Footprint.RowPitch);
        let at = s.x.map(|x| row + u64::from(x) * stride);
        Some(Self {
            buffer: host.uav_buffer(size)?,
            layout,
            at,
        })
    }
}

const PAGE: usize = 128;
const ENTRY_BYTES: u64 = 16;

struct Entry {
    frame: u64,
    resource: usize,
    label: String,
    args: u64,
}

#[derive(Default)]
struct Records {
    stage: Option<Stage>,
    pages: Vec<ID3D12Resource>,
    entries: Vec<Entry>,
}

#[derive(Default)]
pub(super) struct History(Mutex<Records>);

pub(crate) struct Probe {
    src: Arc<HostTexture>,
    state: D3D12_RESOURCE_STATES,
    stage: Stage,
    dest: ID3D12Resource,
    at: u64,
}

impl Probe {
    pub(crate) fn copy(&self, list: &HostList, after: bool) {
        let source = D3D12_RESOURCE_STATE_COPY_SOURCE;
        let dest = D3D12_RESOURCE_STATE_COPY_DEST;
        let common = D3D12_RESOURCE_STATE_COMMON;
        // SAFETY: all resources survive the submission. The scratch returns to
        // COMMON on each use, and the depth plane returns to its tracked state.
        unsafe {
            let src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&self.src.texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            let dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&self.stage.buffer),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: self.stage.layout,
                },
            };
            if self.state != source {
                list.list
                    .ResourceBarrier(&[transition(&self.src.texture, self.state, source)]);
            }
            list.list
                .ResourceBarrier(&[transition(&self.stage.buffer, common, dest)]);
            list.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            list.list
                .ResourceBarrier(&[transition(&self.stage.buffer, dest, source)]);
            for (i, offset) in self.stage.at.iter().enumerate() {
                list.list.CopyBufferRegion(
                    &self.dest,
                    self.at + u64::from(after) * 8 + i as u64 * 4,
                    &self.stage.buffer,
                    *offset,
                    4,
                );
            }
            list.list
                .ResourceBarrier(&[transition(&self.stage.buffer, source, common)]);
            if self.state != source {
                list.list
                    .ResourceBarrier(&[transition(&self.src.texture, source, self.state)]);
            }
        }
    }
}

impl History {
    pub(super) fn begin(
        &self,
        list: &HostList,
        src: Arc<HostTexture>,
        frame: u64,
        resource: usize,
        label: String,
        args: u64,
    ) -> Option<Probe> {
        let state = list.tracked_texture_state(&src)?;
        let mut r = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if r.stage.is_none() {
            r.stage = Some(Stage::new(&src.texture)?);
        }
        let i = r.entries.len() % PAGE;
        if i == 0 {
            r.pages
                .push(host()?.read_back_buffer(ENTRY_BYTES * PAGE as u64)?);
        }
        let probe = Probe {
            src,
            state,
            stage: r.stage.clone()?,
            dest: r.pages.last()?.clone(),
            at: i as u64 * ENTRY_BYTES,
        };
        r.entries.push(Entry {
            frame,
            resource,
            label,
            args,
        });
        probe.copy(list, false);
        Some(probe)
    }

    pub(super) fn save(&self) {
        let r = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if r.entries.is_empty() {
            return;
        }
        let Some(dir) = crate::capture::folder() else {
            return;
        };
        let Some(s) = spec() else { return };
        static OUTPUT: Mutex<()> = Mutex::new(());
        let _output = OUTPUT.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("depth-history.tsv"))
        else {
            return;
        };
        let header = file.metadata().is_ok_and(|m| m.len() == 0);
        let mut out = std::io::BufWriter::new(file);
        if header {
            let _ = writeln!(
                out,
                "frame\tdepth\tcommand\targuments\tx0\tx1\ty\tbefore0\tbefore1\tafter0\tafter1"
            );
        }
        for (entries, page) in r.entries.chunks(PAGE).zip(&r.pages) {
            let mut p = std::ptr::null_mut();
            // SAFETY: the fence has completed both copies in each 16-byte slot.
            unsafe {
                if page.Map(0, None, Some(&mut p)).is_err() || p.is_null() {
                    continue;
                }
                for (i, e) in entries.iter().enumerate() {
                    let [a, b, c, d] = *p.cast::<[f32; 4]>().add(i);
                    let _ = writeln!(
                        out,
                        "{}\t{:#x}\t{}\t{:#x}\t{}\t{}\t{}\t{a}\t{b}\t{c}\t{d}",
                        e.frame, e.resource, e.label, e.args, s.x[0], s.x[1], s.y
                    );
                }
                page.Unmap(0, None);
            }
        }
    }

    pub(super) fn reset(&self) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = Records::default();
    }
}
