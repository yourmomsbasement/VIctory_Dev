//! Compare a row of the first material target immediately before and after
//! each mesh command. Copies on the same command list retain draw ordering.

use std::sync::{Arc, OnceLock};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::host::HostList;
use crate::resource::HostTexture;

const START: u64 = (super::OCCLUSION_OFFSET + super::SLOTS as u64 * 8)
    .next_multiple_of(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT as u64);

struct Row {
    width: u32,
    height: u32,
    y: u32,
    pitch: u32,
    stride: u64,
}

fn selected() -> Option<&'static Row> {
    static ROW: OnceLock<Option<Row>> = OnceLock::new();
    ROW.get_or_init(|| {
        let value = std::env::var("VICTORY_MATERIAL_ROW").ok()?;
        let (size, y) = value.split_once('@')?;
        let (w, h) = size.split_once('x')?;
        let (width, height, y) = (
            w.parse::<u32>().ok()?,
            h.parse::<u32>().ok()?,
            y.parse::<u32>().ok()?,
        );
        if !(1..=D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION).contains(&width)
            || height > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION
            || y >= height
        {
            return None;
        }
        let pitch = (width * 4).next_multiple_of(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
        let stride =
            u64::from(pitch).next_multiple_of(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT.into());
        tracing::info!(width, height, y, "capturing material changes along a row");
        Some(Row {
            width,
            height,
            y,
            pitch,
            stride,
        })
    })
    .as_ref()
}

pub(super) fn requested(resource: usize) -> bool {
    selected().is_some_and(|row| {
        crate::resource::texture_extent(resource as crate::object::This)
            == Some((u64::from(row.width), row.height))
    })
}

pub(super) fn page_bytes() -> u64 {
    selected().map_or(super::OCCLUSION_OFFSET + super::SLOTS as u64 * 8, |r| {
        START + 2 * u64::from(super::SLOTS) * r.stride
    })
}

pub(super) struct Probe {
    row: &'static Row,
    offset: u64,
    source: Arc<HostTexture>,
    dest: ID3D12Resource,
    format: DXGI_FORMAT,
    restore: D3D12_RESOURCE_STATES,
}

impl Probe {
    pub(super) fn new(resource: usize, dest: &ID3D12Resource, slot: u32, compute_x: Option<[u32; 2]>) -> Option<Self> {
        let row = selected()?;
        let source = crate::resource::host_copy(resource as crate::object::This)?;
        // SAFETY: the host texture is held for both copies.
        let desc = unsafe { source.texture.GetDesc() };
        if desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
            || desc.Width != row.width.into()
            || desc.Height != row.height
            || desc.MipLevels != 1
            || desc.DepthOrArraySize != 1
            || desc.SampleDesc.Count != 1
        {
            return None;
        }
        let format = match (compute_x, desc.Format) {
            (Some([first, last]), DXGI_FORMAT_R8_UNORM) if first <= last && last < row.width => DXGI_FORMAT_R8_UNORM,
            (None, DXGI_FORMAT_R8G8B8A8_TYPELESS | DXGI_FORMAT_R8G8B8A8_UNORM) => {
                DXGI_FORMAT_R8G8B8A8_UNORM
            }
            (None, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB) => desc.Format,
            _ => return None,
        };
        Some(Self {
            row,
            offset: START + 2 * u64::from(slot) * row.stride,
            source,
            dest: dest.clone(),
            format,
            restore: if compute_x.is_some() {
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS
            } else {
                D3D12_RESOURCE_STATE_RENDER_TARGET
            },
        })
    }

    pub(super) fn copy(&self, list: &HostList, after: bool) {
        let r = self.row;
        let src = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: the locations borrow these COM pointers for the call.
            pResource: unsafe { std::mem::transmute_copy(&self.source.texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let dest = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: the page owns the readback buffer until the fence completes.
            pResource: unsafe { std::mem::transmute_copy(&self.dest) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: self.offset + u64::from(after) * r.stride,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: self.format,
                        Width: r.width,
                        Height: 1,
                        Depth: 1,
                        RowPitch: r.pitch,
                    },
                },
            },
        };
        let region = D3D12_BOX {
            left: 0,
            top: r.y,
            front: 0,
            right: r.width,
            bottom: r.y + 1,
            back: 1,
        };
        list.settle(&self.source, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
        // SAFETY: a single row in the sole subresource and a disjoint aligned destination.
        unsafe {
            list.list
                .CopyTextureRegion(&dest, 0, 0, 0, &src, Some(&region))
        };
        list.settle(&self.source, self.restore.0 as u32);
    }
}

/// The slot must have both copies completed in this mapped page.
pub(super) unsafe fn mask_pixels(data: *const u8, slot: usize, [first, last]: [u32; 2]) -> String {
    let row = selected().expect("the captured mask has a configured row");
    let offset = START as usize + slot * 2 * row.stride as usize + first as usize;
    let length = (last - first + 1) as usize;
    // SAFETY: Probe::new checked both ends against the single-channel row.
    let [before, after] = [offset, offset + row.stride as usize].map(|at| unsafe {
        std::slice::from_raw_parts(data.add(at), length)
    });
    if length == 1 {
        return format!("{} {}", before[0], after[0]);
    }
    let hex = |bytes: &[u8]| -> String { bytes.iter().map(|b| format!("{b:02x}")).collect() };
    format!("{first}-{last} {} {}", hex(before), hex(after))
}

/// The slot must have both RGBA row copies completed in this mapped page.
pub(super) unsafe fn changes(data: *const u8, slot: usize) -> String {
    let r = selected().expect("a captured row has a configured size");
    let start = START as usize + slot * 2 * r.stride as usize;
    // SAFETY: the readback page contains both aligned width-element arrays.
    let (before, after) = unsafe {
        (
            std::slice::from_raw_parts(data.add(start).cast::<u32>(), r.width as usize),
            std::slice::from_raw_parts(
                data.add(start + r.stride as usize).cast::<u32>(),
                r.width as usize,
            ),
        )
    };
    changed_ranges(before, after)
}

fn changed_ranges(before: &[u32], after: &[u32]) -> String {
    let mut ranges = Vec::new();
    let mut x = 0;
    while x < before.len() {
        if before[x] == after[x] {
            x += 1;
            continue;
        }
        let first = x;
        x += 1;
        while x < before.len() && before[x] != after[x] {
            x += 1;
        }
        ranges.push(if x == first + 1 {
            first.to_string()
        } else {
            format!("{first}-{}", x - 1)
        });
    }
    if ranges.is_empty() {
        ".".to_owned()
    } else {
        ranges.join(",")
    }
}

#[cfg(test)]
mod tests {
    use super::changed_ranges;

    #[test]
    fn row_changes_preserve_disjoint_and_edge_pixels() {
        let before = [0; 8];
        assert_eq!(
            changed_ranges(&before, &[1, 1, 0, 1, 0, 0, 1, 1]),
            "0-1,3,6-7"
        );
        assert_eq!(changed_ranges(&before, &before), ".");
    }
}
