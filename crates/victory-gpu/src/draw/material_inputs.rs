//! Guest constants and selected table inputs for one draw at command recording.
use super::{Root, State, View};
use std::fmt::Write;
use std::sync::OnceLock;
use victory_shader::{ResourceType, RootType, UsedResource};

pub(super) fn selected(shader: &str) -> bool {
    static SHADERS: OnceLock<Vec<String>> = OnceLock::new();
    SHADERS
        .get_or_init(|| {
            std::env::var("VICTORY_TRACE_MATERIAL_INPUTS")
                .unwrap_or_default()
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .iter()
        .any(|s| s == shader)
}

fn save_table(slot: u8, entry: u32) -> bool {
    static TABLES: OnceLock<Vec<(u8, u32)>> = OnceLock::new();
    TABLES
        .get_or_init(|| {
            std::env::var("VICTORY_TRACE_MATERIAL_BUFFERS")
                .unwrap_or_default()
                .split(',')
                .filter_map(|s| {
                    let (slot, entry) = s.trim().split_once('+')?;
                    Some((slot.parse().ok()?, entry.parse().ok()?))
                })
                .collect()
        })
        .contains(&(slot, entry))
}

impl State {
    /// Keep complete bit patterns, including integer fields packed into CBVs.
    pub(super) fn describe_material_inputs(
        &self,
        text: &mut String,
        used: &[UsedResource],
        compute: bool,
    ) {
        let Some(folder) = crate::capture::folder() else {
            return;
        };
        for u in used {
            if u.root == RootType::Constants {
                if let Some(Root::Constants(words)) =
                    self.roots_ref(compute).get(usize::from(u.slot))
                {
                    let _ = write!(
                        text,
                        " input s{}+{} root={words:08x?}",
                        u.slot, u.table_offset_dwords
                    );
                }
                continue;
            }
            if u.root != RootType::Table {
                continue;
            }
            let Some(view) = self.table_view(compute, u.slot, u.table_offset_dwords) else {
                continue;
            };
            let range = match view {
                View::Constant(address, size) => Some((address, u64::from(size))),
                _ if u.kind == ResourceType::Srv => {
                    crate::descriptor::host::backed_view_range(&view)
                }
                _ => None,
            };
            let Some((address, size @ 1..=1_048_576)) = range else {
                continue;
            };
            if size > 4096 && !save_table(u.slot, u.table_offset_dwords) {
                continue;
            }
            let mut bytes = vec![0; size as usize];
            let _ = write!(
                text,
                " input s{}+{} {address:#x}=",
                u.slot, u.table_offset_dwords
            );
            if !crate::host::read_into(address as *const u8, &mut bytes) {
                text.push_str("unreadable");
                continue;
            }
            if size > 4096 {
                match crate::capture::snapshot_material(folder, &bytes) {
                    Ok(name) => text.push_str(&name),
                    Err(_) => text.push_str("write-failed"),
                }
            } else {
                for byte in bytes {
                    let _ = write!(text, "{byte:02x}");
                }
            }
        }
    }
}
