//! Shader breadcrumbs: after a hung or removed device, which shaders' loops
//! ran away.
//!
//! With `VICTORY_SHADER_BREADCRUMBS=1` in a diagnostics build, translated
//! loops store their iteration counts (the largest any lane reached) in one
//! host buffer (see `victory_shader::emit::breadcrumbs`): `BREADCRUMB_SLOTS`
//! slots of `BREADCRUMB_LOOPS` words, a shader's slot taken from its
//! container's digest. The buffer lives in system memory (a shared buffer),
//! so what the GPU wrote survives a device removal, and is cleared before
//! each submission; the shader reaches it through a UAV in the last slot of
//! every view heap (see `HostHeap::breadcrumb_slot`). When the device is
//! removed the failure report names the stages whose loops counted most.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use victory_shader::emit::{BREADCRUMB_LOOPS, BREADCRUMB_SLOTS};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS;

/// The buffer's size in bytes.
const BYTES: u64 = BREADCRUMB_SLOTS as u64 * BREADCRUMB_LOOPS as u64 * 4;

/// How many of the slowest loops a report names.
const REPORTED: usize = 5;

/// Whether breadcrumbs are on (a diagnostics build, the switch set).
pub(crate) fn enabled() -> bool {
    victory_shader::emit::breadcrumbs()
}

/// The buffer and where it is mapped, made on first use.
fn buffer() -> Option<&'static (ID3D12Resource, usize)> {
    static BUFFER: OnceLock<Option<(ID3D12Resource, usize)>> = OnceLock::new();
    BUFFER
        .get_or_init(|| {
            let (buffer, data) = crate::host::host()?.shared_buffer(BYTES)?;
            Some((buffer, data as usize))
        })
        .as_ref()
}

/// The stages by slot, with the pipelines that use them.
fn stages() -> &'static Mutex<BTreeMap<u32, Vec<String>>> {
    static STAGES: Mutex<BTreeMap<u32, Vec<String>>> = Mutex::new(BTreeMap::new());
    &STAGES
}

/// Makes the buffer's UAV (a raw view of all of it) at `at`, a view heap's
/// breadcrumb entry.
pub(crate) fn describe(device: &ID3D12Device, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some((buffer, _)) = buffer() else { return };
    let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV {
                NumElements: (BYTES / 4) as u32,
                Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot inside a live heap and a live buffer.
    unsafe { device.CreateUnorderedAccessView(buffer, None, Some(&desc), at) };
}

/// Zeroes the counts, before a submission (every one is waited for).
pub(crate) fn clear() {
    if let Some(&(_, data)) = enabled().then(buffer).flatten() {
        // SAFETY: the buffer's own mapping, no submission running.
        unsafe { std::ptr::write_bytes(data as *mut u8, 0, BYTES as usize) };
    }
}

/// Remembers that `name`'s pipeline uses the shader of `container`.
pub(crate) fn note_stage(container: &[u8], name: &str) {
    let Some(digest) = container.get(4..20).and_then(|d| <&[u8; 16]>::try_from(d).ok()) else {
        return;
    };
    let mut stages = stages().lock().unwrap_or_else(|e| e.into_inner());
    let names = stages.entry(victory_shader::emit::breadcrumb_slot(digest)).or_default();
    if !names.iter().any(|n| n == name) {
        names.push(name.to_owned());
    }
}

/// The loops that counted most, as report lines (most first): the slot's
/// pipelines, the loop's number and its count. Pure, for testing.
fn summarize(words: &[u32], stages: &BTreeMap<u32, Vec<String>>) -> Vec<String> {
    let mut loops: Vec<(u32, usize, usize)> = words
        .iter()
        .enumerate()
        .filter(|&(_, &count)| count > 0)
        .map(|(i, &count)| (count, i / BREADCRUMB_LOOPS as usize, i % BREADCRUMB_LOOPS as usize))
        .collect();
    loops.sort_unstable_by(|a, b| b.cmp(a));
    loops
        .into_iter()
        .take(REPORTED)
        .map(|(count, slot, n)| {
            let names = stages.get(&(slot as u32)).map_or_else(
                || "a shader not seen".to_owned(),
                |names| names.join(" | "),
            );
            format!("loop {n} of slot {slot} ran {count} iterations ({names})")
        })
        .collect()
}

/// After a device removal: the loops that counted most in the last
/// submission, for the failure report. `None` with breadcrumbs off or no
/// loop that ran.
pub(crate) fn report() -> Option<String> {
    let &(_, data) = enabled().then(buffer).flatten()?;
    // SAFETY: the buffer's own mapping, `BYTES` long; the device is gone
    // or idle, nothing writes it any more.
    let words: Vec<u32> = (0..BYTES as usize / 4)
        .map(|i| unsafe { (data as *const u32).add(i).read_volatile() })
        .collect();
    let stages = stages().lock().unwrap_or_else(|e| e.into_inner());
    let lines = summarize(&words, &stages);
    (!lines.is_empty()).then(|| format!("shader breadcrumbs: {}", lines.join("; ")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_biggest_counts_are_named_with_their_pipelines() {
        let mut words = vec![0u32; (BREADCRUMB_SLOTS * BREADCRUMB_LOOPS) as usize];
        let at = |slot: u32, n: u32| (slot * BREADCRUMB_LOOPS + n) as usize;
        words[at(3, 0)] = 17;
        words[at(3, 2)] = 1_048_576;
        words[at(900, 7)] = 5;
        let mut stages = BTreeMap::new();
        stages.insert(3, vec!["pipeline A".to_owned(), "pipeline B".to_owned()]);
        let lines = summarize(&words, &stages);
        assert_eq!(
            lines,
            [
                "loop 2 of slot 3 ran 1048576 iterations (pipeline A | pipeline B)",
                "loop 0 of slot 3 ran 17 iterations (pipeline A | pipeline B)",
                "loop 7 of slot 900 ran 5 iterations (a shader not seen)",
            ]
        );
        assert!(summarize(&vec![0; words.len()], &stages).is_empty());
    }

    #[test]
    fn at_most_the_slowest_few_are_named() {
        let words: Vec<u32> = (1..=20).collect();
        assert_eq!(summarize(&words, &BTreeMap::new()).len(), REPORTED);
    }

    #[test]
    fn a_stage_is_found_by_its_containers_digest() {
        let mut container = b"DXBC".to_vec();
        container.extend((0..16).map(|b| b + 1));
        container.extend([0; 8]);
        note_stage(&container, "test pipeline");
        note_stage(&container, "test pipeline");
        let digest: [u8; 16] = std::array::from_fn(|i| i as u8 + 1);
        let slot = victory_shader::emit::breadcrumb_slot(&digest);
        assert!(slot < BREADCRUMB_SLOTS);
        let stages = stages().lock().unwrap();
        assert_eq!(stages[&slot], ["test pipeline"]);
    }
}
