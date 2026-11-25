//! Title memory by address, for translated shaders that read and write
//! through raw 40-bit addresses (`victory_shader::emit::SLOT_MEMORY`: the
//! console's ray-tracing structure builder and the programs that walk its
//! structures; the structures themselves lie in plain title memory, in no
//! buffer object).
//!
//! The copies are the ones every other access uses ([`crate::shadow`]: a
//! title buffer's copy, or the 4 MiB window of memory outside buffers), so
//! a root view and a raw address reach the same host bytes. This module
//! keeps what a shader needs to find them:
//!
//! - the **directory**: per 64 KiB page of title address space the number
//!   of the copy that holds the page (0: none) and the page's byte offset
//!   in it; one buffer for all lists, entries only ever go from empty to
//!   filled;
//! - per view heap a run of slots with a raw unordered access view of each
//!   numbered copy (slot 0 of the run: a null view, for pages without a
//!   copy);
//! - the **faults**: the page numbers shaders found empty, read back here
//!   once a frame, so the copies exist from the next frame on. A pointer a
//!   shader takes out of memory cannot be known before it runs; the
//!   addresses a dispatch is given as root views and table-view addresses
//!   are made ready before it.
//!
//! **A page that is not there when a shader touches it is a lost access**:
//! the read gives 0, the store is dropped, and the fault note only helps
//! the next frame. For the structure builder that is the same as a dropped
//! stage, so before a dispatch the host makes ready everything it can see
//! the dispatch name: the whole copy around each root view, around each
//! pair of root constants that is a title address and around each root
//! constant that is one in 256-byte units (the builder passes structures
//! all three ways), and the 4 MiB after each (a structure may cross a
//! window's end).
//! shortcut: what a stage reaches only through pointers it reads out of
//! memory (InitHeaders' table of structure addresses, LoadInstances'
//! instance records) is not made ready unless something else named it;
//! the fault count in the log says whether that happens. The whole answer
//! is the extent of the title's allocation around an address, which the
//! memory module of the layer knows and this one does not ask yet.
//!
//! shortcut: a copy that is numbered is kept for the rest of the run, and
//! there are at most [`COPIES`] of them (4 MiB windows mostly: 4 GiB at
//! most). Every dispatch of such a shader brings all numbered copies up to
//! date and, if it can store, declares all of them written (the bytes are
//! compared on the CPU afterwards, as for any unordered access view): a
//! dirty-page mask written by the shader would bound both.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use victory_shader::emit::{MEMORY_FAULTS, MEMORY_FAULT_SUMMARY, MEMORY_PAGES};
use windows::Win32::Graphics::Direct3D12::*;

use crate::descriptor::host::HostHeap;
use crate::host::{HostList, host};
use crate::shadow::Shadow;

/// Copies that can be numbered (the run of heap slots is one longer). A
/// listed frame of the opening cutscene names about 50 windows.
const COPIES: u32 = 1023;
/// A window of memory outside buffers (`shadow.rs`).
const CELL: u64 = 4 << 20;
/// A directory page.
const PAGE: u64 = 1 << 16;

struct Memory {
    /// Buffer, its mapping, its GPU address.
    directory: (ID3D12Resource, *mut u32, u64),
    faults: (ID3D12Resource, *mut u32, u64),
    seen_faults: Vec<u32>,
    /// Copy `n` is `copies[n - 1]`.
    copies: Vec<Arc<Shadow>>,
    /// Per host view heap (by address): the first slot of its run, and how
    /// many copies have their view there.
    heaps: HashMap<usize, (u32, u32)>,
    /// The frame the faults were last read in.
    frame: u64,
    /// Pages asked for this frame that are not the title's memory.
    refused: std::collections::HashSet<u64>,
}

// SAFETY: the mappings are of buffers the struct owns; all access is under
// the mutex.
unsafe impl Send for Memory {}

static MEMORY: Mutex<Option<Memory>> = Mutex::new(None);

/// What a draw binds.
pub(crate) struct Ready {
    pub(crate) directory: u64,
    pub(crate) faults: u64,
    /// The view heap slot of the null view; copy `n`'s view is `n` further.
    pub(crate) base: u32,
}

impl Memory {
    fn new() -> Option<Memory> {
        let host = host()?;
        let make = |dwords: u64, what: &str| {
            let (buffer, data) = host.shared_buffer(dwords * 4)?;
            crate::host::name(&buffer, || format!("title memory by address: {what}"));
            // SAFETY: a fresh mapping of at least that many bytes.
            unsafe { std::ptr::write_bytes(data, 0, (dwords * 4) as usize) };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some((buffer, data.cast::<u32>(), gpu))
        };
        tracing::info!("title memory by address: a shader reads or writes through raw addresses");
        Some(Memory {
            directory: make(u64::from(MEMORY_PAGES) * 2, "directory")?,
            faults: make(u64::from(MEMORY_FAULTS), "faults")?,
            seen_faults: vec![0; (MEMORY_FAULTS - MEMORY_FAULT_SUMMARY) as usize],
            copies: Vec::new(),
            heaps: HashMap::new(),
            frame: u64::MAX,
            refused: Default::default(),
        })
    }

    /// Gives the page of `address` its copy, if it has none and the memory
    /// is the title's.
    fn map(&mut self, address: u64) {
        let page = address / PAGE;
        // The last entry stays empty: addresses past the directory read it.
        if page >= u64::from(MEMORY_PAGES) - 1 {
            return;
        }
        // SAFETY: inside the directory's mapping.
        let entry = unsafe { self.directory.1.add(page as usize * 2) };
        // SAFETY: as above.
        if unsafe { entry.read() } != 0 {
            return;
        }
        // Each registered copy published every whole page it covers. An
        // empty entry now needs a new copy; avoid allocating its backing
        // storage when there is no descriptor available for it.
        if self.copies.len() == COPIES as usize {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(COPIES, "title memory by address: no number left for another copy; its pages read zeros");
            });
            return;
        }
        let start = page * PAGE;
        // A shader's pointer, or a constant taken for an address, may be
        // anything: only memory the title has.
        if self.refused.contains(&page) {
            return;
        }
        if !crate::texture::committed(start as *const u8, PAGE as usize) {
            self.refused.insert(page);
            return;
        }
        // The copy a root view at that address would get; a buffer that
        // ends inside the page is passed over for what holds all of it.
        let whole = |s: &Arc<Shadow>| s.offset_in(start).is_some_and(|_| s.left(start) >= PAGE);
        let Some(shadow) = crate::shadow::find(start, 1)
            .filter(whole)
            .or_else(|| crate::shadow::find(start, PAGE).filter(whole))
        else {
            return;
        };
        // SAFETY: a live buffer.
        let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
        if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
            return;
        }
        let n = match self.copies.iter().position(|c| Arc::ptr_eq(c, &shadow)) {
            Some(n) => n,
            None => {
                self.copies.push(shadow.clone());
                let n = self.copies.len() - 1;
                // Every page the copy holds whole, not only the one asked
                // for: a structure is more than 64 KiB.
                // shortcut: a page of a title buffer inside this one that
                // has a copy of its own is given this copy too.
                let (first, len) = shadow.span();
                let mut at = first.next_multiple_of(PAGE);
                while at + PAGE <= first + len && at / PAGE < u64::from(MEMORY_PAGES) - 1 {
                    // SAFETY: inside the directory's mapping.
                    let other = unsafe { self.directory.1.add((at / PAGE) as usize * 2) };
                    // SAFETY: as above; the offset first, as below.
                    unsafe {
                        if other.read() == 0 {
                            other.add(1).write_volatile((at - first) as u32);
                            other.write_volatile(n as u32 + 1);
                        }
                    }
                    at += PAGE;
                }
                n
            }
        };
        let offset = shadow.offset_in(start).unwrap_or(0);
        // The offset first: a shader that sees the number sees it too.
        // SAFETY: inside the directory's mapping.
        unsafe {
            entry.add(1).write_volatile(offset as u32);
            entry.write_volatile(n as u32 + 1);
        }
    }

    /// GPU writers only set bits. If a submission is still writing while we
    /// scan, its additions remain visible on the next scan. Previously seen
    /// pages that are still missing get another mapping attempt in rotation.
    fn take_faults(&mut self) {
        let faults = self.faults.1;
        let directory = self.directory.1;
        let mut pages = fault_pages(&mut self.seen_faults, self.frame, |index| {
            // SAFETY: the two bitmap levels occupy MEMORY_FAULTS mapped dwords.
            unsafe { faults.add(index).read_volatile() }
        });
        pages.retain(|&p| {
            // SAFETY: one directory entry per bitmap bit. A root argument
            // or an earlier fault may already have mapped the whole copy.
            unsafe { directory.add(p as usize * 2).read() == 0 }
        });
        if let Some(&first) = pages.first() {
            for &page in &pages {
                self.map(u64::from(page) * PAGE);
            }
            static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 50 {
                let mapped = pages.iter().filter(|&&p| {
                    // SAFETY: the same bounded entries inspected above.
                    unsafe { directory.add(p as usize * 2).read() != 0 }
                }).count();
                tracing::warn!(
                    pages = pages.len(),
                    mapped,
                    copies = self.copies.len(),
                    first = format_args!("{:#x}", u64::from(first) * PAGE),
                    "title memory by address: shaders requested missing pages"
                );
            }
        }
    }

    /// What a dispatch names: the copy around `address` and the 4 MiB
    /// after it.
    fn name(&mut self, address: u64) {
        self.map(address);
        self.map((address & !(CELL - 1)) + CELL);
    }

    /// A shader takes the address of a known table view. Register copies
    /// up to its end, stopping at the first page without usable backing.
    fn name_view(&mut self, address: u64, len: u64) {
        let limit = (u64::from(MEMORY_PAGES) - 1) * PAGE;
        let end = address.saturating_add(len).min(limit);
        let mut at = address;
        while at < end {
            self.map(at);
            // SAFETY: at is below the directory limit.
            let copy = unsafe { self.directory.1.add((at / PAGE) as usize * 2).read() };
            if copy == 0 { break }
            let (base, bytes) = self.copies[copy as usize - 1].span();
            // map published all whole pages in this copy.
            at = (base + bytes) & !(PAGE - 1);
        }
    }
}

fn fault_pages(seen: &mut [u32], frame: u64, read: impl Fn(usize) -> u32) -> Vec<u32> {
    let summary_words = seen.len().div_ceil(32);
    let mut pages = Vec::new();
    for index in 0..summary_words {
        let mut summary = read(index);
        while summary != 0 {
            let word = index * 32 + summary.trailing_zeros() as usize;
            summary &= summary - 1;
            let Some(previous) = seen.get_mut(word) else { continue };
            let bits = read(summary_words + word);
            let mut pending = if word as u64 % 64 == frame % 64 { bits } else { bits & !*previous };
            *previous |= bits;
            while pending != 0 {
                pages.push(word as u32 * 32 + pending.trailing_zeros());
                pending &= pending - 1;
            }
        }
    }
    pages
}

/// Numbered copies this command list has already scheduled for refresh or
/// copy-back. Other recording lists may append to the shared directory.
#[derive(Default)]
pub(crate) struct Coverage {
    reads: usize,
    writes: usize,
}

impl Coverage {
    pub(crate) fn advance(&mut self, count: usize, writes: bool) -> (Range<usize>, Range<usize>) {
        let reads = self.reads..count;
        self.reads = count;
        let writes = if writes {
            std::mem::replace(&mut self.writes, count)..count
        } else {
            count..count
        };
        (reads, writes)
    }
}

/// Before a draw whose shader reads title memory by address: what its
/// root arguments name gets its copies (root views, table-view ranges
/// and each block of root constants, see the module), every numbered copy
/// has its view in `heap`'s run and is brought up to date when `list` is
/// submitted (and, with `writes`, copied back after). None if the host
/// cannot hold it; the draw is then dropped.
pub(crate) fn ready(
    list: &HostList,
    heap: &HostHeap,
    addresses: &[u64],
    constant_blocks: &[&[u32]],
    table_ranges: &[(u64, u64)],
    writes: bool,
) -> Option<Ready> {
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Memory::new();
    }
    let m = guard.as_mut()?;
    let frame = crate::frame::current();
    if m.frame != frame {
        m.frame = frame;
        m.refused.clear();
        m.take_faults();
    }
    for &address in addresses {
        m.name(address);
    }
    for &(address, len) in table_ranges {
        m.name_view(address, len);
    }
    // A title address in two constants (low dword, then bits 32 to 39), or
    // in one in 256-byte units. A constant that only looks like one names
    // memory nothing reads; one that is not the title's memory is refused.
    for &constants in constant_blocks {
        for (i, &c) in constants.iter().enumerate() {
            if let Some(&high) = constants.get(i + 1)
                && (1..=0xFF).contains(&high) && c >= 0x1_0000
            {
                m.name(u64::from(c) | u64::from(high) << 32);
            }
            if c >= 0x0100_0000 {
                m.name(u64::from(c) << 8);
            }
        }
    }
    let key = std::ptr::from_ref(heap) as usize;
    let (base, made) = match m.heaps.get(&key) {
        Some(&run) => run,
        None => {
            // Held for the heap's life: not given back with a list's slots.
            let base = heap.dynamic(COPIES + 1)?;
            heap.null_raw_uav_at(base);
            (base, 0)
        }
    };
    for (n, shadow) in m.copies.iter().enumerate().skip(made as usize) {
        let (_, len) = shadow.span();
        heap.raw_uav_at(base + 1 + n as u32, &shadow.buffer, 0, len);
    }
    m.heaps.insert(key, (base, m.copies.len() as u32));
    let (reads, writes) = list.memory_coverage(m.copies.len(), writes);
    for shadow in &m.copies[reads] {
        let (start, len) = shadow.span();
        shadow.refresh(list, start, len);
    }
    for shadow in &m.copies[writes] {
        let (start, len) = shadow.span();
        shadow.gpu_writes(list, start, len);
    }
    const KEEP_DIRECTORY: usize = usize::MAX - 0x6d;
    if list.first_use(KEEP_DIRECTORY) {
        list.keep(&m.directory.0);
        list.keep(&m.faults.0);
    }
    Some(Ready {
        directory: m.directory.2,
        faults: m.faults.2,
        base,
    })
}

#[cfg(test)]
mod tests {
    use super::{Coverage, fault_pages};

    #[test]
    fn faults_survive_late_gpu_writes_and_unresolved_pages_are_retried() {
        let mut seen = [0; 64];
        let mut bits = [0u32; 66];
        bits[0] = 1 << 3;
        bits[2 + 3] = 1 << 7;
        assert_eq!(fault_pages(&mut seen, 0, |w| bits[w]), [103]);
        bits[2 + 3] |= 1 << 9;
        // Leaf for word 33 is already visible, but its summary bit is late.
        bits[2 + 33] = 1 << 31;
        assert_eq!(fault_pages(&mut seen, 1, |w| bits[w]), [105]);
        bits[1] = 1 << 1;
        assert_eq!(fault_pages(&mut seen, 2, |w| bits[w]), [1087]);
        assert_eq!(fault_pages(&mut seen, 3, |w| bits[w]), [103, 105]);
    }

    #[test]
    fn interleaved_lists_cover_new_copies_and_promote_reads_to_writes() {
        let mut first = Coverage::default();
        let mut second = Coverage::default();
        assert_eq!(first.advance(2, true), (0..2, 0..2));
        assert_eq!(second.advance(4, false), (0..4, 4..4));
        assert_eq!(first.advance(4, false), (2..4, 4..4));
        assert_eq!(first.advance(4, true), (4..4, 2..4));
        assert_eq!(second.advance(5, true), (4..5, 0..5));
        assert_eq!(second.advance(5, true), (5..5, 5..5));
    }
}
