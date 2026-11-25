//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection. Blocks found
//! wholly uncommitted are skipped while they remain so; one query can
//! confirm a whole region, including ordinary reservations between views.
//!
//! shortcut: writes into physical pages through another view (xmem's
//! `write_physical`, DirectStorage's page destinations) are not seen at
//! the guarded addresses; they go to the texture pool, whose textures are
//! hashed, not guarded. A system call writing into a guarded block fails
//! (none is known to).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, MutexGuard};

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
    /// Page ranges asked ahead for a submission's refreshes ([`ask_ahead`])
    /// and the asking that did: `writes` answers for a range inside one
    /// from the journal, without another kernel call.
    ahead: Vec<(usize, usize)>,
    ahead_asked: u64,
    /// The ahead asking stands for a whole submission ([`ask_once`]): no
    /// other ahead asking is made, and `ahead` stays, until
    /// [`ask_once_end`].
    kept: bool,
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    found: Vec::new(),
    ahead: Vec::new(),
    ahead_asked: 0,
    kept: false,
});

/// Ranges of one submission closer than this are asked about together:
/// walking the pages between them costs less than another kernel call (a
/// call that resets written pages costs 30-70 microseconds whatever the
/// range).
const AHEAD_GAP: usize = (8 << 20) / PAGE;
/// The most one asking covers (its answer needs 8 bytes per 4 KiB page).
const AHEAD_MAX: usize = (1 << 30) / PAGE;

/// Asks the kernel about many ranges `(start, end)` at once, before the
/// [`writes`] of each: ranges close together are one asking, and `writes`
/// then answers from the journal until [`ask_ahead_done`]. The world's
/// submissions refresh thousands of host copies (6,664 askings a frame,
/// 456 ms, in run 14). Ranges that are not write-watched, or lie alone, are
/// left to `writes`.
pub(crate) fn ask_ahead(ranges: &[(usize, usize)]) {
    let pieces = page_ranges(ranges);
    if pieces.len() < 2 {
        return;
    }
    {
        let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
        // The submission's one asking stands ([`ask_once`]): what it does not
        // hold is left to `writes`.
        if j.kept {
            return;
        }
        j.ahead.clear();
    }
    let asked = asked_runs(ahead_runs(&pieces));
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    j.ahead.clear();
    // Numbered once the kernel has answered: a page an asking made meanwhile
    // found written has a lower number, and the copies read memory after this.
    j.asked += 1;
    j.ahead_asked = j.asked;
    note_asked(j, asked);
}

/// `ranges` (byte ranges) as page ranges, the empty ones left out, sorted.
fn page_ranges(ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut pieces: Vec<(usize, usize)> = ranges
        .iter()
        .filter(|r| r.0 < r.1)
        .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
        .collect();
    pieces.sort_unstable();
    pieces
}

/// `pieces` (page ranges, sorted) as the runs one kernel asking each takes:
/// those closer than [`AHEAD_GAP`] together, up to [`AHEAD_MAX`].
fn ahead_runs(pieces: &[(usize, usize)]) -> Vec<Vec<(usize, usize)>> {
    let mut runs: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut i = 0;
    while i < pieces.len() {
        let first = pieces[i].0;
        let mut last = pieces[i].1;
        let mut k = i + 1;
        while k < pieces.len()
            && pieces[k].0 <= last + AHEAD_GAP
            && last.max(pieces[k].1) - first <= AHEAD_MAX
        {
            last = last.max(pieces[k].1);
            k += 1;
        }
        runs.push(pieces[i..k].to_vec());
        i = k;
    }
    runs
}

/// Whether the kernel askings of a submission's ranges run on the threads
/// kept for the snapshots (`crate::shadow::wide_run`) rather than one after
/// another on the submitting thread (`VICTORY_WIDE_ASK`, on by default; `0`
/// asks one by one).
///
/// Each asking is a reservation of its own (the title's allocations are),
/// some 60 a submission and 2,744 a frame in a world scene, 35.7 ms a
/// frame of the frame line's "copies". Askings of different reservations
/// gave 2.6-3.1 times the throughput on eight threads (the kernel's reset
/// of the dirty bits is in part serial).
fn wide_ask() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_ASK").is_none_or(|v| v != "0"))
}

/// Askings below this many go one by one: handing them out costs more.
const WIDE_ASK_MIN: usize = 8;

/// The kernel's answers: page ranges asked, with the pages found written.
type Answers = Vec<((usize, usize), Vec<usize>)>;

/// What the kernel answered for each of `runs` (pieces in order, as
/// [`ask_run`] takes them); a run that is not one write-watched reservation
/// is asked by halves, and what still fails is left out. Without the
/// journal's lock: the threads that run the askings also run other work,
/// which may need it.
fn asked_runs(runs: Vec<Vec<(usize, usize)>>) -> Answers {
    if !wide_ask() || runs.len() < WIDE_ASK_MIN {
        let mut found = Vec::new();
        let mut out = Vec::new();
        for run in &runs {
            ask_run(run, &mut found, &mut out);
        }
        return out;
    }
    let out: std::sync::Arc<Mutex<Answers>> = std::sync::Arc::new(Mutex::new(Vec::new()));
    let jobs_n = runs.len().min(16);
    let mut groups: Vec<Vec<Vec<(usize, usize)>>> = (0..jobs_n).map(|_| Vec::new()).collect();
    for (n, run) in runs.into_iter().enumerate() {
        groups[n % jobs_n].push(run);
    }
    let jobs: Vec<Box<dyn FnOnce() + Send>> = groups
        .into_iter()
        .map(|group| {
            let out = out.clone();
            Box::new(move || {
                let mut found = Vec::new();
                let mut mine = Vec::new();
                for run in &group {
                    ask_run(run, &mut found, &mut mine);
                }
                out.lock().unwrap_or_else(|e| e.into_inner()).extend(mine);
            }) as Box<dyn FnOnce() + Send>
        })
        .collect();
    crate::shadow::wide_run(jobs);
    let mut out = out.lock().unwrap_or_else(|e| e.into_inner());
    std::mem::take(&mut *out)
}

/// Notes the kernel's answers in the journal under the ahead asking: the
/// pages found written, and the ranges it now holds.
fn note_asked(j: &mut Journal, asked: Answers) {
    let now = j.ahead_asked;
    for (range, pages) in asked {
        for page in pages {
            j.written.insert(page, now);
        }
        j.ahead.push(range);
    }
}

/// Since the last frame line: the kernel askings that took a millisecond
/// or more, their microseconds, and the longest asking's. An asking walks
/// page tables, some ten microseconds for most ranges; in run 75 the 2,644
/// askings of a frame took 219 ms in the lines where the pass over the
/// heap's views ran long, 25-50 ms in the others. These numbers say
/// whether a few askings wait for something or all of them are slow.
#[cfg(feature = "diagnostics")]
static SLOW: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

#[cfg(feature = "diagnostics")]
fn slow(us: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if us >= 1000 {
        SLOW[0].fetch_add(1, Relaxed);
        SLOW[1].fetch_add(us, Relaxed);
    }
    SLOW[2].fetch_max(us, Relaxed);
}

/// [`SLOW`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn slow_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| SLOW[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// The submission's refreshes are done: `writes` asks the kernel again.
pub(crate) fn ask_ahead_done() {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if !j.kept {
        j.ahead.clear();
    }
}

/// Whether a submission asks the kernel once, before its first refresh,
/// about every range its lists refresh (`VICTORY_ASK_ONCE=1`; off by
/// default).
///
/// A submission's refreshes are done in groups, between the title's own
/// buffer copies (each copy flushes the refreshes recorded before it), and
/// every group asked the kernel again: 2,644 askings a frame, 70 a
/// submission. With the switch the journal answers all of them from one
/// asking made when the submission starts.
///
/// What changes: memory the title's other threads write while the
/// submission is being prepared is seen by the next submission's asking,
/// not by this one's later groups. What this submission's lists read was
/// written before they were submitted, so before the asking. Our own
/// writes into title memory while preparing are told to the journal
/// ([`wrote`]) or end the asking ([`ask_once_end`]).
pub(crate) fn once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default since run 81 (with card buffers: 375-450 ms a frame
    // against 573-647, the picture unchanged over four minutes);
    // `VICTORY_ASK_ONCE=0` asks per range, as before.
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_ONCE").is_none_or(|v| v != "0"))
}

/// With [`once`]: asks about `ranges` (everything a submission's lists
/// refresh) as [`ask_ahead`] does, and lets that asking stand until
/// [`ask_once_end`].
pub(crate) fn ask_once(ranges: &[(usize, usize)]) {
    if !once() {
        return;
    }
    ask_once_end();
    let pieces = page_ranges(ranges);
    // The runs of ranges close together first, as [`ask_ahead`] asks them.
    let mut asked = asked_runs(ahead_runs(&pieces));
    // A range that lies alone, or in a run the kernel refused, would be left
    // to `writes`, which asks once per flush: with one asking for the
    // submission that was every such range asked again at each of its 70
    // flushes (run 76: 2,300 askings a frame with the switch on, as without
    // it). They are asked here, once; what is not write-watched stays with
    // `writes`.
    let mut alone: Vec<(usize, usize)> = pieces
        .iter()
        .copied()
        .filter(|&(first, last)| !asked.iter().any(|&((a, b), _)| a <= first && last <= b))
        .collect();
    alone.dedup();
    // (a run of one piece: asked whole, not left to `writes`)
    asked.extend(asked_runs(alone.into_iter().map(|piece| vec![piece, piece]).collect()));
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    j.ahead.clear();
    // One number for the submission's asking, given once the kernel has
    // answered all of it.
    j.asked += 1;
    j.ahead_asked = j.asked;
    note_asked(j, asked);
    j.kept = !j.ahead.is_empty();
}

/// The submission's one asking stands no longer: its refreshes are done,
/// or title memory was written in a way the journal was not told.
pub(crate) fn ask_once_end() {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if j.kept {
        j.kept = false;
        j.ahead.clear();
    }
}

/// We wrote title memory `[at, at + len)` while a submission's one asking
/// stands (the title's buffer copies, done at submission): its pages count
/// as written after that asking, so the refreshes that follow copy them.
pub(crate) fn wrote(at: usize, len: usize) {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if !j.kept || len == 0 {
        return;
    }
    j.asked += 1;
    let now = j.asked;
    for page in at / PAGE..at.saturating_add(len).div_ceil(PAGE) {
        j.written.insert(page, now);
    }
}

/// One kernel asking for `pieces` (page ranges in order); by halves if it
/// fails, as a run across two reservations or into memory that is not
/// write-watched does. A piece alone is left to [`writes`].
fn ask_run(pieces: &[(usize, usize)], found: &mut Vec<usize>, out: &mut Answers) {
    if pieces.len() < 2 {
        return;
    }
    let first = pieces[0].0;
    let last = pieces.iter().map(|p| p.1).max().unwrap_or(first);
    if let Some(pages) = ask_kernel(found, first, last) {
        out.push(((first, last), pages));
        return;
    }
    if pieces.len() == 2 && pieces[0] == pieces[1] {
        return; // a range alone ([`ask_once`]) that is not write-watched
    }
    let (a, b) = pieces.split_at(pieces.len() / 2);
    ask_run(a, found, out);
    ask_run(b, found, out);
}

/// Asks the kernel for the written pages of pages `[first, last)`, resetting
/// them: the pages found; `None` (and nothing changed) if the range is not
/// write-watched memory of one reservation. `found` is room for the answer.
fn ask_kernel(found: &mut Vec<usize>, first: usize, last: usize) -> Option<Vec<usize>> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    found.resize(count.max(found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    if failed {
        return None;
    }
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    let mut pages: Vec<usize> = found[..count].iter().map(|&at| at / PAGE).collect();
    pages.dedup();
    Some(pages)
}

/// What [`writes`] would say of `[start, end)` from the journal alone, if
/// the range lies in one asked ahead ([`ask_ahead`], [`ask_once`]): the
/// pages written since tracking began and the asking that last found each
/// written, with the number of the asking that stands (as [`writes`] gives
/// it). `None` if the kernel would have to be asked.
pub(crate) fn asked(start: usize, end: usize) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    j.ahead
        .iter()
        .any(|&(a, b)| a <= first && last <= b)
        .then(|| (j.ahead_asked, j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()))
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // Asked ahead with the submission's other ranges: the journal has it.
    if j.ahead.iter().any(|&(a, b)| a <= first && last <= b) {
        return Some((
            j.ahead_asked,
            j.written
                .range(first..last)
                .map(|(&p, &w)| (p, w))
                .collect(),
        ));
    }
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    let guarded = if failed {
        count = 0;
        if !guarded {
            return None;
        }
        #[cfg(feature = "diagnostics")]
        let guarding = std::time::Instant::now();
        let written = guard(first * PAGE, last * PAGE);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
        written?
    } else {
        Vec::new()
    };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    j.asked += 1;
    let now = j.asked;
    let pages = j.found[..count].iter().map(|&at| at / PAGE).chain(guarded);
    for page in pages.collect::<Vec<_>>() {
        j.written.insert(page, now);
    }
    Some((
        now,
        j.written
            .range(first..last)
            .map(|(&p, &w)| (p, w))
            .collect(),
    ))
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

struct Guards {
    /// Blocks we made read-only.
    protected: BTreeSet<usize>,
    /// Of those, the ones not written since a copy took them.
    clean: BTreeSet<usize>,
    /// Blocks already copied as zeros. While they remain wholly uncommitted,
    /// the copy can skip hundreds of MB of failing kernel reads per world frame.
    /// With the frame they were last found so: a block found absent in a
    /// placeholder is not asked about again in that frame ([`absent_every`]).
    absent: BTreeMap<usize, u64>,
    /// The runtime's placeholder reservations (first block -> end block),
    /// which only its mappings fill ([`title_memory_placeholder`]).
    placeholders: BTreeMap<usize, usize>,
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BTreeSet::new(),
    clean: BTreeSet::new(),
    absent: BTreeMap::new(),
    placeholders: BTreeMap::new(),
});

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    // A range whose first block the guards hold already (read-only by our
    // hand, or noted absent) is a view: no question to Windows. The
    // question walks the page tables to the end of the pages that share the
    // first one's state, and a guarded view is one long run of read-only
    // pages: 0.6 ms for the title's 174 MB heap, 16 times a frame, and most
    // of the 68 ms a frame of "guards" (run 58).
    // `VICTORY_GUARD_ASK=1` asks every time, as before.
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let known = !*ASK.get_or_init(|| std::env::var_os("VICTORY_GUARD_ASK").is_some_and(|v| v == "1")) && {
        let g = guards();
        let first = start / BLOCK;
        g.protected.contains(&first) || g.absent.contains_key(&first)
    };
    if !known && !is_view(start, end) {
        return None;
    }
    install();
    let mut g = guards();
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // The range's clean blocks, in order: read off the set in one pass,
    // not looked up one block at a time. A frame asks about some 10 GB of
    // view memory, 150,000 blocks, nearly all clean: the lookups were most
    // of the 60-90 ms a frame of "guards" (run 54). What the loop adds to
    // the set meanwhile is the block it stands at or ones before it.
    let clean: Vec<usize> = g.clean.range(first..last).copied().collect();
    if clean.len() == last - first {
        // Nothing written, mapped or new since the range was last taken.
        return Some(written);
    }
    let mut next_clean = 0;
    let mut uncommitted_until = first;
    let frame = crate::frame::current();
    let every = absent_every();
    for block in first..=last {
        while next_clean < clean.len() && clean[next_clean] < block {
            next_clean += 1;
        }
        let is_clean = next_clean < clean.len() && clean[next_clean] == block;
        let mut open = block < last && !is_clean;
        if open && let Some(&seen) = g.absent.get(&block) {
            if seen == frame && !every && in_placeholder(&g.placeholders, block) {
                // Found absent in this frame already, in a placeholder: only
                // the runtime's mappings fill one, and they say so
                // ([`title_memory_remapped`]).
                open = false;
            } else {
                // Ordinary commits need no runtime mapping notification. Reuse
                // this asking's query for the following blocks in the same region.
                if block >= uncommitted_until {
                    uncommitted_until = uncommitted_end(block);
                }
                open = block >= uncommitted_until;
                if open {
                    g.absent.remove(&block);
                } else {
                    g.absent.insert(block, frame);
                }
            }
        }
        if open {
            written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            if g.protected.contains(&block) {
                // Written through the runtime, still read-only.
                g.clean.insert(block);
            } else {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() {
            protect(&mut g, from, block);
        }
    }
    Some(written)
}

/// Whether a block found absent in a placeholder is asked about again at
/// every refresh (`VICTORY_GUARD_ABSENT_EVERY=1`), not once a frame.
///
/// The title's page pool is a placeholder that the runtime maps views into
/// block by block. An unmapped block stays a placeholder of its own, so one
/// query sees one block: a refresh over 174 MB of them cost 977 microseconds
/// (7 for a mapped, clean view of that size), and a world frame refreshes
/// such ranges 17 times. Nothing but a mapping (`MEM_REPLACE_PLACEHOLDER`,
/// the runtime's, which it tells of: [`title_memory_remapped`]) puts memory
/// in a placeholder; a block of an ordinary reservation can be committed
/// without a word, and is asked about every time as before.
/// shortcut: once a frame, not never: the title frees its reservations with
/// the host's `VirtualFree`, which says nothing, and the addresses of a freed
/// placeholder could be reserved and committed again.
fn absent_every() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GUARD_ABSENT_EVERY").is_some_and(|v| v == "1"))
}

/// Whether `block` lies in a placeholder reservation of the runtime's.
fn in_placeholder(placeholders: &BTreeMap<usize, usize>, block: usize) -> bool {
    placeholders.range(..=block).next_back().is_some_and(|(_, &end)| block < end)
}

/// The runtime reserved `[at, at + len)` as a placeholder
/// (`XMemVirtualAlloc` of 64 KiB pages, filled by `XMemMapPhysicalPages`).
pub fn title_memory_placeholder(at: usize, len: usize) {
    let mut g = guards();
    let (first, end) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    // (a reservation made again where an old one was replaces it)
    let stale: Vec<usize> = g.placeholders.range(first..end).map(|(&b, _)| b).collect();
    for b in stale {
        g.placeholders.remove(&b);
    }
    g.placeholders.insert(first, end);
}

/// Whether `[start, end)` is view memory. The title's views lie in
/// placeholders, mapped and unmapped block by block as it streams, so a
/// range can begin in a block that is not mapped right now: the first
/// pages of the range that are there decide. A range with nothing there
/// counts as a view (its blocks are noted absent, and looked at again when
/// the runtime maps them).
///
/// Looking at the first address only took such a range for "not a view":
/// its copy was marked not watched for good and copied whole at every
/// refresh, around its unmapped pages (run 54: 300 such ranges, 570 MB
/// copied a frame for them, 255 copies a frame reading around holes).
/// `VICTORY_GUARD_HOLES=0` looks at the first address only, as before.
fn is_view(start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    static HOLES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let holes = *HOLES.get_or_init(|| std::env::var_os("VICTORY_GUARD_HOLES").is_none_or(|v| v != "0"));
    let mut at = start;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return false;
        }
        if info.Type == MEM_MAPPED {
            return true;
        }
        // Committed memory that is no view (the title's heaps, ours), or
        // the answer as it was before.
        if info.State == MEM_COMMIT || !holes {
            return false;
        }
        // A hole (a placeholder, or free): on to what follows it.
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= at || next >= end {
            return true;
        }
        at = next;
    }
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail, and
/// noting those not committed (unmapped) absent.
fn protect(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        }
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.extend(from..to);
        g.clean.extend(from..to);
    };
    if set(from, to) {
        note(g, from, to);
        return;
    }
    for block in from..to {
        if set(block, block + 1) {
            note(g, block, block + 1);
        } else if block < uncommitted_end(block) {
            g.absent.insert(block, crate::frame::current());
        }
    }
}

/// The end (exclusive) of the whole uncommitted blocks starting at `block`.
fn uncommitted_end(block: usize) -> usize {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    if n != 0 && matches!(info.State, MEM_RESERVE | MEM_FREE) {
        (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK
    } else {
        block
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let blocks: Vec<usize> = g
        .protected
        .range(start / BLOCK..end.div_ceil(BLOCK))
        .copied()
        .collect();
    for &block in &blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
        let mut old = 0;
        // SAFETY: a block we made read-only.
        unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
    }
    !blocks.is_empty()
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    // The length is the title's (`fread`'s size times count, saturated).
    open(&mut guards(), at, at.saturating_add(len));
    // A placed texture there changes with no descriptor written.
    crate::descriptor::host::remapped(at, len);
}

/// Reads of title memory in place under way, and changes of what it shows
/// under way or waiting to begin, each with the range it is of.
///
/// The runtime does not change a mapping in one step. To unmap one page of
/// a view it unmaps the whole view and maps what stays again
/// (`xmem.rs` `unmap`), and a view it maps is a reservation until the next
/// call commits it (`map`, then `commit`). For that moment pages the title
/// never gave up are "reserved, never committed", and a thread reading
/// them in place takes an access violation: run 71 in `Layout::detile`,
/// run 74 in the hash before it, both under `Staging::refresh_now`, which
/// had asked whether the texture was all mapped and been told yes. Asking
/// cannot help; not changing the mapping while such a read is under way
/// does.
///
/// By range: one lock for all of title memory made every texture refresh
/// wait for every page the title streamed in, wherever it lay. With 5,700
/// texture slots streaming that was 2,028 of the 2,093 ms a frame spent on
/// textures (RTX 5080, 2026-11-24), and each change waited for every read.
struct Mappings {
    /// Reads under way: number, start, end.
    reads: Vec<(u64, usize, usize)>,
    /// Changes under way or waiting: number, start, end. One that waits is
    /// here already, so reads that begin after it wait their turn.
    changes: Vec<(u64, usize, usize)>,
    next: u64,
}

static MAPPINGS: Mutex<Mappings> = Mutex::new(Mappings {
    reads: Vec::new(),
    changes: Vec::new(),
    next: 0,
});
static SETTLED: std::sync::Condvar = std::sync::Condvar::new();

fn mappings() -> MutexGuard<'static, Mappings> {
    MAPPINGS.lock().unwrap_or_else(|e| e.into_inner())
}

fn overlaps(ranges: &[(u64, usize, usize)], at: usize, end: usize) -> bool {
    ranges.iter().any(|&(_, a, e)| a < end && at < e)
}

/// What the runtime holds while it changes what title memory shows
/// ([`title_memory_remapping`]).
pub struct Remapping(u64);

impl Drop for Remapping {
    fn drop(&mut self) {
        mappings().changes.retain(|c| c.0 != self.0);
        SETTLED.notify_all();
    }
}

/// The runtime is about to map, unmap or move physical pages in title
/// memory `[at, at + len)`: until what it returns is dropped, no read in
/// place of that memory is under way and none begins
/// ([`title_memory_reading`]). Taken around the whole change, the unmapping
/// and the mapping again; the range is all that may go away meanwhile (the
/// whole of every view the change cuts).
pub fn title_memory_remapping(at: usize, len: usize) -> Remapping {
    let end = at.saturating_add(len);
    let mut m = mappings();
    let id = m.next;
    m.next += 1;
    m.changes.push((id, at, end));
    // Changes of one range take turns in the order they came.
    while overlaps(&m.reads, at, end)
        || m.changes.iter().any(|&(other, a, e)| other < id && a < end && at < e)
    {
        m = SETTLED.wait(m).unwrap_or_else(|e| e.into_inner());
    }
    Remapping(id)
}

/// What a read of title memory in place holds ([`title_memory_reading`]).
pub struct Reading(u64);

impl Drop for Reading {
    fn drop(&mut self) {
        let mut m = mappings();
        if let Some(i) = m.reads.iter().position(|r| r.0 == self.0) {
            m.reads.swap_remove(i);
        }
        let waited_for = !m.changes.is_empty();
        drop(m);
        if waited_for {
            SETTLED.notify_all();
        }
    }
}

/// For reading title memory `[at, at + len)` in place: until what it
/// returns is dropped, the runtime changes no mapping there, so memory
/// found mapped stays mapped. Not to be taken twice by one thread, nor held
/// across anything that maps title memory.
pub fn title_memory_reading(at: usize, len: usize) -> Reading {
    let end = at.saturating_add(len);
    let mut m = mappings();
    while overlaps(&m.changes, at, end) {
        m = SETTLED.wait(m).unwrap_or_else(|e| e.into_inner());
    }
    begin(&mut m, at, end)
}

/// [`title_memory_reading`] if it can be had at once: `None` while the
/// runtime changes a mapping there (or waits to), for a caller that has
/// another way to read and would rather not wait.
pub fn title_memory_reading_now(at: usize, len: usize) -> Option<Reading> {
    let end = at.saturating_add(len);
    let mut m = mappings();
    (!overlaps(&m.changes, at, end)).then(|| begin(&mut m, at, end))
}

fn begin(m: &mut Mappings, at: usize, end: usize) -> Reading {
    let id = m.next;
    m.next += 1;
    m.reads.push((id, at, end));
    Reading(id)
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded (nor known
/// unmapped).
pub fn title_memory_remapped(at: usize, len: usize) {
    // A placed texture there arrives, goes or changes with no descriptor
    // written.
    crate::descriptor::host::remapped(at, len);
    // The views the runtime told us of there are no longer what it shows.
    crate::inplace::views_gone(at, len);
    // And nothing stays in place over it.
    crate::inplace::title_memory_freeing(at, len);
    let mut g = guards();
    let range = at / BLOCK..at.saturating_add(len).div_ceil(BLOCK);
    let absent: Vec<usize> = g.absent.range(range.clone()).map(|(&block, _)| block).collect();
    for block in absent {
        g.absent.remove(&block);
    }
    let blocks: Vec<usize> = g.protected.range(range).copied().collect();
    for block in blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
    }
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let mut g = guards();
        if open(&mut g, at, at + 1) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// Cost of the guards on a view as big as the title's 174 MB heap, which
    /// a world frame refreshes 17 times (412 guard faults a frame in all):
    /// a refresh with nothing written, one after 24 blocks were written, and
    /// a guard fault. `cargo test -p victory-gpu --release guard_cost -- --ignored --nocapture`.
    #[test]
    #[ignore = "a measurement, not a check"]
    fn guard_cost_on_a_heap_sized_view() {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, SEC_COMMIT};
        const SIZE: usize = 0xae0_0000;
        const BLOCKS: usize = SIZE / BLOCK;
        // SAFETY: a fresh committed section and one writable view of it, kept
        // for the test process (the guard state is shared).
        let section = unsafe {
            CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE | SEC_COMMIT, 0, SIZE as u32, std::ptr::null())
        };
        assert!(!section.is_null());
        // SAFETY: as above.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE) };
        let at = view.Value as usize;
        assert_ne!(at, 0);
        title_memory_remapped(at, SIZE);
        crate::title_memory_view(at, SIZE, section as usize, 0);
        let refresh = || guard(at, at + SIZE).expect("a view");
        assert_eq!(refresh().len(), SIZE / PAGE, "first refresh takes every block");
        let time = |f: &mut dyn FnMut()| {
            let started = std::time::Instant::now();
            f();
            started.elapsed().as_secs_f64() * 1e6
        };
        const ROUNDS: usize = 200;
        let clean: f64 = (0..ROUNDS).map(|_| time(&mut || assert!(refresh().is_empty()))).sum::<f64>() / ROUNDS as f64;
        let mut seed = 0x9E37_79B9u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed as usize) % BLOCKS
        };
        let (mut faults, mut dirty, mut written) = (0.0, 0.0, 0usize);
        for _ in 0..ROUNDS {
            let blocks: Vec<usize> = (0..24).map(|_| next()).collect();
            faults += time(&mut || {
                for &b in &blocks {
                    // SAFETY: inside the view; the first write to a guarded
                    // block faults and the handler opens it.
                    unsafe { ((at + b * BLOCK + 64) as *mut u32).write_volatile(b as u32) };
                }
            });
            dirty += time(&mut || written += refresh().len());
        }
        eprintln!(
            "GUARD-COST {} MB view: clean refresh {clean:.1} us; refresh after 24 written blocks {:.1} us ({:.1} blocks found); \
             a guard fault {:.2} us",
            SIZE >> 20,
            dirty / ROUNDS as f64,
            written as f64 / ROUNDS as f64 / (BLOCK / PAGE) as f64,
            faults / (ROUNDS * 24) as f64,
        );

        // The title's page pool: a placeholder the runtime maps views into
        // block by block, left as one 64 KiB placeholder per block once they
        // are unmapped. Here every block is such a placeholder, none mapped.
        use windows_sys::Win32::System::Memory::{VirtualAlloc2, VirtualFree};
        const MEM_RESERVE_PLACEHOLDER: u32 = 0x4_0000;
        const MEM_PRESERVE_PLACEHOLDER: u32 = 0x2;
        const PAGE_NOACCESS: u32 = 0x01;
        // SAFETY: a fresh placeholder of ours, split in place, kept.
        let pool = unsafe {
            VirtualAlloc2(0 as _, std::ptr::null(), SIZE, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, std::ptr::null_mut(), 0)
        } as usize;
        assert_ne!(pool, 0);
        for b in 0..BLOCKS - 1 {
            // SAFETY: splits the placeholder at a block boundary.
            let ok = unsafe { VirtualFree((pool + b * BLOCK) as *mut _, BLOCK, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) };
            assert_ne!(ok, 0, "split at block {b}");
        }
        title_memory_placeholder(pool, SIZE);
        let pool_refresh = || guard(pool, pool + SIZE).expect("unmapped placeholders count as a view");
        let _ = pool_refresh(); // every block, the first time
        let absent: f64 = (0..20).map(|_| time(&mut || assert!(pool_refresh().is_empty()))).sum::<f64>() / 20.0;
        eprintln!(
            "GUARD-COST {} MB pool of {BLOCKS} unmapped 64 KiB placeholders: refresh {absent:.1} us ({})",
            SIZE >> 20,
            if absent_every() { "asking about every block, VICTORY_GUARD_ABSENT_EVERY=1" } else { "found absent in this frame already" }
        );
    }

    /// An absent block of a placeholder the runtime reserved is asked about
    /// once a frame: what fills it comes with a mapping notification, after
    /// which the next refresh looks again. (A plain reservation stands in for
    /// the placeholder: committing it without a word shows the skip.)
    #[test]
    fn absent_placeholder_blocks_are_asked_about_once_a_frame() {
        // SAFETY: a fresh reservation of ours, nothing committed; kept (the
        // guard state is shared).
        let p = unsafe { VirtualAlloc(std::ptr::null(), 4 * BLOCK, MEM_RESERVE, PAGE_READWRITE) } as usize;
        assert_ne!(p, 0);
        title_memory_placeholder(p, 4 * BLOCK);
        let refresh = || guard(p, p + 4 * BLOCK).expect("a range with nothing there counts as a view");
        let _ = refresh(); // every block, the first time
        assert!(refresh().is_empty(), "absent");
        // SAFETY: commits block 1 of our reservation and writes into it.
        unsafe {
            assert_ne!(VirtualAlloc((p + BLOCK) as *const _, BLOCK, MEM_COMMIT, PAGE_READWRITE) as usize, 0);
            ((p + BLOCK + 8) as *mut u32).write_volatile(7);
        }
        if !absent_every() {
            assert!(refresh().is_empty(), "found absent in this frame already");
        }
        title_memory_remapped(p + BLOCK, BLOCK);
        let pages: Vec<usize> = ((p + BLOCK) / PAGE..(p + 2 * BLOCK) / PAGE).collect();
        assert_eq!(refresh(), pages, "the block the runtime mapped is taken");
        assert!(refresh().is_empty(), "and guarded from then on");
    }

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A submission's one asking over many reservations (asked on several
    /// threads past `WIDE_ASK_MIN`): runs that cross reservations are asked
    /// by halves and then range by range, every written page is noted under
    /// the one number of the asking, and the journal answers for every range.
    #[test]
    fn one_asking_over_many_reservations_notes_every_written_page() {
        const RESERVATIONS: usize = 12;
        // SAFETY: fresh write-watched allocations, kept (see the end).
        let blocks: Vec<usize> = (0..RESERVATIONS)
            .map(|_| unsafe {
                VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) as usize
            })
            .collect();
        assert!(blocks.iter().all(|&p| p != 0));
        let ranges: Vec<(usize, usize)> = blocks.iter().map(|&p| (p, p + 2 * PAGE)).collect();
        // Clean first, then a write in the second page of every other one.
        for &(start, end) in &ranges {
            writes(start, end, false).expect("watched");
        }
        for (n, &p) in blocks.iter().enumerate() {
            if n % 2 == 0 {
                // SAFETY: committed pages of ours.
                unsafe { ((p + PAGE + 16) as *mut u8).write(n as u8) };
            }
        }
        if !once() {
            return; // VICTORY_ASK_ONCE=0
        }
        ask_once(&ranges);
        let mut stood = None;
        for (n, &(start, end)) in ranges.iter().enumerate() {
            let (asking, pages) = asked(start, end).expect("the one asking holds every range");
            assert_eq!(*stood.get_or_insert(asking), asking, "one number for the whole asking");
            // (the journal is shared: pages of memory another test freed
            // here keep their older numbers)
            let now: Vec<usize> = pages.iter().filter(|p| p.1 == asking).map(|p| p.0).collect();
            let expected: Vec<usize> = if n % 2 == 0 { vec![start / PAGE + 1] } else { Vec::new() };
            assert_eq!(now, expected, "reservation {n}");
        }
        ask_once_end();
        // The allocations are kept: the journal is shared, and a test given
        // these addresses next would find this test's pages noted written.
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        let landed = unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) };
        assert_eq!(landed, b"from the disk");
        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some((view / PAGE..view / PAGE + 1).collect())
        );
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
        title_memory_remapped(view, 2 * BLOCK);
    }

    #[test]
    fn writes_in_partly_committed_view_blocks_are_reported() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };

        const SIZE: usize = 3 * BLOCK;
        struct View(MEMORY_MAPPED_VIEW_ADDRESS);
        impl Drop for View {
            fn drop(&mut self) {
                title_memory_remapped(self.0.Value as usize, SIZE);
                // SAFETY: our view is still mapped.
                unsafe { UnmapViewOfFile(self.0) };
            }
        }
        // SAFETY: a new reserved pagefile section. The view holds its own
        // reference after we close the handle, including on assertion failure.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            View(view)
        };
        let start = view.0.Value as usize;
        let commit = |at: usize, len| {
            // SAFETY: these pages belong to our reserved view.
            let committed = unsafe { VirtualAlloc(at as _, len, MEM_COMMIT, PAGE_READWRITE) };
            assert_eq!(committed as usize, at);
        };
        let early = start + BLOCK + 3 * KERNEL_PAGE;
        let late = start + 2 * BLOCK + 3 * KERNEL_PAGE;
        commit(start, BLOCK);
        commit(early, KERNEL_PAGE);
        let refresh = || guard(start, start + SIZE).unwrap();
        assert_eq!(refresh(), [start / PAGE, early / PAGE, late / PAGE]);
        // The last block is still entirely reserved.
        assert!(!refresh().contains(&(late / PAGE)));

        commit(late, KERNEL_PAGE); // Ordinary commit, without a runtime mapping notification.
        for at in [early, late] {
            assert!(writable(at));
            // SAFETY: each address is inside its committed read-write page.
            unsafe { (at as *mut u8).write_volatile(1) };
        }
        assert_eq!(refresh(), [early / PAGE, late / PAGE]);
    }

    /// A range of view memory whose first block is not mapped is a view
    /// all the same: its blocks are reported once, then the ones written
    /// or mapped since.
    #[test]
    fn a_range_that_begins_in_an_unmapped_block_is_guarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of three; the middle block is mapped first.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (3 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        // Nothing mapped at all: a view with every block absent.
        assert!(is_view(p, p + 3 * BLOCK));
        let middle = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        // The range begins in the placeholder, its middle block is a view.
        assert!(is_view(p, p + 3 * BLOCK));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()), "nothing written since");
        // SAFETY: inside the mapped middle block.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        let first = map(0);
        title_memory_remapped(p, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(0).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // Memory of ours that is no view stays none.
        let mut local = [0u8; 16];
        let at = local.as_mut_ptr() as usize;
        assert!(!is_view(at, at + local.len()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), middle, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// Blocks of a placeholder between views are reported once, then not
    /// again (they cannot be guarded) until the runtime maps them.
    #[test]
    fn unmapped_blocks_are_reported_once_until_mapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of two; block 0 shows the section's first block, as the
        // runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let first = map(0);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some(Vec::new()),
            "nothing written since"
        );
        let second = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }
}

#[cfg(test)]
mod mapping_tests {
    use super::{title_memory_reading, title_memory_reading_now, title_memory_remapping};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    /// Addresses no other test's memory has.
    const AT: usize = 0x7000_0000_0000;
    const LEN: usize = 0x10000;

    /// Runs `f` on a thread; whether it has ended (after 100 ms).
    fn blocked(f: impl FnOnce() + Send + 'static) -> (Arc<AtomicBool>, std::thread::JoinHandle<()>) {
        let done = Arc::new(AtomicBool::new(false));
        let (began, wait) = mpsc::channel();
        let thread = {
            let done = done.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                f();
                done.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        (done, thread)
    }

    /// A change of mapping waits for a read in place of that memory that is
    /// under way, and a read in place waits for a change of mapping there.
    #[test]
    fn reads_in_place_and_remappings_wait_for_each_other() {
        let at = AT;
        let reading = title_memory_reading(at, LEN);
        let (changed, change) = blocked(move || drop(title_memory_remapping(at + LEN - 1, 1)));
        assert!(!changed.load(Ordering::SeqCst), "the mapping changed under a read in place");
        // One that waits goes before the reads that come after it.
        assert!(title_memory_reading_now(at, LEN).is_none());
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
        let changing = title_memory_remapping(at, LEN);
        assert!(title_memory_reading_now(at, 1).is_none());
        let (read, reader) = blocked(move || drop(title_memory_reading(at, 1)));
        assert!(!read.load(Ordering::SeqCst), "a read in place began during a change of mapping");
        drop(changing);
        reader.join().unwrap();
        assert!(read.load(Ordering::SeqCst));
    }

    /// Reads and changes of other memory do not wait for each other, nor do
    /// reads of the same memory.
    #[test]
    fn reads_and_changes_of_other_memory_go_on() {
        let at = AT + 16 * LEN;
        let changing = title_memory_remapping(at, LEN);
        assert!(title_memory_reading_now(at + LEN, LEN).is_some());
        assert!(title_memory_reading_now(at - LEN, LEN).is_some());
        drop(title_memory_reading(at + LEN, LEN));
        drop(title_memory_remapping(at + LEN, LEN));
        drop(changing);
        let one = title_memory_reading(at, LEN);
        drop(title_memory_reading_now(at, LEN).expect("reads share"));
        drop(title_memory_remapping(at + LEN, LEN));
        drop(one);
    }

    /// Changes of one range take turns, and none is left behind.
    #[test]
    fn many_reads_and_changes_all_end() {
        let at = AT + 64 * LEN;
        let inside = Arc::new(AtomicBool::new(false));
        let threads: Vec<_> = (0..8)
            .map(|t| {
                let inside = inside.clone();
                std::thread::spawn(move || {
                    for i in 0..400 {
                        let from = at + (i * 7 + t) % 4 * LEN;
                        if t % 4 == 0 {
                            let _change = title_memory_remapping(at, 4 * LEN);
                            assert!(!inside.swap(true, Ordering::SeqCst));
                            inside.store(false, Ordering::SeqCst);
                        } else {
                            let _read = title_memory_reading(from, LEN);
                            assert!(!inside.load(Ordering::SeqCst));
                        }
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        let m = super::mappings();
        let ours = |r: &&(u64, usize, usize)| r.1 >= at && r.2 <= at + 4 * LEN;
        assert_eq!(m.reads.iter().filter(ours).count(), 0);
        assert_eq!(m.changes.iter().filter(ours).count(), 0);
    }
}
