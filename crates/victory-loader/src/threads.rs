//! Thread notices for title images. The prepared copies have no entry point
//! and no TLS callbacks, so the Windows loader tells them nothing when a
//! thread starts or ends. Our own TLS callback, which Windows calls on every
//! thread of the process whoever created it (title, host CRT or our
//! runtime), passes each notice on to every attached image: its TLS
//! callbacks, then a DLL's `DllMain`.

use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::{DLL_PROCESS_DETACH, LoadedImage};

const DLL_THREAD_ATTACH: u32 = 2;
const DLL_THREAD_DETACH: u32 = 3;

struct Mapped {
    notices: Arc<ImageNotices>,
    /// Successful loads sharing this module handle.
    loads: usize,
}

// Snapshots keep the metadata alive. Callbacks may revoke a later notice
// while the snapshot is being dispatched without holding the registry lock.
struct ImageNotices {
    image: LoadedImage,
    /// Its `DLL_PROCESS_ATTACH` is done; only then does it get thread notices.
    attached: AtomicBool,
    /// `DisableThreadLibraryCalls`: no more `DllMain` thread notices.
    quiet: AtomicBool,
}

/// Every title image we mapped, in load order.
static IMAGES: Mutex<Vec<Mapped>> = Mutex::new(Vec::new());

fn images() -> std::sync::MutexGuard<'static, Vec<Mapped>> {
    IMAGES.lock().unwrap_or_else(|e| e.into_inner())
}

pub(crate) fn mapped(image: &LoadedImage) {
    // SAFETY: reads our own static, so the linker keeps it (and its section).
    unsafe { std::ptr::from_ref(&ON_THREAD).read_volatile() };
    let mut images = images();
    if let Some(m) = images
        .iter_mut()
        .find(|m| m.notices.image.base() == image.base())
    {
        m.loads += 1;
        return;
    }
    images.push(Mapped {
        notices: Arc::new(ImageNotices {
            image: image.clone(),
            attached: AtomicBool::new(false),
            quiet: AtomicBool::new(false),
        }),
        loads: 1,
    });
}

/// Its `DLL_PROCESS_ATTACH` is done. The list keeps attach order, which is
/// the Windows loader's initialisation order.
pub(crate) fn attached(base: usize) {
    let mut images = images();
    if let Some(i) = images
        .iter()
        .position(|m| m.notices.image.base() == base && !m.notices.attached.load(Ordering::SeqCst))
    {
        let mut m = images.remove(i);
        // A snapshot of an earlier attachment may still be in flight.
        if Arc::get_mut(&mut m.notices).is_none() {
            m.notices = Arc::new(ImageNotices {
                image: m.notices.image.clone(),
                attached: AtomicBool::new(false),
                quiet: AtomicBool::new(m.notices.quiet.load(Ordering::SeqCst)),
            });
        }
        m.notices.attached.store(true, Ordering::SeqCst);
        images.push(m);
    }
}

/// The CRT runs exit handlers in reverse order. Register before the first
/// title TLS callback or DllMain has a chance to install its own handler.
pub(crate) fn register_process_detach() -> anyhow::Result<()> {
    static RESULT: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    // SAFETY: the callback lives for the lifetime of the process.
    let result = RESULT.get_or_init(|| unsafe { atexit(detach_all) });
    anyhow::ensure!(*result == 0, "cannot register process detach");
    Ok(())
}

pub(crate) fn detached(base: usize) {
    if let Some(m) = images().iter().find(|m| m.notices.image.base() == base) {
        m.notices.attached.store(false, Ordering::SeqCst);
    }
}

unsafe extern "C" {
    fn atexit(f: extern "C" fn()) -> i32;
}

/// Set once the process is ending: no more thread notices.
static EXITING: AtomicBool = AtomicBool::new(false);

/// How long title images may take to detach before the process ends anyway
/// (a thread ended while holding a lock their detach code wants).
const DETACH_DEADLINE: Duration = Duration::from_secs(5);

/// Process exit (the CRT's `exit`, which titles call), in `ExitProcess`'s
/// order: the title's exit functions have run; every other thread ends
/// (title threads and ours); every attached image gets
/// `DLL_PROCESS_DETACH`, DLLs in reverse initialisation order, then the
/// exe's TLS callbacks; then the CRT ends the process.
extern "C" fn detach_all() {
    EXITING.store(true, Ordering::SeqCst);
    let attached = attached_images();
    if attached.is_empty() {
        return;
    }
    let (dlls, exes): (Vec<_>, Vec<_>) = attached
        .into_iter()
        .rev()
        .partition(|i| i.image.info().is_dll());
    tracing::info!(
        images = dlls.len() + exes.len(),
        "process exit: ending other threads, detaching title images"
    );
    // As on Windows, a lock a thread held when it ended stays taken; detach
    // code that waits for one would hang the exit. The watchdog runs before
    // the other threads end: a thread that ends can hold the heap lock or
    // the loader lock, and a new thread needs both to start. After the
    // deadline it only ends the process; it does not log, because logging
    // needs the heap and the log writer's lock.
    // Its thread id, set when it runs its own code; 0 until then.
    static WATCHDOG: AtomicU32 = AtomicU32::new(0);
    drop(std::thread::spawn(|| {
        // SAFETY: a plain query of the calling thread.
        let id = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() };
        WATCHDOG.store(id, Ordering::SeqCst);
        std::thread::sleep(DETACH_DEADLINE);
        end_process();
    }));
    // Its start (loader notices, Rust's thread set-up) is done only when
    // it runs its own code. The loader notices need the loader lock. This
    // thread can hold it, or a thread that waits for this thread can hold
    // it. Then the watchdog does not start, so this wait has the same
    // deadline.
    let waiting = Instant::now();
    let watchdog = loop {
        match WATCHDOG.load(Ordering::SeqCst) {
            0 if waiting.elapsed() >= DETACH_DEADLINE => end_process(),
            0 => std::thread::yield_now(),
            id => break id,
        }
    };
    if end_other_threads(&[watchdog]).is_err() {
        end_process();
    }
    for notices in dlls.iter().chain(&exes) {
        if !notices.attached.swap(false, Ordering::SeqCst) {
            continue;
        }
        // SAFETY: attached images are bound. A non-null third argument
        // tells the entry points the process is ending.
        unsafe {
            notices.image.notify(
                DLL_PROCESS_DETACH,
                true,
                std::ptr::without_provenance_mut(1),
            )
        };
    }
}

/// Ends this process when the title images do not detach in time. It does
/// not use the heap or any lock: a thread that ended can hold them.
fn end_process() -> ! {
    // SAFETY: ends this process, which is exiting anyway.
    unsafe {
        windows_sys::Win32::System::Threading::TerminateProcess(
            windows_sys::Win32::System::Threading::GetCurrentProcess(),
            1,
        )
    };
    std::process::abort()
}

/// Ends every thread of the process but this one and those in `keep` (and
/// the ones started while doing so), as `ExitProcess` does first; except
/// the system's thread-pool workers, which `TerminateThread` refuses (the
/// process fails fast) and which run no title code (no title image uses
/// that pool).
fn end_other_threads(keep: &[u32]) -> Result<(), ()> {
    use windows_sys::Win32::Foundation::{ERROR_NO_MORE_FILES, GetLastError, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcessId, GetCurrentThreadId};
    // SAFETY: plain queries of the calling process and thread.
    let (process, me) = unsafe { (GetCurrentProcessId(), GetCurrentThreadId()) };
    loop {
        // SAFETY: requests a read-only snapshot and takes ownership on success.
        let raw = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) };
        if raw == INVALID_HANDLE_VALUE {
            return Err(());
        }
        // SAFETY: this is the sole owner of the returned snapshot.
        let snapshot = unsafe { OwnedHandle::from_raw_handle(raw) };
        let mut entry = THREADENTRY32 {
            dwSize: size_of::<THREADENTRY32>() as u32,
            ..Default::default()
        };
        let mut rescan = false;
        // SAFETY: the snapshot is live and entry has the advertised size.
        let mut more = unsafe { Thread32First(snapshot.as_raw_handle(), &mut entry) };
        while more != 0 {
            if !has_owner_field(entry.dwSize) {
                return Err(());
            }
            let id = entry.th32ThreadID;
            if entry.th32OwnerProcessID == process && id != me && !keep.contains(&id) {
                rescan |= end_thread(id, process)?;
            }
            entry.dwSize = size_of::<THREADENTRY32>() as u32;
            // SAFETY: as for Thread32First.
            more = unsafe { Thread32Next(snapshot.as_raw_handle(), &mut entry) };
        }
        // SAFETY: queried immediately after the failed enumeration call.
        if unsafe { GetLastError() } != ERROR_NO_MORE_FILES {
            return Err(());
        }
        if !rescan {
            return Ok(());
        }
    }
}

fn has_owner_field(bytes: u32) -> bool {
    use windows_sys::Win32::System::Diagnostics::ToolHelp::THREADENTRY32;
    bytes as usize >= std::mem::offset_of!(THREADENTRY32, th32OwnerProcessID) + size_of::<u32>()
}

/// Returns whether the snapshot may have changed. A thread ID can be reused
/// between enumeration and OpenThread, including by a different process.
fn end_thread(id: u32, process: u32) -> Result<bool, ()> {
    use windows_sys::Win32::Foundation::{ERROR_INVALID_PARAMETER, GetLastError, WAIT_OBJECT_0};
    use windows_sys::Win32::System::Threading::{
        INFINITE, OpenThread, THREAD_QUERY_INFORMATION, THREAD_SYNCHRONIZE, THREAD_TERMINATE,
        TerminateThread, WaitForSingleObject,
    };
    // SAFETY: an ID from the snapshot, with the rights needed below.
    let raw = unsafe {
        OpenThread(
            THREAD_TERMINATE | THREAD_QUERY_INFORMATION | THREAD_SYNCHRONIZE,
            0,
            id,
        )
    };
    if raw.is_null() {
        // SAFETY: read immediately after OpenThread. A vanished ID is normal.
        return (unsafe { GetLastError() } == ERROR_INVALID_PARAMETER)
            .then_some(true)
            .ok_or(());
    }
    // SAFETY: sole owner of the opened handle.
    let thread = unsafe { OwnedHandle::from_raw_handle(raw) };
    let raw = thread.as_raw_handle();
    if !belongs_to_process(raw, process)? {
        return Ok(true);
    }
    match starts_in(raw, c"ntdll.dll") {
        Some(true) => return Ok(false),
        None => {
            // SAFETY: a thread can exit naturally before its start address is queried.
            return (unsafe { WaitForSingleObject(raw, 0) } == WAIT_OBJECT_0)
                .then_some(true)
                .ok_or(());
        }
        Some(false) => (),
    }
    // SAFETY: a verified thread of this process. Detach may start only when
    // the thread is signaled. The exit watchdog bounds this wait.
    let ended = unsafe { TerminateThread(raw, 0) };
    let timeout = if ended != 0 { INFINITE } else { 0 };
    // SAFETY: the owned handle has SYNCHRONIZE access.
    (unsafe { WaitForSingleObject(raw, timeout) } == WAIT_OBJECT_0)
        .then_some(true)
        .ok_or(())
}

fn belongs_to_process(
    thread: windows_sys::Win32::Foundation::HANDLE,
    process: u32,
) -> Result<bool, ()> {
    // SAFETY: queries the handle. Windows reports failure for an invalid handle.
    match unsafe { windows_sys::Win32::System::Threading::GetProcessIdOfThread(thread) } {
        0 => Err(()),
        owner => Ok(owner == process),
    }
}

pub(crate) fn unmapped(base: usize) {
    let mut images = images();
    if let Some(i) = images.iter().position(|m| m.notices.image.base() == base) {
        images[i].loads -= 1;
        if images[i].loads == 0 {
            images[i].notices.attached.store(false, Ordering::SeqCst);
            images.remove(i);
        }
    }
}

/// `DisableThreadLibraryCalls` for a title image; `None` if `base` is not
/// one. Fails, as on Windows, for an image with static TLS.
pub fn disable_thread_calls(base: usize) -> Option<bool> {
    let images = images();
    let m = images.iter().find(|m| m.notices.image.base() == base)?;
    if m.notices.image.info().tls.is_some() {
        return Some(false);
    }
    m.notices.quiet.store(true, Ordering::SeqCst);
    Some(true)
}

fn attached_images() -> Vec<Arc<ImageNotices>> {
    images()
        .iter()
        .filter(|m| m.notices.attached.load(Ordering::SeqCst))
        .map(|m| m.notices.clone())
        .collect()
}

/// Windows calls every pointer between the CRT's markers `.CRT$XLA` and
/// `.CRT$XLZ`; `XLAV` sorts ahead of the usual `XLB`, so ours runs before
/// any other callback linked into the exe.
#[used]
#[unsafe(link_section = ".CRT$XLAV")]
static ON_THREAD: unsafe extern "system" fn(*mut c_void, u32, *mut c_void) = on_thread;

unsafe extern "system" fn on_thread(_: *mut c_void, reason: u32, _: *mut c_void) {
    if reason != DLL_THREAD_ATTACH && reason != DLL_THREAD_DETACH {
        return;
    }
    if EXITING.load(Ordering::SeqCst) {
        return; // our exit watchdog, or threads ending with the process
    }
    // A snapshot: title code may map and attach another DLL from here.
    let notify = attached_images();
    if notify.is_empty() {
        return;
    }
    // Rust's std sets its threads up after this callback and aborts the
    // process if their per-thread state was touched first (the current
    // thread's id, which our logging asks for when title code calls us).
    // That state is put back as the thread started.
    let saved = (reason == DLL_THREAD_ATTACH && started_by_rust()).then(OwnTls::save);
    // Starts in load order, ends in reverse, as the Windows loader does.
    let notify_one = |notices: &Arc<ImageNotices>| {
        if notices.attached.load(Ordering::SeqCst) {
            let dll_main = !notices.quiet.load(Ordering::SeqCst);
            // SAFETY: attached images are bound. Reentrant detach/unmap
            // invalidates the old registration before the next dispatch.
            unsafe { notices.image.notify(reason, dll_main, std::ptr::null_mut()) };
        }
    };
    if reason == DLL_THREAD_ATTACH {
        notify.iter().for_each(notify_one);
    } else {
        notify.iter().rev().for_each(notify_one);
    }
    if let Some(saved) = saved {
        saved.restore();
    }
}

/// Our own image's static TLS block on this thread (Rust's std keeps its
/// per-thread state there), as copied before title code ran.
struct OwnTls {
    block: *mut u8,
    saved: Vec<MaybeUninit<u8>>,
}

/// The TLS directory the linker emits for our image (the CRT's symbol).
#[repr(C)]
struct TlsDirectory {
    start: usize,
    end: usize,
    index: *const u32,
    callbacks: usize,
    zero_fill: u32,
    characteristics: u32,
}

unsafe extern "C" {
    static _tls_used: TlsDirectory;
}

impl OwnTls {
    fn save() -> OwnTls {
        let array: *const *mut u8;
        // SAFETY: reads TEB.ThreadLocalStoragePointer; the directory is our
        // image's, and Windows set up its block on this thread before
        // calling TLS callbacks.
        let (block, len) = unsafe {
            std::arch::asm!("mov {}, gs:[0x58]", out(reg) array, options(nostack, readonly));
            let dir = &raw const _tls_used;
            let block = *array.add((*dir).index.read() as usize);
            (block, (*dir).end - (*dir).start + (*dir).zero_fill as usize)
        };
        // TLS may contain uninitialized padding. Copy its representation
        // without constructing a slice of initialized bytes over the block.
        let mut saved = Vec::with_capacity(len);
        // SAFETY: the linker describes this thread's live block and the new
        // allocation holds len MaybeUninit<u8> elements.
        unsafe {
            std::ptr::copy_nonoverlapping(block.cast(), saved.as_mut_ptr(), len);
            saved.set_len(len);
        }
        OwnTls { block, saved }
    }

    fn restore(self) {
        // SAFETY: the same block on the same thread, no reference to it held.
        unsafe {
            std::ptr::copy_nonoverlapping(self.saved.as_ptr(), self.block.cast(), self.saved.len())
        };
    }
}

/// Whether Rust's std started this thread (its start routine is in our
/// exe): the title's main thread and our runtime's workers, which run
/// title callbacks.
fn started_by_rust() -> bool {
    // SAFETY: the calling thread's pseudo handle.
    starts_in(
        unsafe { windows_sys::Win32::System::Threading::GetCurrentThread() },
        c"",
    ) == Some(true)
}

/// Whether `thread`'s start routine lies in the module named `module`
/// (empty: our exe).
fn starts_in(
    thread: windows_sys::Win32::Foundation::HANDLE,
    module: &std::ffi::CStr,
) -> Option<bool> {
    use windows_sys::Wdk::System::Threading::{
        NtQueryInformationThread, ThreadQuerySetWin32StartAddress,
    };
    use windows_sys::Win32::System::LibraryLoader::{
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        GetModuleHandleA, GetModuleHandleExW,
    };

    let mut start = 0usize;
    // SAFETY: writes one pointer-sized start address.
    let status = unsafe {
        NtQueryInformationThread(
            thread,
            ThreadQuerySetWin32StartAddress,
            (&raw mut start).cast(),
            size_of::<usize>() as u32,
            std::ptr::null_mut(),
        )
    };
    let mut found = std::ptr::null_mut();
    let name = if module.is_empty() {
        std::ptr::null()
    } else {
        module.as_ptr().cast()
    };
    if status < 0 || start == 0 {
        return None;
    }
    // SAFETY: looks up the module containing an address, taking no reference.
    if unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            start as *const u16,
            &mut found,
        )
    } == 0
    {
        return None;
    }
    // SAFETY: a module handle by name (or our exe's); no reference taken.
    let target = unsafe { GetModuleHandleA(name) };
    (!target.is_null()).then_some(found == target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};
    use std::sync::atomic::AtomicUsize;
    use windows_sys::Win32::Foundation::{WAIT_OBJECT_0, WAIT_TIMEOUT};
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcessId, GetCurrentThread, GetExitCodeThread, GetThreadId, INFINITE, Sleep,
        WaitForSingleObject,
    };

    #[test]
    fn snapshot_requires_the_complete_owner_field() {
        use windows_sys::Win32::System::Diagnostics::ToolHelp::THREADENTRY32;
        let end = std::mem::offset_of!(THREADENTRY32, th32OwnerProcessID) as u32 + 4;
        assert!(!has_owner_field(0));
        assert!(!has_owner_field(end - 1));
        assert!(has_owner_field(end));
        assert!(has_owner_field(size_of::<THREADENTRY32>() as u32));
    }

    #[test]
    fn classification_requires_a_valid_handle_owner_and_module() {
        // SAFETY: plain queries for this test thread.
        let (thread, pid) = unsafe { (GetCurrentThread(), GetCurrentProcessId()) };
        assert_eq!(belongs_to_process(thread, pid), Ok(true));
        assert_eq!(belongs_to_process(thread, pid.wrapping_add(1)), Ok(false));
        assert_eq!(belongs_to_process(std::ptr::null_mut(), 0), Err(()));
        assert_eq!(starts_in(std::ptr::null_mut(), c"ntdll.dll"), None);
        assert_eq!(starts_in(thread, c"victory-missing-test-module.dll"), None);
        assert_eq!(starts_in(thread, c""), Some(true));
        assert_eq!(starts_in(thread, c"ntdll.dll"), Some(false));
    }

    fn child(test: &str, marker: &str) {
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test, "--nocapture"])
            .env("VICTORY_THREAD_TEST", test)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let start = Instant::now();
        while child.try_wait().unwrap().is_none() {
            if start.elapsed() > Duration::from_secs(20) {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("{test} timed out");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let out = child.wait_with_output().unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success() && stdout.contains(marker),
            "{test} exited with {}\n{stdout}\n{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
    }

    fn in_child(test: &str) -> bool {
        std::env::var("VICTORY_THREAD_TEST").as_deref() == Ok(test)
    }

    fn sleeper() -> std::thread::JoinHandle<()> {
        let started = Arc::new(AtomicBool::new(false));
        let ready = started.clone();
        let thread = std::thread::spawn(move || {
            ready.store(true, Ordering::Release);
            // SAFETY: sleeps without holding application locks.
            unsafe { Sleep(INFINITE) };
        });
        while !started.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        thread
    }

    #[test]
    fn exit_ends_the_other_threads_first() {
        const TEST: &str = "threads::tests::exit_ends_the_other_threads_first";
        if !in_child(TEST) {
            return child(TEST, "ENDED KEPT");
        }
        let ends = sleeper();
        let kept = sleeper();
        // SAFETY: both JoinHandles own live thread handles.
        let kept_id = unsafe { GetThreadId(kept.as_raw_handle()) };
        assert_ne!(kept_id, 0);
        end_other_threads(&[kept_id]).unwrap();
        let mut code = u32::MAX;
        // SAFETY: the handles are still owned here and code is writable.
        unsafe {
            assert_eq!(WaitForSingleObject(ends.as_raw_handle(), 0), WAIT_OBJECT_0);
            assert_ne!(GetExitCodeThread(ends.as_raw_handle(), &mut code), 0);
            assert_eq!(WaitForSingleObject(kept.as_raw_handle(), 0), WAIT_TIMEOUT);
        }
        assert_eq!(code, 0);
        println!("ENDED KEPT");
        std::process::exit(0);
    }

    #[test]
    fn empty_process_detach_keeps_host_threads_alive() {
        const TEST: &str = "threads::tests::empty_process_detach_keeps_host_threads_alive";
        if !in_child(TEST) {
            return child(TEST, "HOST RUNNING");
        }
        let host = sleeper();
        detach_all();
        // SAFETY: the JoinHandle still owns this thread handle.
        let status = unsafe { WaitForSingleObject(host.as_raw_handle(), 0) };
        assert_eq!(status, WAIT_TIMEOUT);
        println!("HOST RUNNING");
        std::process::exit(0);
    }

    static EVENTS: Mutex<Vec<(usize, u32, bool)>> = Mutex::new(Vec::new());
    static QUIET_NEXT: AtomicUsize = AtomicUsize::new(0);
    static REMOVE_NEXT: AtomicUsize = AtomicUsize::new(0);
    static REMOVE_ON_EXIT: AtomicUsize = AtomicUsize::new(0);
    static FIRST_DLL: AtomicUsize = AtomicUsize::new(0);
    thread_local! {
        static CALLBACK_TLS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }
    static TITLE_EXIT: AtomicBool = AtomicBool::new(false);

    extern "C" fn title_exit() {
        TITLE_EXIT.store(true, Ordering::SeqCst);
    }

    extern "system" fn dll_main(module: *mut c_void, reason: u32, reserved: *mut c_void) -> i32 {
        EVENTS
            .lock()
            .unwrap()
            .push((module as usize, reason, !reserved.is_null()));
        if reason == crate::DLL_PROCESS_ATTACH
            && module as usize == FIRST_DLL.load(Ordering::SeqCst)
        {
            // SAFETY: the synthetic callback lives for the process lifetime.
            assert_eq!(unsafe { atexit(title_exit) }, 0);
        }
        if reason == DLL_PROCESS_DETACH {
            let remove = REMOVE_ON_EXIT.swap(0, Ordering::SeqCst);
            if remove != 0 {
                unmapped(remove);
            }
        }
        if reason == DLL_THREAD_ATTACH {
            CALLBACK_TLS.set(7);
            let quiet = QUIET_NEXT.swap(0, Ordering::SeqCst);
            if quiet != 0 {
                assert_eq!(disable_thread_calls(quiet), Some(true));
            }
            let remove = REMOVE_NEXT.swap(0, Ordering::SeqCst);
            if remove != 0 {
                detached(remove);
                unmapped(remove);
            }
        }
        1
    }

    extern "system" fn exe_tls(module: *mut c_void, reason: u32, reserved: *mut c_void) {
        EVENTS
            .lock()
            .unwrap()
            .push((module as usize, reason, !reserved.is_null()));
        if reason == DLL_PROCESS_DETACH {
            assert!(TITLE_EXIT.load(Ordering::SeqCst));
            let events = std::mem::take(&mut *EVENTS.lock().unwrap());
            let bases: Vec<_> = events
                .into_iter()
                .map(|(base, r, terminating)| {
                    assert_eq!(r, DLL_PROCESS_DETACH);
                    assert!(terminating);
                    base
                })
                .collect();
            let base = module as usize;
            assert_eq!(bases, [base - 2, base - 3, base]);
            println!("DLLS THEN EXE");
        }
    }

    #[test]
    fn registry_lifecycle_and_reentrant_callbacks() {
        const TEST: &str = "threads::tests::registry_lifecycle_and_reentrant_callbacks";
        if !in_child(TEST) {
            return child(TEST, "DLLS THEN EXE");
        }
        // Synthetic descriptors use real functions in this test executable.
        // The module argument is only an opaque identity in those callbacks.
        let bytes = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let exe_info = victory_pe::ImageInfo::parse(&bytes).unwrap();
        // SAFETY: the host executable is mapped for the process lifetime.
        let base = unsafe {
            windows_sys::Win32::System::LibraryLoader::GetModuleHandleA(std::ptr::null())
        } as usize;
        let make_dll = |offset| {
            let address = base + offset;
            let mut info = exe_info.clone();
            info.characteristics |= victory_pe::layout::file_flags::DLL;
            info.entry_rva = u32::try_from(dll_main as *const () as usize - address).unwrap();
            info.tls = None;
            LoadedImage::synthetic(address as _, info)
        };
        let (first, quiet, removed) = (make_dll(0), make_dll(1), make_dll(2));
        let mut info = exe_info;
        let exe_base = base + 3;
        info.tls.as_mut().unwrap().callback_rvas =
            vec![u32::try_from(exe_tls as *const () as usize - exe_base).unwrap()];
        let exe = LoadedImage::synthetic(exe_base as _, info);
        for image in [&removed, &exe, &quiet, &first] {
            mapped(image);
        }
        assert_eq!(disable_thread_calls(0), None);
        assert_eq!(disable_thread_calls(exe_base), Some(false));
        FIRST_DLL.store(first.base(), Ordering::SeqCst);
        for image in [&first, &quiet, &removed] {
            // SAFETY: these fixtures call only dll_main above.
            unsafe { image.attach() }.unwrap();
        }
        attached(exe_base);
        mapped(&quiet);
        unmapped(quiet.base());
        attached(first.base());
        EVENTS.lock().unwrap().clear();
        std::thread::spawn(|| assert_eq!(CALLBACK_TLS.get(), 0))
            .join()
            .unwrap();
        let start = [first.base(), quiet.base(), removed.base(), exe_base];
        let expected: Vec<_> = start
            .iter()
            .map(|&b| (b, DLL_THREAD_ATTACH, false))
            .chain(start.iter().rev().map(|&b| (b, DLL_THREAD_DETACH, false)))
            .collect();
        assert_eq!(*EVENTS.lock().unwrap(), expected);
        EVENTS.lock().unwrap().clear();
        QUIET_NEXT.store(quiet.base(), Ordering::SeqCst);
        REMOVE_NEXT.store(removed.base(), Ordering::SeqCst);
        std::thread::spawn(|| assert_eq!(CALLBACK_TLS.get(), 0))
            .join()
            .unwrap();
        assert_eq!(
            *EVENTS.lock().unwrap(),
            [
                (first.base(), DLL_THREAD_ATTACH, false),
                (exe_base, DLL_THREAD_ATTACH, false),
                (exe_base, DLL_THREAD_DETACH, false),
                (first.base(), DLL_THREAD_DETACH, false),
            ]
        );
        assert_eq!(disable_thread_calls(removed.base()), None);
        EVENTS.lock().unwrap().clear();
        // A new attachment at the same base must not revive old snapshots.
        let old = attached_images()
            .into_iter()
            .find(|n| n.image.base() == quiet.base())
            .unwrap();
        mapped(&removed);
        attached(removed.base());
        detached(quiet.base());
        attached(quiet.base());
        assert!(!old.attached.load(Ordering::SeqCst));
        // The quiet DLL's process detach still runs. It unloads the next
        // image before that image's turn in the process-detach snapshot.
        REMOVE_ON_EXIT.store(removed.base(), Ordering::SeqCst);
        unsafe extern "C" {
            fn exit(code: i32) -> !;
        }
        // SAFETY: this isolated child exercises the same host CRT exit as a title.
        unsafe { exit(0) }
    }
}
