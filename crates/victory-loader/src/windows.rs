//! Loading a prepared image as a real Windows module and binding it.

use std::collections::HashSet;
use std::ffi::{CString, c_void};
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use anyhow::{Context, Result, bail};
use victory_pe::{ImageInfo, ImportSymbol};
use windows_sys::Win32::Foundation::{FreeLibrary, HMODULE};
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryExW};
use windows_sys::Win32::System::Memory::{
    MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualProtect, VirtualQuery,
};

use crate::thunks::{ImportSite, ThunkBlock, UnimplementedPolicy};
use crate::{
    DLL_PROCESS_ATTACH, DLL_PROCESS_DETACH, ImportResolver, PreparedImage, Provenance, Resolution,
    prepare,
};

/// A title image mapped into this process. A clone names the same mapping
/// (no reference of its own): unload one copy only.
#[derive(Debug, Clone)]
pub struct LoadedImage {
    module: HMODULE,
    path: PathBuf,
    info: ImageInfo,
}

// SAFETY: HMODULE is a process-wide handle (the base address); using it from
// any thread is fine.
unsafe impl Send for LoadedImage {}
// SAFETY: as above; `LoadedImage` has no interior mutability.
unsafe impl Sync for LoadedImage {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingImport {
    pub dll: String,
    pub symbol: String,
}

/// Outcome of [`LoadedImage::bind_imports`].
#[derive(Debug, Default, Clone)]
pub struct BindReport {
    pub hle: usize,
    pub host: usize,
    pub title: usize,
    pub missing: Vec<MissingImport>,
}

/// Writes the prepared copy into `dir` and maps it with the Windows loader.
///
/// `dir` receives derived game data, so it must be outside the repository.
pub fn load(prepared: &PreparedImage, dir: &Path) -> Result<LoadedImage> {
    victory_core::ensure_outside_repo(dir)?;
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let path = dir.join(prepared.file_name());
    write_if_changed(&path, &prepared.bytes)?;

    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    // SAFETY: `wide` is a NUL-terminated path. The image has no entry point
    // and no imports, so loading it runs no title code.
    let module = unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), 0) };
    if module.is_null() {
        bail!(
            "LoadLibraryExW({}) failed: {}{}",
            path.display(),
            std::io::Error::last_os_error(),
            if prepared.info.has_relocations {
                ""
            } else {
                " (image has no relocations; its preferred base may be taken)"
            }
        );
    }
    let loaded = LoadedImage {
        module,
        path,
        info: prepared.info.clone(),
    };
    // A DLL indexes its TLS properly; only an exe assumes index 0.
    if !loaded.info.is_dll() {
        // SAFETY: just mapped, and no title code has run.
        unsafe { crate::tls::redirect(loaded.base(), &loaded.info) }?;
    }
    if std::env::var_os(crate::hooks::NO_WATCHDOG_ENV).is_some_and(|v| !v.is_empty() && v != "0") {
        // SAFETY: just mapped, and no title code has run.
        let applied = unsafe { crate::hooks::disarm_watchdog(loaded.base(), &prepared.original_hash) };
        tracing::info!(applied, "title watchdog hooks ({})", crate::hooks::NO_WATCHDOG_ENV);
    }
    tracing::info!(
        base = format_args!("{:#x}", loaded.base()),
        preferred = format_args!("{:#x}", loaded.info.image_base),
        path = %loaded.path.display(),
        "image mapped"
    );
    crate::threads::mapped(&loaded);
    Ok(loaded)
}

/// Maps the DLLs the title ships in `title_dir` that `main` imports,
/// directly or through each other, in initialisation order (dependencies
/// first). Only DLL names `wanted` accepts are considered, so DLLs the host
/// or our runtime provide stay with them. Imports are not bound.
pub fn load_title_dlls(
    main: &ImageInfo,
    title_dir: &Path,
    cache_dir: &Path,
    wanted: &dyn Fn(&str) -> bool,
) -> Result<Vec<(String, LoadedImage)>> {
    let mut loaded = Vec::new();
    let dirs = (title_dir, cache_dir);
    map_imports(main, dirs, wanted, &mut HashSet::new(), &mut loaded)?;
    Ok(loaded)
}

/// Maps the DLL `name` from `title_dir` like [`load_title_dlls`] does, with
/// the title DLLs it imports (dependencies first, `name` last); empty if
/// `wanted` refuses it or the folder has no such file. For DLLs a title
/// loads itself; `wanted` must refuse those already mapped.
pub fn load_title_dll(
    name: &str,
    title_dir: &Path,
    cache_dir: &Path,
    wanted: &dyn Fn(&str) -> bool,
) -> Result<Vec<(String, LoadedImage)>> {
    let mut loaded = Vec::new();
    let dirs = (title_dir, cache_dir);
    map_dll(name, dirs, wanted, &mut HashSet::new(), &mut loaded)?;
    Ok(loaded)
}

fn map_imports(
    info: &ImageInfo,
    dirs: (&Path, &Path),
    wanted: &dyn Fn(&str) -> bool,
    seen: &mut HashSet<String>,
    loaded: &mut Vec<(String, LoadedImage)>,
) -> Result<()> {
    for module in info.imports.iter().chain(&info.delay_imports) {
        map_dll(&module.dll, dirs, wanted, seen, loaded)?;
    }
    Ok(())
}

fn map_dll(
    name: &str,
    dirs: (&Path, &Path),
    wanted: &dyn Fn(&str) -> bool,
    seen: &mut HashSet<String>,
    loaded: &mut Vec<(String, LoadedImage)>,
) -> Result<()> {
    let name = name.to_ascii_lowercase();
    let path = dirs.0.join(&name);
    if !seen.insert(name.clone()) || !wanted(&name) || !path.is_file() {
        return Ok(());
    }
    let original = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let prepared = prepare(&original).with_context(|| format!("preparing {}", path.display()))?;
    let image = load(&prepared, dirs.1).with_context(|| format!("loading {}", path.display()))?;
    map_imports(&image.info, dirs, wanted, seen, loaded)?;
    loaded.push((name, image));
    Ok(())
}

/// Resolves through `inner`, then through the exports of the title's DLLs.
pub struct WithTitleDlls<'a> {
    pub inner: &'a dyn ImportResolver,
    pub dlls: &'a [(String, LoadedImage)],
}

impl ImportResolver for WithTitleDlls<'_> {
    fn resolve(&self, dll: &str, symbol: &ImportSymbol) -> Resolution {
        let found = self.inner.resolve(dll, symbol);
        if found != Resolution::Missing {
            return found;
        }
        self.dlls
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(dll))
            .and_then(|(_, image)| image.export_symbol(symbol))
            .map_or(Resolution::Missing, |address| Resolution::Found {
                address,
                provenance: Provenance::Title,
            })
    }
}

fn write_if_changed(path: &Path, bytes: &[u8]) -> Result<()> {
    crate::publish::write_if_changed(path, bytes)
        .with_context(|| format!("publishing {}", path.display()))
}

/// Binds a slot before the title can call through it. The parser checks
/// the slot's extent, but does not enforce u64 alignment.
///
/// # Safety
/// `slot` must point to eight writable bytes that no other thread accesses.
unsafe fn write_iat_slot(slot: *mut u8, address: u64) {
    // SAFETY: guaranteed by the caller; no alignment is required.
    unsafe { slot.cast::<u64>().write_unaligned(address) };
}

/// Every protection region touched while making an IAT slot writable.
/// Restore earlier changes too if setting up a later region fails.
struct IatProtections(Vec<(*const c_void, usize, u32)>);

impl IatProtections {
    fn restore(&mut self) -> Result<()> {
        let mut error = None;
        while let Some((at, len, protection)) = self.0.pop() {
            let mut ignored = 0;
            // SAFETY: these are the exact regions whose protections we changed.
            if unsafe { VirtualProtect(at, len, protection, &mut ignored) } == 0 && error.is_none()
            {
                error = Some(std::io::Error::last_os_error());
            }
        }
        match error {
            Some(error) => Err(error).context("restoring IAT protection"),
            None => Ok(()),
        }
    }
}

impl Drop for IatProtections {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

/// # Safety
/// The eight-byte range at `slot` must be in reserved memory owned by the
/// caller. No other thread may access it or change its protections while
/// it is being bound. Uncommitted pages are reported as an error.
unsafe fn write_protected_iat_slot(slot: *mut u8, address: u64) -> Result<()> {
    let end = (slot as usize)
        .checked_add(8)
        .context("IAT slot address overflow")?;
    let mut at = slot as usize;
    let mut protections = IatProtections(Vec::with_capacity(2));
    while at < end {
        // SAFETY: zero is valid for this output structure.
        let mut region: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: queries the current slot segment; `region` is a sized output.
        if unsafe { VirtualQuery(at as _, &mut region, size_of_val(&region)) } == 0 {
            return Err(std::io::Error::last_os_error()).context("querying IAT protection");
        }
        let region_end = (region.BaseAddress as usize)
            .checked_add(region.RegionSize)
            .filter(|&end| end > at)
            .context("invalid IAT protection region")?;
        let len = end.min(region_end) - at;
        let mut old = 0;
        // SAFETY: this segment ends within one queried protection region.
        if unsafe { VirtualProtect(at as _, len, PAGE_READWRITE, &mut old) } == 0 {
            return Err(std::io::Error::last_os_error()).context("making IAT writable");
        }
        protections.0.push((at as _, len, old));
        at += len;
    }
    // SAFETY: every byte is writable now; the caller guarantees exclusive access.
    unsafe { write_iat_slot(slot, address) };
    protections.restore()
}

impl LoadedImage {
    #[cfg(test)]
    pub(super) fn synthetic(module: HMODULE, info: ImageInfo) -> Self {
        Self {
            module,
            path: PathBuf::new(),
            info,
        }
    }

    pub fn base(&self) -> usize {
        self.module as usize
    }

    /// Parsed from the original image (imports, entry point, TLS callbacks).
    pub fn info(&self) -> &ImageInfo {
        &self.info
    }

    /// Address of an export of the image, if it has one by that name.
    pub fn export(&self, name: &str) -> Option<usize> {
        let name = CString::new(name).ok()?;
        // SAFETY: valid module handle and NUL-terminated name.
        unsafe { GetProcAddress(self.module, name.as_ptr().cast()) }
            .map(|f| f as *const () as usize)
    }

    /// Address of an export, by name or ordinal.
    pub fn export_symbol(&self, symbol: &ImportSymbol) -> Option<usize> {
        match symbol {
            ImportSymbol::Name { name, .. } => self.export(name),
            // SAFETY: MAKEINTRESOURCE-style ordinal lookup on a valid module.
            ImportSymbol::Ordinal(o) => unsafe { GetProcAddress(self.module, *o as usize as _) }
                .map(|f| f as *const () as usize),
        }
    }

    /// Fills every IAT slot of the image.
    ///
    /// Each import goes to `resolver`; whatever it cannot resolve gets an
    /// unimplemented-import stub that behaves according to `policy`.
    pub fn bind_imports(
        &self,
        resolver: &dyn ImportResolver,
        policy: UnimplementedPolicy,
    ) -> Result<BindReport> {
        let mut report = BindReport::default();
        let mut slots: Vec<(u32, usize)> = Vec::with_capacity(self.info.import_count());
        let mut unresolved: Vec<(u32, &str, &ImportSymbol)> = Vec::new();

        // Delay-load slots are bound eagerly too, so the title's own
        // delay-load helper (which would ask the host) never runs.
        for module in self.info.imports.iter().chain(&self.info.delay_imports) {
            for entry in &module.entries {
                match resolver.resolve(&module.dll, &entry.symbol) {
                    Resolution::Found {
                        address,
                        provenance,
                    } => {
                        match provenance {
                            Provenance::Hle => report.hle += 1,
                            Provenance::Host => report.host += 1,
                            Provenance::Title => report.title += 1,
                        }
                        slots.push((entry.iat_slot_rva, address));
                    }
                    Resolution::Missing => {
                        unresolved.push((entry.iat_slot_rva, &module.dll, &entry.symbol));
                    }
                }
            }
        }

        if !unresolved.is_empty() {
            let mut block = ThunkBlock::new(unresolved.len())?;
            for (slot, dll, symbol) in unresolved {
                let site: &'static ImportSite = Box::leak(Box::new(ImportSite {
                    dll: dll.to_owned(),
                    symbol: symbol.to_string(),
                    policy,
                    calls: AtomicU64::new(0),
                }));
                slots.push((slot, block.emit(site)?));
                report.missing.push(MissingImport {
                    dll: site.dll.clone(),
                    symbol: site.symbol.clone(),
                });
            }
            block.seal()?;
        }

        self.write_slots(&slots)?;
        tracing::info!(
            hle = report.hle,
            host = report.host,
            missing = report.missing.len(),
            "imports bound"
        );
        Ok(report)
    }

    /// Protects and restores each slot on its own: import tables can sit in
    /// sections with different protections (delay-load tables in `.didat`).
    fn write_slots(&self, slots: &[(u32, usize)]) -> Result<()> {
        for &(rva, address) in slots {
            let slot = (self.base() + rva as usize) as *mut u8;
            // SAFETY: the parser bounds every slot, and the image is bound
            // before title code can access its imports.
            unsafe { write_protected_iat_slot(slot, address as u64) }
                .with_context(|| format!("binding IAT slot {rva:#x}"))?;
        }
        Ok(())
    }

    /// Calls the image's TLS callbacks with `reason` (e.g. `DLL_PROCESS_ATTACH`).
    ///
    /// # Safety
    /// Runs title code. Imports must be bound first.
    pub unsafe fn run_tls_callbacks(&self, reason: u32) {
        // SAFETY: forwarded from the caller.
        unsafe { self.tls_callbacks(reason, std::ptr::null_mut()) }
    }

    /// # Safety
    /// See [`Self::run_tls_callbacks`].
    unsafe fn tls_callbacks(&self, reason: u32, reserved: *mut c_void) {
        let Some(tls) = &self.info.tls else { return };
        for &rva in &tls.callback_rvas {
            type TlsCallback = unsafe extern "system" fn(*mut c_void, u32, *mut c_void);
            // SAFETY: the RVA came from the image's own TLS directory.
            let callback: TlsCallback = unsafe { std::mem::transmute(self.base() + rva as usize) };
            // SAFETY: the caller guarantees imports are bound.
            unsafe { callback(self.module, reason, reserved) };
        }
    }

    /// Initialises a DLL as the Windows loader would: its TLS callbacks,
    /// then `DllMain(DLL_PROCESS_ATTACH)`. From then on it hears of every
    /// thread start and end (see `threads`).
    ///
    /// # Safety
    /// Runs title code. Imports must be bound first, and the DLLs this one
    /// depends on attached.
    pub unsafe fn attach(&self) -> Result<()> {
        crate::threads::register_process_detach()?;
        // SAFETY: forwarded from the caller.
        if unsafe { self.notify(DLL_PROCESS_ATTACH, true, std::ptr::null_mut()) } == Some(0) {
            bail!("DllMain returned FALSE");
        }
        crate::threads::attached(self.base());
        Ok(())
    }

    /// `DLL_PROCESS_DETACH` as `FreeLibrary` sends it (TLS callbacks, then
    /// `DllMain`); the image hears of no more threads.
    ///
    /// # Safety
    /// Runs title code. The image must have been attached (or its attach
    /// failed), and no other thread may be using it.
    pub unsafe fn detach(&self) {
        crate::threads::detached(self.base());
        // SAFETY: forwarded from the caller.
        unsafe { self.notify(DLL_PROCESS_DETACH, true, std::ptr::null_mut()) };
    }

    /// Runs the image's TLS callbacks with `reason`, then, if `dll_main` and
    /// the image is a DLL with an entry point, its `DllMain`, whose result
    /// is returned. `reserved` is the entry points' third argument.
    ///
    /// # Safety
    /// Runs title code. Imports must be bound first.
    pub(crate) unsafe fn notify(
        &self,
        reason: u32,
        dll_main: bool,
        reserved: *mut c_void,
    ) -> Option<i32> {
        // SAFETY: forwarded from the caller.
        unsafe { self.tls_callbacks(reason, reserved) };
        if !dll_main || !self.info.is_dll() || self.info.entry_rva == 0 {
            return None;
        }
        type DllMain = unsafe extern "system" fn(*mut c_void, u32, *mut c_void) -> i32;
        // SAFETY: the entry RVA came from the original image header.
        let main: DllMain =
            unsafe { std::mem::transmute(self.base() + self.info.entry_rva as usize) };
        // SAFETY: forwarded from the caller.
        Some(unsafe { main(self.module, reason, reserved) })
    }

    /// Starts the title: on a new thread with the stack size from the image
    /// header, attaches `dlls` in order (see [`Self::attach`]), runs the
    /// image's TLS callbacks (`DLL_PROCESS_ATTACH`), then its original entry
    /// point. Returns the thread so the caller can join it.
    ///
    /// # Safety
    /// Runs title code. Imports of the image and of `dlls` must be bound.
    pub unsafe fn spawn_main(
        self: Arc<Self>,
        dlls: Vec<(String, LoadedImage)>,
    ) -> Result<std::thread::JoinHandle<Result<u32>>> {
        if self.info.entry_rva == 0 {
            bail!("image has no entry point");
        }
        crate::threads::register_process_detach()?;
        let stack = usize::try_from(self.info.stack_reserve)
            .unwrap_or(usize::MAX)
            .max(1 << 20);
        std::thread::Builder::new()
            .name("title-main".into())
            .stack_size(stack)
            .spawn(move || {
                for (name, dll) in &dlls {
                    // SAFETY: forwarded from the caller of `spawn_main`.
                    unsafe { dll.attach() }.with_context(|| format!("attaching {name}"))?;
                }
                // SAFETY: forwarded from the caller of `spawn_main`.
                unsafe { self.run_tls_callbacks(DLL_PROCESS_ATTACH) };
                crate::threads::attached(self.base());
                type Entry = unsafe extern "system" fn(*mut c_void) -> u32;
                // SAFETY: the entry RVA came from the original image header.
                let entry: Entry =
                    unsafe { std::mem::transmute(self.base() + self.info.entry_rva as usize) };
                // SAFETY: forwarded from the caller of `spawn_main`.
                Ok(unsafe { entry(std::ptr::null_mut()) })
            })
            .context("spawning title main thread")
    }

    /// Unmaps the image.
    ///
    /// # Safety
    /// No title code may be running or run later, and no pointer into the
    /// image may be used afterwards.
    pub unsafe fn unload(self) {
        crate::threads::unmapped(self.base());
        // SAFETY: guaranteed by the caller.
        unsafe { FreeLibrary(self.module) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READONLY, VirtualAlloc, VirtualFree,
    };
    use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};

    struct Pages {
        base: *mut u8,
        page_size: usize,
    }

    impl Pages {
        fn new(commit_second: bool) -> Self {
            // SAFETY: zeroed output for the system query.
            let mut info: SYSTEM_INFO = unsafe { std::mem::zeroed() };
            // SAFETY: valid, sized output.
            unsafe { GetSystemInfo(&mut info) };
            let page_size = info.dwPageSize as usize;
            assert!(page_size >= 8);
            // SAFETY: a fresh two-page reservation owned only by this test.
            let base = unsafe {
                VirtualAlloc(std::ptr::null(), 2 * page_size, MEM_RESERVE, PAGE_READWRITE)
            } as *mut u8;
            assert!(!base.is_null());
            let pages = Self { base, page_size };
            let committed = if commit_second {
                2 * page_size
            } else {
                page_size
            };
            // SAFETY: commits bytes within our reservation.
            let result =
                unsafe { VirtualAlloc(base.cast(), committed, MEM_COMMIT, PAGE_READWRITE) };
            assert!(!result.is_null());
            // SAFETY: the committed region is writable and exclusively ours.
            unsafe { std::ptr::write_bytes(base, 0xA5, committed) };
            pages
        }

        fn protect(&self, page: usize, protection: u32) {
            assert!(page < 2);
            let mut old = 0;
            // SAFETY: one committed page in our reservation.
            let result = unsafe {
                VirtualProtect(
                    self.base.add(page * self.page_size).cast(),
                    self.page_size,
                    protection,
                    &mut old,
                )
            };
            assert_ne!(result, 0);
        }

        fn info(&self, page: usize) -> MEMORY_BASIC_INFORMATION {
            assert!(page < 2);
            // SAFETY: zeroed output for the memory query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a queried address in our reservation and valid output.
            let result = unsafe {
                VirtualQuery(
                    self.base.add(page * self.page_size).cast(),
                    &mut info,
                    size_of_val(&info),
                )
            };
            assert_ne!(result, 0);
            info
        }
    }

    impl Drop for Pages {
        fn drop(&mut self) {
            // SAFETY: releases our entire reservation after its last use.
            unsafe { VirtualFree(self.base.cast(), 0, MEM_RELEASE) };
        }
    }

    #[test]
    fn iat_binding_preserves_each_crossed_page_protection() {
        for (first, second) in [
            (PAGE_READONLY, PAGE_READWRITE),
            (PAGE_READWRITE, PAGE_READONLY),
            (PAGE_READONLY, PAGE_READONLY),
        ] {
            let pages = Pages::new(true);
            pages.protect(0, first);
            pages.protect(1, second);
            let start = pages.page_size - 4;
            let address = 0x1122_3344_5566_7788u64;
            // SAFETY: both pages belong to this test and no other thread accesses them.
            unsafe { write_protected_iat_slot(pages.base.add(start), address) }.unwrap();
            assert_eq!(pages.info(0).Protect, first);
            assert_eq!(pages.info(1).Protect, second);
            // SAFETY: both pages remain readable, and no Rust reference aliases this mapping.
            let bytes = unsafe { std::slice::from_raw_parts(pages.base, 2 * pages.page_size) };
            assert!(bytes[..start].iter().all(|&b| b == 0xA5));
            assert_eq!(bytes[start..start + 8], address.to_le_bytes());
            assert!(bytes[start + 8..].iter().all(|&b| b == 0xA5));
        }
    }

    #[test]
    fn iat_binding_failure_restores_earlier_pages_without_writing() {
        let pages = Pages::new(false);
        pages.protect(0, PAGE_READONLY);
        // SAFETY: the reservation is ours, and the helper rejects its uncommitted second page.
        let result = unsafe { write_protected_iat_slot(pages.base.add(pages.page_size - 4), 42) };
        assert!(result.is_err());
        assert_eq!(pages.info(0).Protect, PAGE_READONLY);
        assert_eq!(pages.info(1).State, MEM_RESERVE);
        // SAFETY: the first page is still committed and readable.
        let bytes = unsafe { std::slice::from_raw_parts(pages.base, pages.page_size) };
        assert!(bytes.iter().all(|&b| b == 0xA5));
    }

    #[test]
    fn iat_slot_writes_accept_every_byte_alignment() {
        #[repr(C, align(8))]
        struct Aligned([u8; 24]);

        let address = 0x1122_3344_5566_7788u64;
        for offset in 0..8 {
            let mut bytes = Aligned([0xA5; 24]);
            let start = 8 + offset;
            // SAFETY: the eight-byte destination is inside our exclusively
            // borrowed buffer, including when its address is unaligned.
            unsafe { write_iat_slot(bytes.0.as_mut_ptr().add(start), address) };
            assert!(bytes.0[..start].iter().all(|&b| b == 0xA5));
            assert_eq!(bytes.0[start..start + 8], address.to_le_bytes());
            assert!(bytes.0[start + 8..].iter().all(|&b| b == 0xA5));
        }
    }
}
