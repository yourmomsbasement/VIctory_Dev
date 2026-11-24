//! Stubs for imports nobody implements yet.
//!
//! Each unresolved import gets 32 bytes of generated code:
//!
//! ```text
//! mov rcx, <&'static ImportSite>   ; 48 B9 imm64
//! mov rax, <unimplemented_import>  ; 48 B8 imm64
//! jmp rax                          ; FF E0
//! ```
//!
//! The jump leaves RSP and the caller's shadow space intact, so the handler
//! returns straight to the caller. Bring-up then shows exactly which API to
//! implement next, rather than failing at load time.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, bail};
use windows_sys::Win32::System::Diagnostics::Debug::{
    DebugBreak, FlushInstructionCache, IsDebuggerPresent,
};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_READWRITE, VirtualAlloc,
    VirtualFree, VirtualProtect,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// What an unimplemented import does when called.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnimplementedPolicy {
    /// Log, break into an attached debugger, then abort. The default:
    /// silently continuing usually crashes later and further from the cause.
    #[default]
    Trap,
    /// Log once per import and return 0. Useful to push through bring-up.
    ReturnZero,
}

/// Identifies one unresolved import. Leaked: lives as long as the process.
#[derive(Debug)]
pub struct ImportSite {
    pub dll: String,
    pub symbol: String,
    pub policy: UnimplementedPolicy,
    pub calls: AtomicU64,
}

static UNIMPLEMENTED_CALLS: AtomicU64 = AtomicU64::new(0);

/// Total number of calls that reached an unimplemented-import stub.
pub fn unimplemented_call_count() -> u64 {
    UNIMPLEMENTED_CALLS.load(Ordering::Relaxed)
}

extern "system" fn unimplemented_import(site: &'static ImportSite) -> u64 {
    UNIMPLEMENTED_CALLS.fetch_add(1, Ordering::Relaxed);
    let previous = site.calls.fetch_add(1, Ordering::Relaxed);
    match site.policy {
        UnimplementedPolicy::ReturnZero => {
            if previous == 0 {
                tracing::warn!(dll = %site.dll, symbol = %site.symbol, "unimplemented import called, returning 0");
            }
            0
        }
        UnimplementedPolicy::Trap => {
            tracing::error!(dll = %site.dll, symbol = %site.symbol, "unimplemented import called");
            // Into the crash log too: `abort` ends the process without
            // running the handlers that write it.
            let (dll, symbol) = (&site.dll, &site.symbol);
            let text = format!("VIctory: unimplemented import {dll}!{symbol} called");
            crate::log_crash(&text);
            // SAFETY: plain Win32 calls without preconditions.
            unsafe {
                if IsDebuggerPresent() != 0 {
                    DebugBreak();
                }
            }
            std::process::abort();
        }
    }
}

const THUNK_SIZE: usize = 32;

/// Executable memory holding one stub per unresolved import.
pub(crate) struct ThunkBlock {
    base: *mut u8,
    len: usize,
    used: usize,
}

impl ThunkBlock {
    /// Reserves room for `count` stubs (writable until [`Self::seal`]).
    pub(crate) fn new(count: usize) -> Result<Self> {
        let len = count
            .checked_mul(THUNK_SIZE)
            .ok_or_else(|| anyhow::anyhow!("import stub block size overflow"))?;
        // SAFETY: fresh anonymous allocation; no existing memory is touched.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len.max(1), // VirtualAlloc rounds up to whole pages.
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            )
        } as *mut u8;
        if base.is_null() {
            bail!(
                "VirtualAlloc for import stubs failed: {}",
                std::io::Error::last_os_error()
            );
        }
        Ok(ThunkBlock { base, len, used: 0 })
    }

    /// Emits a stub for `site` and returns its address.
    pub(crate) fn emit(&mut self, site: &'static ImportSite) -> Result<usize> {
        if self.used == self.len {
            bail!("import stub block is full");
        }
        let mut code = [0xCC_u8; THUNK_SIZE]; // int3 padding
        code[0..2].copy_from_slice(&[0x48, 0xB9]);
        code[2..10].copy_from_slice(&(site as *const ImportSite as u64).to_le_bytes());
        code[10..12].copy_from_slice(&[0x48, 0xB8]);
        let handler = unimplemented_import as extern "system" fn(&'static ImportSite) -> u64;
        code[12..20].copy_from_slice(&(handler as *const () as u64).to_le_bytes());
        code[20..22].copy_from_slice(&[0xFF, 0xE0]);
        // SAFETY: `used + THUNK_SIZE <= len`, and the block is still writable.
        let at = unsafe {
            let at = self.base.add(self.used);
            std::ptr::copy_nonoverlapping(code.as_ptr(), at, THUNK_SIZE);
            at
        };
        self.used += THUNK_SIZE;
        Ok(at as usize)
    }

    /// Makes the block read+execute. The block is intentionally leaked:
    /// title code may hold these addresses for the life of the process.
    pub(crate) fn seal(self) -> Result<()> {
        self.seal_with_cache_flush(|base, len| {
            // SAFETY: flushing our own process's cache for the block being sealed.
            unsafe { FlushInstructionCache(GetCurrentProcess(), base, len) }
        })
    }

    fn seal_with_cache_flush(self, flush: impl FnOnce(*const c_void, usize) -> i32) -> Result<()> {
        if self.used == 0 {
            return Ok(());
        }
        let mut old = 0;
        // SAFETY: `base..base+len` is the allocation made in `new`.
        let ok = unsafe { VirtualProtect(self.base as _, self.len, PAGE_EXECUTE_READ, &mut old) };
        if ok == 0 {
            bail!(
                "VirtualProtect for import stubs failed: {}",
                std::io::Error::last_os_error()
            );
        }
        if flush(self.base.cast(), self.len) == 0 {
            bail!(
                "FlushInstructionCache for import stubs failed: {}",
                std::io::Error::last_os_error()
            );
        }
        // Only successfully sealed stubs can be published to the title.
        std::mem::forget(self);
        Ok(())
    }
}

impl Drop for ThunkBlock {
    fn drop(&mut self) {
        // SAFETY: this is the reservation base. MEM_RELEASE requires size zero.
        unsafe { VirtualFree(self.base.cast(), 0, MEM_RELEASE) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_DECOMMIT, MEM_FREE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };

    static SITE: ImportSite = ImportSite {
        dll: String::new(),
        symbol: String::new(),
        policy: UnimplementedPolicy::ReturnZero,
        calls: AtomicU64::new(0),
    };

    fn memory_info(base: *const u8) -> MEMORY_BASIC_INFORMATION {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        let size = size_of_val(&info);
        // SAFETY: VirtualQuery inspects the address without dereferencing it.
        assert_eq!(unsafe { VirtualQuery(base.cast(), &mut info, size) }, size);
        info
    }

    #[test]
    fn capacity_matches_the_requested_slot_count() {
        for count in [0, 1, 128, 129] {
            let mut block = ThunkBlock::new(count).unwrap();
            for slot in 0..count {
                // Repeated imports still occupy distinct, aligned slots.
                let at = block.emit(&SITE).unwrap();
                assert_eq!(at, block.base as usize + slot * THUNK_SIZE);
                assert_eq!(at % THUNK_SIZE, 0);
            }
            assert_eq!(
                block.emit(&SITE).unwrap_err().to_string(),
                "import stub block is full"
            );
        }
    }

    #[test]
    fn overflowing_slot_counts_fail_before_allocation() {
        for count in [usize::MAX / THUNK_SIZE + 1, usize::MAX] {
            let err = ThunkBlock::new(count).err().expect("size overflow");
            assert_eq!(err.to_string(), "import stub block size overflow");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn encoded_stub_preserves_both_addresses_and_returns_to_its_caller() {
        let mut block = ThunkBlock::new(1).unwrap();
        let at = block.emit(&SITE).unwrap();
        assert_eq!(memory_info(block.base).Protect, PAGE_READWRITE);
        // SAFETY: emit initialized all 32 bytes of this committed slot.
        let bytes = unsafe { std::slice::from_raw_parts(at as *const u8, THUNK_SIZE) };
        assert_eq!(&bytes[..2], &[0x48, 0xB9]);
        assert_eq!(
            u64::from_le_bytes(bytes[2..10].try_into().unwrap()),
            &SITE as *const _ as u64
        );
        assert_eq!(&bytes[10..12], &[0x48, 0xB8]);
        assert_eq!(
            u64::from_le_bytes(bytes[12..20].try_into().unwrap()),
            unimplemented_import as *const () as u64
        );
        assert_eq!(&bytes[20..22], &[0xFF, 0xE0]);
        assert_eq!(&bytes[22..], &[0xCC; 10]);

        block.seal().unwrap();
        assert_eq!(memory_info(at as *const u8).Protect, PAGE_EXECUTE_READ);
        // SAFETY: the sealed x64 stub supplies its own RCX argument and tail
        // jumps to the system-ABI handler, which returns a u64.
        let result = unsafe {
            let stub: unsafe extern "system" fn() -> u64 = std::mem::transmute(at);
            stub()
        };
        assert_eq!(result, 0);
        assert_eq!(SITE.calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn unpublished_blocks_release_the_reservation() {
        // Isolate address-reuse checks from allocations by parallel tests.
        const CHILD: &str = "VICTORY_THUNK_RELEASE_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "thunks::tests::unpublished_blocks_release_the_reservation",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        let block = ThunkBlock::new(1).unwrap();
        let base = block.base;
        drop(block);
        assert_eq!(memory_info(base).State, MEM_FREE);

        let block = ThunkBlock::new(0).unwrap();
        let base = block.base;
        block.seal().unwrap();
        assert_eq!(memory_info(base).State, MEM_FREE);

        let mut block = ThunkBlock::new(1).unwrap();
        block.emit(&SITE).unwrap();
        let base = block.base;
        // SAFETY: decommit our own page but keep its reservation. Protecting
        // it must now fail, and Drop can still release the reservation.
        let decommitted = unsafe { VirtualFree(base.cast(), block.len, MEM_DECOMMIT) };
        assert_ne!(decommitted, 0);
        let err = block.seal().unwrap_err();
        assert_eq!(memory_info(base).State, MEM_FREE);
        assert!(err.to_string().contains("VirtualProtect"));

        let mut block = ThunkBlock::new(1).unwrap();
        block.emit(&SITE).unwrap();
        let base = block.base;
        let err = block
            .seal_with_cache_flush(|_, _| {
                // SAFETY: sets only this thread's last-error code.
                unsafe { windows_sys::Win32::Foundation::SetLastError(5) };
                0
            })
            .unwrap_err();
        assert_eq!(memory_info(base).State, MEM_FREE);
        assert!(err.to_string().contains("FlushInstructionCache"));
    }

    /// In a child process of the next test: calls an unimplemented import
    /// that traps, with the crash log where `VICTORY_TRAP_LOG` says.
    #[test]
    fn trap_child() {
        let Some(log) = std::env::var_os("VICTORY_TRAP_LOG") else {
            return;
        };
        crate::log_crashes_to(log.into());
        let site = Box::leak(Box::new(ImportSite {
            dll: "probe.dll".into(),
            symbol: "Missing".into(),
            policy: UnimplementedPolicy::Trap,
            calls: AtomicU64::new(0),
        }));
        unimplemented_import(site);
    }

    #[test]
    fn a_trapped_import_is_in_the_crash_log() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("crashes.log");
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "thunks::tests::trap_child", "--nocapture"])
            .env("VICTORY_TRAP_LOG", &log)
            .output()
            .unwrap();
        assert!(!out.status.success(), "the trap ends the process");
        let text = std::fs::read_to_string(&log).unwrap_or_default();
        let named = text.contains("VIctory: unimplemented import probe.dll!Missing called");
        assert!(named, "crash log: {text:?}");
    }
}
