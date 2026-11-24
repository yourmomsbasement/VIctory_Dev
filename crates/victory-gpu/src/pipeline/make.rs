//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and Grand
//! Theft Auto VI's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline and waits for it within a budget per
//! frame, so the cheap ones (menus, one-off passes such as a colour-grading
//! bake) are there at once as before; past the budget a draw or dispatch
//! whose pipeline is not ready is dropped, as an untranslated one is.
//!
//! shortcut: the budget (half a second of waiting per title frame,
//! `VICTORY_PIPELINE_WAIT_MS`) is a guess; a pass the title runs only once and whose pipeline takes longer
//! is lost.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queued: AtomicBool,
    made: OnceLock<Option<HostPipeline>>,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw has used it yet.
    key: String,
    used: AtomicBool,
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let mut key = blake3::Hasher::new();
        for s in &shaders {
            key.update(s.stage.as_bytes());
            key.update(s.hash.as_bytes());
        }
        key.update(
            graphics
                .as_ref()
                .map(|g| g.describe())
                .unwrap_or_default()
                .as_bytes(),
        );
        Pending {
            shaders,
            graphics,
            queued: AtomicBool::new(false),
            made: OnceLock::new(),
            key: key.finalize().to_hex()[..32].to_owned(),
            used: AtomicBool::new(false),
        }
    }

    fn make(&self) {
        self.made
            .get_or_init(|| HostPipeline::new(&self.shaders, self.graphics.as_deref()));
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }
}

/// Why a pipeline has no host pipeline (diagnostics): its shaders that do
/// not translate, else "refused" or "being made"; `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    match p.made.get() {
        Some(Some(_)) => None,
        None => Some("being made".into()),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Waiting allowed per title frame, over all threads: 500 ms unless
/// `VICTORY_PIPELINE_WAIT_MS` says otherwise (0: never wait, a draw whose
/// pipeline is not made yet is dropped until it is; one-off passes such as
/// a colour-grading bake may then be lost).
fn budget() -> Duration {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    Duration::from_millis(*MS.get_or_init(|| {
        std::env::var("VICTORY_PIPELINE_WAIT_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(500)
    }))
}

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    if p.made.get().is_none() {
        start(p);
        wait(p, crate::frame::current());
    }
    let made = p.made.get()?.as_ref();
    if made.is_some() && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
    }
    made
}

/// Queues `p` for the workers, once.
fn start(p: &Arc<Pending>) {
    if !p.queued.swap(true, Ordering::Relaxed) {
        PEAK.fetch_max(
            WAITING.fetch_add(1, Ordering::Relaxed) + 1,
            Ordering::Relaxed,
        );
        queue(p.clone());
    }
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    if !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p);
    }
}

/// Waits for `p` while frame `frame` has budget left.
fn wait(p: &Pending, frame: u64) {
    static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
    static WAITED_US: AtomicU64 = AtomicU64::new(0);
    if FRAME.swap(frame, Ordering::Relaxed) != frame {
        WAITED_US.store(0, Ordering::Relaxed);
    }
    let left = budget().saturating_sub(Duration::from_micros(WAITED_US.load(Ordering::Relaxed)));
    let started = Instant::now();
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        held = done
            .wait_timeout(held, rest)
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
    WAITED_US.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// Stack of each pipeline worker: room for the driver's shader compiler,
/// which runs on it (reserved address space; committed only as used).
const PIPELINE_STACK: usize = 64 << 20;

/// Hands `p` to the workers (half the logical processors, 2 to 8), started
/// on first use.
fn queue(p: Arc<Pending>) {
    static SENDER: OnceLock<std::sync::mpsc::Sender<Arc<Pending>>> = OnceLock::new();
    let sender = SENDER.get_or_init(|| {
        let (sender, receiver) = std::sync::mpsc::channel::<Arc<Pending>>();
        let receiver = Arc::new(Mutex::new(receiver));
        let workers = std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).clamp(2, 8));
        for i in 0..workers {
            let receiver = receiver.clone();
            let spawned = std::thread::Builder::new()
                .name(format!("victory-pipelines-{i}"))
                // The driver compiles pipelines on this thread, recursively:
                // NVIDIA's compiler (nvgpucomp64) overflowed the default 2 MB
                // on a world compute program and refused it as out of memory.
                .stack_size(PIPELINE_STACK)
                .spawn(move || {
                    // Below the title's own threads (loading, recording).
                    // SAFETY: a plain call on our own thread.
                    unsafe {
                        use windows::Win32::System::Threading::*;
                        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
                    }
                    loop {
                        let next = receiver.lock().unwrap_or_else(|e| e.into_inner()).recv();
                        let Ok(p) = next else { return };
                        p.make();
                        if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                            drained();
                        } else {
                            super::library::save(false);
                        }
                    }
                });
            if let Err(e) = spawned {
                tracing::warn!(error = %e, "no pipeline worker");
            }
        }
        sender
    });
    if let Err(e) = sender.send(p) {
        // No worker: made here instead.
        e.0.make();
        WAITING.fetch_sub(1, Ordering::Relaxed);
    }
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tests that queue pipelines read the process-wide queue and budget
    /// counters, which the workers change from their own threads: they run
    /// one at a time, with nothing left in flight.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn serial() -> std::sync::MutexGuard<'static, ()> {
        SERIAL.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A title shader container (no code, so it never translates).
    fn shader(stage: &'static str, hash: &str) -> Arc<Shader> {
        Arc::new(Shader {
            stage,
            hash: hash.to_owned(),
            container: Vec::new(),
            translated: OnceLock::new(),
        })
    }

    /// Waits for the worker's answer to `p`.
    fn made(p: &Pending) {
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(p.made.get().is_some(), "a worker made the pipeline");
    }

    /// Waits for the workers to finish everything queued.
    fn idle() {
        let started = Instant::now();
        while waiting() > 0 && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert_eq!(waiting(), 0, "the workers finished every queued pipeline");
    }

    /// The key of a pipeline with the default graphics state, as `f` leaves
    /// it.
    fn state_key(f: impl FnOnce(&mut GraphicsState)) -> String {
        let mut g = GraphicsState::default();
        f(&mut g);
        Pending::new(Vec::new(), Some(Box::new(g))).key
    }

    /// Reads a synthetic pipeline-state subobject into `g`, as a stream
    /// hands one over: its type, then its description.
    fn sub<T>(g: &mut GraphicsState, kind: u32, value: T) {
        #[repr(C)]
        struct Sub<T> {
            kind: u32,
            value: T,
        }
        let sub = Sub { kind, value };
        // SAFETY: a whole subobject of type `kind`: its description follows
        // the type at +4, or at +8 for those holding pointers (where
        // `repr(C)` puts them), as `GraphicsState::read` expects.
        unsafe { g.read(kind, std::ptr::from_ref(&sub).cast()) };
    }

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let _serial = serial();
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        assert!(p.queued.load(Ordering::Relaxed));
        made(&p);
        assert!(matches!(p.made.get(), Some(None)));
        idle();
    }

    /// The key names the shaders, so the pipelines a run drew with are found
    /// again by the next run's early start.
    #[test]
    fn keys_identify_the_shaders_a_pipeline_is_made_from() {
        let vs = shader("vertex", "0123456789abcdef");
        let ps = shader("pixel", "fedcba9876543210");
        let key = |shaders: Vec<Arc<Shader>>| Pending::new(shaders, None).key;
        assert_eq!(
            key(vec![vs.clone(), ps.clone()]),
            key(vec![vs.clone(), ps.clone()])
        );
        assert_ne!(key(vec![vs.clone()]), key(vec![vs.clone(), ps.clone()]));
        assert_ne!(
            key(vec![vs.clone()]),
            key(vec![shader("pixel", "0123456789abcdef")])
        );
        assert_ne!(
            key(vec![vs.clone()]),
            key(vec![shader("vertex", "fedcba9876543210")])
        );
        // No state at all (a compute pipeline) is not the default state.
        assert_ne!(
            key(vec![vs.clone()]),
            Pending::new(vec![vs], Some(Box::new(GraphicsState::default()))).key
        );
        let empty = key(Vec::new());
        assert_eq!(
            empty.len(),
            32,
            "sixteen bytes, as the early list holds them"
        );
        assert!(empty.bytes().all(|b| b.is_ascii_hexdigit()), "{empty}");
    }

    /// Every part of the graphics state is in the key: a pipeline must never
    /// be started from the host pipeline of another state.
    #[test]
    fn keys_identify_every_part_of_the_graphics_state() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::core::PCSTR;

        // A subobject's type as the stream that holds it writes it.
        let kind = |t: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE| t.0 as u32;
        let base = state_key(|_| {});
        assert_eq!(base, state_key(|_| {}));
        let element = [D3D12_INPUT_ELEMENT_DESC {
            SemanticName: PCSTR(c"POSITION".as_ptr().cast()),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32B32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 0,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        }];
        let changes: [(&str, String); 10] = [
            (
                "blend",
                state_key(|g| {
                    sub(
                        g,
                        kind(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND),
                        D3D12_BLEND_DESC {
                            IndependentBlendEnable: true.into(),
                            ..Default::default()
                        },
                    )
                }),
            ),
            (
                "sample mask",
                state_key(|g| {
                    sub(
                        g,
                        kind(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK),
                        0xffu32,
                    )
                }),
            ),
            (
                "rasterizer",
                state_key(|g| {
                    sub(
                        g,
                        kind(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER),
                        D3D12_RASTERIZER_DESC {
                            CullMode: D3D12_CULL_MODE_FRONT,
                            ..Default::default()
                        },
                    )
                }),
            ),
            (
                "depth-stencil",
                state_key(|g| {
                    sub(
                        g,
                        kind(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL),
                        D3D12_DEPTH_STENCIL_DESC {
                            DepthFunc: D3D12_COMPARISON_FUNC_GREATER,
                            ..Default::default()
                        },
                    )
                }),
            ),
            (
                "input layout",
                state_key(|g| {
                    sub(
                        g,
                        kind(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT),
                        D3D12_INPUT_LAYOUT_DESC {
                            pInputElementDescs: element.as_ptr(),
                            NumElements: element.len() as u32,
                        },
                    )
                }),
            ),
            (
                "strip cut",
                state_key(|g| {
                    sub(
                        g,
                        kind(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE),
                        D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF,
                    )
                }),
            ),
            (
                "topology",
                state_key(|g| {
                    sub(
                        g,
                        kind(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY),
                        D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
                    )
                }),
            ),
            (
                "render targets",
                state_key(|g| {
                    sub(
                        g,
                        kind(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS),
                        D3D12_RT_FORMAT_ARRAY {
                            NumRenderTargets: 1,
                            RTFormats: [DXGI_FORMAT_R8G8B8A8_UNORM; 8],
                        },
                    )
                }),
            ),
            (
                "depth format",
                state_key(|g| {
                    sub(
                        g,
                        kind(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT),
                        DXGI_FORMAT_D32_FLOAT,
                    )
                }),
            ),
            (
                "samples",
                state_key(|g| {
                    sub(
                        g,
                        kind(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC),
                        DXGI_SAMPLE_DESC {
                            Count: 4,
                            Quality: 0,
                        },
                    )
                }),
            ),
        ];
        for (what, key) in changes {
            assert_ne!(key, base, "the key must change with the {what}");
        }
    }

    /// A second use of a refused pipeline is answered at once: it is neither
    /// queued nor waited for again.
    #[test]
    fn a_made_pipeline_is_not_queued_again() {
        let _serial = serial();
        idle();
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(p.made.set(None).is_ok());
        let started = Instant::now();
        assert!(ready(&p).is_none());
        assert!(ready(&p).is_none());
        assert!(!p.queued.load(Ordering::Relaxed), "it was never queued");
        assert_eq!(waiting(), 0);
        assert!(
            started.elapsed() < Duration::from_millis(250),
            "no budget spent"
        );
    }

    /// A first use queues the pipeline once, and its making leaves nothing
    /// waiting (the frame line's count comes back to zero).
    #[test]
    fn queueing_once_leaves_nothing_waiting() {
        let _serial = serial();
        idle();
        let p = Arc::new(Pending::new(Vec::new(), None));
        start(&p);
        start(&p);
        assert!(p.queued.load(Ordering::Relaxed));
        made(&p);
        idle();
    }

    /// A pipeline whose shaders do not translate is refused once, not queued
    /// for again on every draw.
    #[test]
    fn a_pipeline_whose_shaders_do_not_translate_is_refused_once() {
        let _serial = serial();
        idle();
        let p = Arc::new(Pending::new(
            vec![shader("vertex", "0123456789abcdef")],
            None,
        ));
        assert!(ready(&p).is_none());
        made(&p);
        assert!(matches!(p.made.get(), Some(None)));
        let started = Instant::now();
        assert!(ready(&p).is_none());
        assert!(
            started.elapsed() < Duration::from_millis(250),
            "no budget spent"
        );
        idle();
    }

    /// A refused pipeline is not listed for the next run's early start: only
    /// the ones a draw used are.
    #[test]
    fn a_refused_pipeline_is_not_listed_for_an_early_start() {
        let _serial = serial();
        idle();
        let p = Arc::new(Pending::new(
            vec![shader("vertex", "aaaaaaaaaaaaaaaa")],
            None,
        ));
        assert!(ready(&p).is_none());
        made(&p);
        assert!(matches!(p.made.get(), Some(None)));
        assert!(!p.used.load(Ordering::Relaxed));
        assert!(!crate::pipeline::library::wanted_early(&p.key));
        idle();
    }

    /// At creation a pipeline is started at once only if an earlier run drew
    /// with it.
    #[test]
    fn early_starts_only_pipelines_an_earlier_run_drew_with() {
        let _serial = serial();
        idle();
        let used = Arc::new(Pending::new(
            vec![shader("vertex", "1111111111111111")],
            None,
        ));
        let unused = Arc::new(Pending::new(
            vec![shader("vertex", "2222222222222222")],
            None,
        ));
        early(&unused);
        assert!(
            !unused.queued.load(Ordering::Relaxed),
            "it was never drawn with"
        );
        crate::pipeline::library::note_used(&used.key);
        early(&used);
        assert!(
            used.queued.load(Ordering::Relaxed),
            "an earlier run drew with it"
        );
        idle();
    }

    /// The waiting budget is the frame's, not the pipeline's: once spent, the
    /// frame waits for nothing more, and the next frame has a full one.
    #[test]
    fn the_wait_budget_is_spent_once_per_frame() {
        let _serial = serial();
        let p = Pending::new(Vec::new(), None);
        let frame = u64::MAX / 2; // a frame number no other test uses
        let started = Instant::now();
        wait(&p, frame);
        let first = started.elapsed();
        assert!(
            first >= budget() / 2,
            "the first wait spends the frame's budget: {first:?}"
        );
        assert!(first < budget() + Duration::from_secs(5), "{first:?}");
        let started = Instant::now();
        wait(&p, frame);
        assert!(
            started.elapsed() < Duration::from_millis(250),
            "the budget is spent"
        );
        let started = Instant::now();
        wait(&p, frame + 1);
        assert!(
            started.elapsed() >= budget() / 2,
            "a new frame has a full budget: {:?}",
            started.elapsed()
        );
    }

    /// A pipeline that is being made says so (diagnostics).
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_pipeline_being_made_says_so() {
        let p = Pending::new(Vec::new(), None);
        assert_eq!(missing(&p).as_deref(), Some("being made"));
    }

    /// A refused pipeline names the shaders that do not translate
    /// (diagnostics).
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_refused_pipeline_names_the_shaders_that_do_not_translate() {
        let vs = shader("vertex", "0123456789abcdef");
        let ps = shader("pixel", "fedcba9876543210");
        assert!(vs.translated.set(None).is_ok());
        let p = Pending::new(vec![vs, ps], None);
        assert!(p.made.set(None).is_ok());
        assert_eq!(missing(&p).as_deref(), Some("vertex 0123456789abcdef"));
    }

    /// A pipeline refused for its state, whose shaders all translate, names
    /// its shaders instead (diagnostics).
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_pipeline_refused_for_its_state_says_refused() {
        let p = Pending::new(vec![shader("vertex", "0123456789abcdef")], None);
        assert!(p.made.set(None).is_ok());
        let why = missing(&p).expect("a reason");
        assert!(why.starts_with("refused "), "{why}");
        assert!(why.contains("vertex 0123456789abcdef"), "{why}");
    }
}
