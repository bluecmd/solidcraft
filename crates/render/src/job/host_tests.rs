use super::*;
use solidcraft_geom::{Aabb3, Mesh};

fn request() -> ImageRequest {
    ImageRequest {
        integrator: Default::default(),
        raytrace: None,
        camera: Default::default(),
        width: 16,
        height: 16,
        background: [40, 80, 120],
        transparent_background: false,
    }
}

#[test]
fn legacy_image_requests_default_to_raster_and_reject_unknown_integrators() {
    let mut value = serde_json::to_value(request()).unwrap();
    value.as_object_mut().unwrap().remove("integrator");
    let decoded: ImageRequest = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(decoded.integrator, Integrator::Raster);
    value["integrator"] = serde_json::json!("gpu_pathtrace");
    assert!(serde_json::from_value::<ImageRequest>(value).is_err());
}

#[cfg(feature = "raytrace-cpu")]
#[test]
fn cooperative_raytrace_descriptor_matches_native_only_validation() {
    let host = CpuHost::cooperative();
    assert!(host.validate_integrator(Integrator::Raytrace).is_err());
    assert!(matches!(host.descriptor_for(Integrator::Raytrace).unwrap().availability, Availability::Unavailable { .. }));
    assert!(host.validate_integrator(Integrator::Raster).is_ok());
}

#[cfg(not(feature = "raytrace-cpu"))]
#[test]
fn compiled_out_raytracing_never_enters_the_host_queue() {
    let host = CpuHost::cooperative();
    let mut request = request();
    request.integrator = Integrator::Raytrace;
    assert!(host.validate_integrator(Integrator::Raytrace).unwrap_err().message.contains("not compiled in"));
    assert!(host.descriptor_for(Integrator::Raytrace).is_err());
    let error = host.submit(JobId { session: 1, serial: 1, generation: 1 }, 0, empty(), request).err().unwrap();
    assert_eq!(error.code, ErrorCode::HostUnavailable);
    assert_eq!(host.admitted_bytes(), 0);
    assert!(!host.pump(1).unwrap());
}
fn empty() -> ProductScene {
    ProductScene { meshes: Vec::new(), bounds: Aabb3::EMPTY }
}
fn id(owner: u64, serial: u64) -> JobId {
    JobId { session: owner, serial, generation: serial }
}
fn drain(host: &CpuHost) {
    for _ in 0..10_000 {
        if !host.pump(4096).unwrap() {
            return;
        }
    }
    panic!("small cooperative jobs did not drain");
}

#[test]
fn queue_is_bounded_and_cancelled_entries_release_their_reservations() {
    let host = CpuHost::cooperative();
    let a = host.submit(id(10, 1), 100, empty(), request()).unwrap();
    let b = host.submit(id(20, 1), 200, empty(), request()).unwrap();
    let before = host.admitted_bytes();
    assert_eq!(host.submit(id(30, 1), 300, empty(), request()).err().unwrap().code, ErrorCode::QueueFull);
    assert_eq!(host.admitted_bytes(), before);
    assert_eq!(a.cancel().unwrap().progress.phase, Phase::Cancelled);
    let c = host.submit(id(30, 1), 300, empty(), request()).unwrap();
    drain(&host);
    assert_eq!(a.status().unwrap().progress.phase, Phase::Cancelled);
    for (handle, owner, generation) in [(&b, 20, 200), (&c, 30, 300)] {
        let frame = handle.frame().unwrap().unwrap();
        assert_eq!(frame.info.job, id(owner, 1));
        assert_eq!(frame.info.scene_generation, generation);
        assert_eq!(frame.info.executor.as_str(), "cpu-product");
        assert_eq!(image::load_from_memory(frame.png()).unwrap().width(), 16);
        assert_eq!(frame.info.png_bytes, frame.png().len() as u64);
        assert_eq!(frame.info.png_sha256, format!("{:x}", Sha256::digest(frame.png())));
        assert_eq!(handle.status().unwrap().frame.unwrap().png_sha256, frame.info.png_sha256);
    }
    assert!(host.admitted_bytes() > 0 && host.admitted_bytes() < before);
    let frame = b.frame().unwrap().unwrap();
    drop(b);
    drop(c);
    assert!(host.admitted_bytes() > 0, "an externally retained frame retains its lease");
    drop(frame);
    assert_eq!(host.admitted_bytes(), 0);
}

#[test]
fn cancellation_while_encoding_cannot_publish_a_late_frame() {
    let host = CpuHost::cooperative();
    let job = host.submit(id(1, 1), 8, empty(), request()).unwrap();
    for _ in 0..10_000 {
        host.pump(64).unwrap();
        let status = job.status().unwrap();
        if status.progress.phase == Phase::Encoding && status.progress.done > 0 {
            break;
        }
    }
    assert_eq!(job.status().unwrap().progress.phase, Phase::Encoding);
    assert_eq!(job.cancel().unwrap().progress.phase, Phase::Cancelling);
    drain(&host);
    assert_eq!(job.status().unwrap().progress.phase, Phase::Cancelled);
    assert!(job.frame().unwrap().is_none());
    assert_eq!(host.admitted_bytes(), 0);
}

#[test]
fn malformed_preparation_fails_the_job_without_stopping_the_host() {
    let host = CpuHost::cooperative();
    let scene = ProductScene {
        bounds: Aabb3::EMPTY,
        meshes: vec![crate::ProductMesh {
            name: "Broken".into(),
            mesh: Arc::new(Mesh { triangles: vec![[99, 99, 99]], ..Default::default() }),
            color: [0; 3],
            opacity: 1.0,
            surface: None,
            faces: Vec::new(),
        }],
    };
    let broken = host.submit(id(1, 1), 1, scene, request()).unwrap();
    drain(&host);
    let status = broken.status().unwrap();
    assert_eq!(status.progress.phase, Phase::Failed);
    assert_eq!(status.error.unwrap().code, ErrorCode::InvalidScene);
    assert!(status.frame.is_none(), "a failed preparation path has no cached PNG identity");
    assert!(broken.frame().unwrap().is_none());
    assert!(host.is_available());
    let healthy = host.submit(id(1, 2), 1, empty(), request()).unwrap();
    drain(&host);
    assert_eq!(healthy.status().unwrap().progress.phase, Phase::Completed);
}

#[test]
fn closing_the_host_cancels_active_and_queued_work() {
    let host = CpuHost::cooperative();
    let budget = host.0.shared.budget.clone();
    let active = host.submit(id(1, 1), 1, empty(), request()).unwrap();
    let queued = host.submit(id(2, 1), 1, empty(), request()).unwrap();
    host.pump(1).unwrap();
    drop(host);
    assert_eq!(active.status().unwrap().progress.phase, Phase::Cancelled);
    assert_eq!(queued.status().unwrap().progress.phase, Phase::Cancelled);
    assert_eq!(budget.used.load(Ordering::Acquire), 0);
}

#[test]
fn a_foreign_frame_is_rejected_even_if_a_broken_host_puts_it_in_the_result_slot() {
    let host = CpuHost::cooperative();
    let job = host.submit(id(1, 1), 1, empty(), request()).unwrap();
    drain(&host);
    {
        let mut data = job.0.data.lock().unwrap();
        let frame = Arc::get_mut(data.frame.as_mut().unwrap()).unwrap();
        frame.info.job.session = 2;
    }
    assert_eq!(job.frame().err().unwrap().code, ErrorCode::WrongOwner);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_job_finishes_without_any_poll_or_ui_pump() {
    let host = CpuHost::for_platform().unwrap();
    assert_eq!(host.kind(), HostKind::NativeWorker);
    let job = host.submit(id(1, 1), 12, empty(), request()).unwrap();
    // A generous deadlock watchdog, not a performance assertion.
    let status = job.wait_timeout(std::time::Duration::from_secs(60)).unwrap();
    assert_eq!(status.progress.phase, Phase::Completed);
    assert_eq!(job.frame().unwrap().unwrap().info.scene_generation, 12);
    let frame = job.frame().unwrap().unwrap();
    assert_eq!(frame.info.png_bytes, frame.png().len() as u64);
    assert_eq!(frame.info.png_sha256, format!("{:x}", Sha256::digest(frame.png())));
    assert_eq!(status.frame.unwrap().png_sha256, frame.info.png_sha256);
}

/// Isolate the existing integrator-to-host completion boundary with a real PNG
/// from the production encoder. This does not pretend to render the noisy pixels.
fn pending_digest_fixture() -> (CpuHost, JobHandle, Running, usize) {
    let host = CpuHost::cooperative();
    let mut request = request();
    request.width = 256;
    request.height = 128;
    let mut rgba = vec![255; request.width as usize * request.height as usize * 4];
    let mut seed = 0x6a09_e667u32;
    for pixel in rgba.as_chunks_mut::<4>().0 {
        for channel in pixel.iter_mut().take(3) {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *channel = seed as u8;
        }
    }
    let mut encoder = super::super::encoding::PngEncoding::new(
        request.width,
        request.height,
        u64::from(request.width) * u64::from(request.height) * 8 + 65_536,
        CancelToken::default(),
    )
    .unwrap();
    let png = loop {
        if let Some(png) = encoder.step(&rgba).unwrap() {
            break png;
        }
    };
    assert!(png.len() > PNG_HASH_CHUNK_BYTES * PNG_HASH_CHUNKS_PER_ADVANCE);
    let length = png.len();
    let job = host.submit(id(61, 1), 99, empty(), request).unwrap();
    let work = host.0.shared.queue.lock().unwrap().pop_front().unwrap();
    let mut running = Running::begin(work).unwrap();
    // Exactly the same state transition used when WorkTask returns Complete.
    running.task = None;
    running.pending_png = Some(PendingPng::new(png));
    (host, job, running, length)
}

#[test]
fn png_digest_is_incremental_and_keeps_the_original_allocation_and_lease_until_completion() {
    let (host, job, mut running, length) = pending_digest_fixture();
    let before = host.admitted_bytes();
    let pointer = running.pending_png.as_ref().unwrap().png.as_ptr();
    assert!(running.advance(0).unwrap());
    assert_eq!(running.pending_png.as_ref().unwrap().hashed, 0, "zero units cannot hash a chunk");
    assert_eq!(host.admitted_bytes(), before);
    assert!(job.frame().unwrap().is_none());
    assert!(running.advance(usize::MAX).unwrap());
    assert_eq!(running.pending_png.as_ref().unwrap().hashed, PNG_HASH_CHUNK_BYTES * PNG_HASH_CHUNKS_PER_ADVANCE);
    assert_eq!(host.admitted_bytes(), before, "hashing must retain the full original work lease");
    assert!(job.frame().unwrap().is_none());
    let mut previous = job.status().unwrap().progress.done;
    while running.advance(1).unwrap() {
        let progress = job.status().unwrap().progress;
        assert_eq!(progress.phase, Phase::Encoding);
        assert_eq!(progress.total, length as u64);
        assert_eq!(progress.done - previous, PNG_HASH_CHUNK_BYTES as u64);
        assert_eq!(host.admitted_bytes(), before);
        assert!(job.frame().unwrap().is_none());
        previous = progress.done;
    }
    assert!(running.task.is_none() && running.pending_png.is_none() && running.reservation.is_none());
    let status = job.status().unwrap();
    assert_eq!(status.progress.phase, Phase::Completed);
    let frame = job.frame().unwrap().unwrap();
    assert_eq!(frame.png().as_ptr(), pointer, "the original Vec moves into the retained frame without cloning");
    assert_eq!(frame.info.png_bytes, length as u64);
    assert_eq!(frame.info.png_sha256.len(), 64);
    assert!(frame.info.png_sha256.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)));
    assert_eq!(frame.info.png_sha256, format!("{:x}", Sha256::digest(frame.png())));
    assert_eq!(status.frame.unwrap().png_sha256, frame.info.png_sha256);
    let decoded = image::load_from_memory(frame.png()).unwrap().to_rgba8();
    assert_eq!(decoded.dimensions(), (256, 128));
    assert!(decoded.pixels().all(|p| p.0[3] == 255));
    assert!(decoded.pixels().any(|p| p.0[..3] != decoded.get_pixel(0, 0).0[..3]));
    let actual_metadata =
        std::mem::size_of::<Running>() + std::mem::size_of::<RenderFrame>() + frame.info.png_sha256.capacity() + 2 * crate::MAX_RENDER_ID_BYTES + 32;
    assert!(actual_metadata <= FRAME_METADATA_BYTES as usize, "digest Vec/offset/state and actual hex allocation must be admitted");
    assert_eq!(host.admitted_bytes(), frame.png.capacity() as u64 + FRAME_METADATA_BYTES);
    for _ in 0..3 {
        let current = job.status().unwrap();
        assert_eq!(current.frame.unwrap().png_sha256, frame.info.png_sha256);
        assert_eq!(frame.png().as_ptr(), pointer);
    }
    drop(running);
    drop(job);
    assert!(host.admitted_bytes() > 0);
    drop(frame);
    assert_eq!(host.admitted_bytes(), 0);
}

#[test]
fn cancellation_during_png_hashing_and_host_disposal_cannot_publish_a_frame() {
    for close_host in [false, true] {
        let (host, job, running, _) = pending_digest_fixture();
        let budget = host.0.shared.budget.clone();
        *host.0.running.lock().unwrap() = Some(running);
        assert!(host.pump(1).unwrap());
        assert!(host.0.running.lock().unwrap().as_ref().unwrap().pending_png.as_ref().unwrap().hashed > 0);
        assert_eq!(job.status().unwrap().progress.phase, Phase::Encoding);
        assert!(job.frame().unwrap().is_none());
        if close_host {
            drop(host);
        } else {
            assert_eq!(job.cancel().unwrap().progress.phase, Phase::Cancelling);
            assert!(!host.pump(1).unwrap());
            drop(host);
        }
        assert_eq!(job.status().unwrap().progress.phase, Phase::Cancelled);
        assert!(job.frame().unwrap().is_none());
        assert_eq!(budget.used.load(Ordering::Acquire), 0);
    }
}

#[test]
fn a_cancel_after_the_last_digest_chunk_still_wins_the_publication_lock() {
    let (host, job, mut running, _) = pending_digest_fixture();
    let pending = running.pending_png.as_mut().unwrap();
    while !pending.step(1, &job.0.cancel).unwrap() {}
    assert_eq!(pending.hashed, pending.png.len());
    job.cancel().unwrap();
    assert!(!running.guarded_step(1));
    assert_eq!(job.status().unwrap().progress.phase, Phase::Cancelled);
    assert!(job.frame().unwrap().is_none());
    drop(running);
    assert_eq!(host.admitted_bytes(), 0);
}

#[test]
fn png_digest_metadata_fits_the_existing_allowance_without_changing_admission() {
    // Fixed digest/offset state and the completed frame (including its bounded
    // hex string and backend/executor IDs) fit the existing 1024-byte allowance.
    assert!(std::mem::size_of::<PendingPng>() <= 256);
    assert!(
        std::mem::size_of::<Running>() + std::mem::size_of::<RenderFrame>() + 64 + 2 * crate::MAX_RENDER_ID_BYTES + 32
            <= FRAME_METADATA_BYTES as usize
    );
    let host = CpuHost::cooperative();
    let expected = admit(&empty(), &request()).unwrap().estimated_total_bytes;
    let job = host.submit(id(70, 1), 1, empty(), request()).unwrap();
    assert_eq!(host.admitted_bytes(), expected);
    assert!(job.status().unwrap().frame.is_none());
    job.cancel().unwrap();
    drain(&host);
    assert_eq!(host.admitted_bytes(), 0);
}

/// An executor whose task reports, as it is dropped, how much of the pool is admitted, and
/// completes after `steps` steps (or never, cancelled on its last one with `cancel_last`).
struct DropProbe {
    admitted_at_drop: Arc<AtomicU64>,
    budget: Arc<std::sync::OnceLock<Arc<Budget>>>,
    estimated: u64,
}
struct ProbeTask {
    admitted_at_drop: Arc<AtomicU64>,
    budget: Arc<std::sync::OnceLock<Arc<Budget>>>,
    steps: u32,
}
impl Drop for ProbeTask {
    fn drop(&mut self) {
        let used = self.budget.get().map_or(u64::MAX, |b| b.used.load(Ordering::Acquire));
        self.admitted_at_drop.store(used, Ordering::Release);
    }
}
impl ImageWork for ProbeTask {
    fn step(&mut self, _units: usize) -> Result<TaskPoll> {
        self.steps += 1;
        if self.steps < 3 {
            return Ok(TaskPoll::Pending(Progress { phase: Phase::Running, done: u64::from(self.steps), total: 3 }));
        }
        Ok(TaskPoll::Complete(vec![0u8; 100]))
    }
}
impl ImageExecutor for DropProbe {
    fn descriptor(&self) -> Result<ExecutorDescriptor> {
        CpuHost::cooperative().descriptor()
    }
    fn admit(&self, _: &ProductScene, _: &ImageRequest, _: JobOptions) -> Result<Admission> {
        Ok(Admission {
            triangles: 0,
            face_appearances: 0,
            geometry_bytes: 0,
            pixel_scratch_bytes: self.estimated,
            preparation_bytes: 0,
            estimated_total_bytes: self.estimated,
            encoded_limit: 4096,
        })
    }
    fn begin(&self, _: ProductScene, _: ImageRequest, _: JobOptions, _: CancelToken) -> Result<Box<dyn ImageWork>> {
        Ok(Box::new(ProbeTask { admitted_at_drop: self.admitted_at_drop.clone(), budget: self.budget.clone(), steps: 0 }))
    }
}

fn probe_host(estimated: u64) -> (CpuHost, Arc<AtomicU64>) {
    let admitted_at_drop = Arc::new(AtomicU64::new(0));
    let budget = Arc::new(std::sync::OnceLock::new());
    let host = CpuHost::cooperative_with(Arc::new(DropProbe { admitted_at_drop: admitted_at_drop.clone(), budget: budget.clone(), estimated }));
    let _ = budget.set(host.0.shared.budget.clone());
    (host, admitted_at_drop)
}

#[test]
fn a_completed_task_is_dropped_while_its_full_reservation_still_pays_for_it() {
    let estimated = 1 << 20;
    let (host, admitted_at_drop) = probe_host(estimated);
    let job = host.submit(id(1, 1), 1, empty(), request()).unwrap();
    drain(&host);
    assert_eq!(job.status().unwrap().progress.phase, Phase::Completed);
    // While the task (and its scratch) was being dropped, its whole charge was still held.
    assert_eq!(admitted_at_drop.load(Ordering::Acquire), estimated);
    // Afterwards only the PNG's lease remains, and the frame keeps it.
    let frame = job.frame().unwrap().unwrap();
    assert!(host.admitted_bytes() > 0 && host.admitted_bytes() < estimated, "{}", host.admitted_bytes());
    drop(job);
    drop(frame);
    assert_eq!(host.admitted_bytes(), 0);
}

#[test]
fn a_cancel_on_the_final_step_publishes_no_frame_and_drops_the_task() {
    let estimated = 1 << 20;
    let (host, admitted_at_drop) = probe_host(estimated);
    let job = host.submit(id(1, 1), 1, empty(), request()).unwrap();
    // Two pending steps, then cancel before the step that would complete.
    host.pump(1).unwrap();
    host.pump(1).unwrap();
    job.cancel().unwrap();
    drain(&host);
    assert_eq!(job.status().unwrap().progress.phase, Phase::Cancelled);
    assert!(job.frame().unwrap().is_none());
    assert_eq!(admitted_at_drop.load(Ordering::Acquire), estimated, "the task went before its charge");
    assert_eq!(host.admitted_bytes(), 0);
}
