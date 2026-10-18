//! Registered executors share the CPU host's completion order: the task and its scratch are
//! dropped under the publication lock before the lease shrinks to the PNG and is published.
use super::*;
use std::sync::OnceLock;

const SCRATCH: u64 = 1 << 20;
const PNG: usize = 4096;

#[derive(Default)]
struct Probe {
    budget: OnceLock<Arc<Budget>>,
    /// Budget in use when the task was dropped.
    dropped: Arc<Mutex<Option<u64>>>,
    cancel_on_final_step: bool,
}

struct ProbeWork {
    budget: Arc<Budget>,
    dropped: Arc<Mutex<Option<u64>>>,
    cancel: CancelToken,
    cancel_on_final_step: bool,
    steps: u32,
}

impl ImageWork for ProbeWork {
    fn step(&mut self, _units: usize) -> Result<TaskPoll> {
        self.steps += 1;
        if self.steps < 3 {
            return Ok(TaskPoll::Pending(Progress { phase: Phase::Running, done: u64::from(self.steps), total: 3 }));
        }
        if self.cancel_on_final_step {
            self.cancel.cancel();
        }
        let mut png = Vec::new();
        png.try_reserve_exact(PNG).map_err(|_| JobError::new(ErrorCode::AllocationFailed, "test PNG"))?;
        png.resize(PNG, 0);
        Ok(TaskPoll::Complete(png))
    }
}

impl Drop for ProbeWork {
    fn drop(&mut self) {
        if let Ok(mut dropped) = self.dropped.lock() {
            *dropped = Some(self.budget.used.load(Ordering::Acquire));
        }
    }
}

impl ImageExecutor for Probe {
    fn descriptor(&self) -> Result<ExecutorDescriptor> {
        let mut descriptor = CpuHost::cooperative().descriptor()?;
        descriptor.id = ExecutorId::new("test-probe").map_err(|e| JobError::new(ErrorCode::Internal, e.to_string()))?;
        descriptor.profiles[0].features.push(crate::RenderFeature::Progressive);
        Ok(descriptor)
    }
    fn admit(&self, _: &ProductScene, _: &ImageRequest, options: JobOptions) -> Result<Admission> {
        options.validate()?;
        Ok(Admission {
            triangles: 0,
            face_appearances: 0,
            geometry_bytes: 0,
            pixel_scratch_bytes: SCRATCH,
            preparation_bytes: 0,
            estimated_total_bytes: SCRATCH,
            encoded_limit: PNG as u64,
        })
    }
    fn begin(&self, _: ProductScene, _: ImageRequest, _: JobOptions, cancel: CancelToken) -> Result<Box<dyn ImageWork>> {
        let budget = self.budget.get().ok_or_else(|| JobError::new(ErrorCode::Internal, "probe has no budget"))?.clone();
        Ok(Box::new(ProbeWork { budget, dropped: self.dropped.clone(), cancel, cancel_on_final_step: self.cancel_on_final_step, steps: 0 }))
    }
}

fn request() -> ImageRequest {
    ImageRequest {
        integrator: Integrator::Raster,
        raytrace: None,
        camera: Default::default(),
        width: 8,
        height: 8,
        background: [0; 3],
        transparent_background: false,
    }
}

fn host(cancel_on_final_step: bool) -> (ImageHost, Arc<Mutex<Option<u64>>>) {
    let probe = Arc::new(Probe { cancel_on_final_step, ..Default::default() });
    let dropped = probe.dropped.clone();
    let host = ImageHost::cooperative_with(probe.clone());
    assert!(probe.budget.set(host.0.shared.budget.clone()).is_ok());
    (host, dropped)
}

fn submit(host: &ImageHost) -> JobHandle {
    let scene = ProductScene { meshes: Vec::new(), bounds: solidcraft_geom::Aabb3::EMPTY };
    let options = JobOptions { samples: 3, ..Default::default() };
    host.submit_with_options(JobId { session: 1640, serial: 1, generation: 1 }, 1, scene, request(), options).unwrap()
}

#[test]
fn registered_task_drops_with_its_scratch_lease_before_png_publication() {
    let (host, dropped) = host(false);
    let job = submit(&host);
    assert_eq!(host.admitted_bytes(), SCRATCH);
    assert!(host.pump(1).unwrap());
    assert!(host.pump(1).unwrap());
    assert_eq!(*dropped.lock().unwrap(), None, "pending work keeps its task");
    assert_eq!(job.status().unwrap().progress.phase, Phase::Running);
    assert!(host.pump(1).unwrap());
    // During Drop the task's original scratch charge was still held; the shrink came after.
    assert_eq!(*dropped.lock().unwrap(), Some(SCRATCH));
    assert_eq!(job.status().unwrap().progress.phase, Phase::Encoding);
    assert!(job.frame().unwrap().is_none(), "the completed PNG is not published before hashing");
    assert_eq!(host.admitted_bytes(), SCRATCH, "bounded hashing retains the full lease after integrator scratch drops");
    assert!(!host.pump(1).unwrap());
    assert!(host.0.running.lock().unwrap().is_none());
    let status = job.status().unwrap();
    assert_eq!(status.progress.phase, Phase::Completed);
    assert_eq!(status.options.map(|o| o.samples), Some(3));
    assert_eq!(status.resolved.executor.as_str(), "test-probe");
    let frame = job.frame().unwrap().unwrap();
    // Only the PNG lease remains; this request retains no metadata bytes.
    assert_eq!(host.admitted_bytes(), PNG as u64 + 1024);
    assert_eq!(frame.png().len(), PNG);
    assert_eq!(frame.info.png_bytes, PNG as u64);
    assert_eq!(frame.info.png_sha256, format!("{:x}", Sha256::digest(vec![0; PNG])));
    drop(frame);
    assert_eq!(host.admitted_bytes(), PNG as u64 + 1024, "the job record still owns its frame");
    drop(job);
    assert_eq!(host.admitted_bytes(), 0);
}

#[test]
fn registered_cancellation_on_the_final_step_publishes_no_frame() {
    let (host, dropped) = host(true);
    let job = submit(&host);
    assert!(host.pump(1).unwrap());
    assert!(host.pump(1).unwrap());
    assert!(!host.pump(1).unwrap());
    let status = job.status().unwrap();
    assert_eq!(status.progress.phase, Phase::Cancelled);
    assert!(status.frame.is_none());
    assert!(job.frame().unwrap().is_none());
    assert_eq!(*dropped.lock().unwrap(), Some(SCRATCH));
    assert_eq!(host.admitted_bytes(), 0);
}

#[test]
fn registered_hosts_refuse_cpu_raytrace_and_keep_their_own_pool_and_cap() {
    let (host, _) = host(false);
    let cpu = CpuHost::cooperative();
    assert_eq!(host.pool_limit_bytes(), cpu.pool_limit_bytes());
    assert_eq!(cpu.job_limit_bytes(), super::super::image::MAX_JOB_BYTES);
    assert_eq!(host.job_limit_bytes(), MEMORY_LIMIT);
    assert!(!Arc::ptr_eq(&host.0.shared.budget, &cpu.0.shared.budget));
    assert!(host.validate_integrator(Integrator::Raytrace).is_err());
    assert!(host.descriptor_for(Integrator::Raytrace).is_err());
    let mut raytrace = request();
    raytrace.integrator = Integrator::Raytrace;
    let scene = ProductScene { meshes: Vec::new(), bounds: solidcraft_geom::Aabb3::EMPTY };
    assert!(host.submit(JobId { session: 1640, serial: 2, generation: 2 }, 1, scene.clone(), raytrace).is_err());
    let mut options = request();
    options.raytrace = Some(Default::default());
    assert!(host.submit(JobId { session: 1640, serial: 3, generation: 3 }, 1, scene, options).is_err());
    assert_eq!(host.admitted_bytes(), 0);
    assert!(!host.pump(1).unwrap());
}
