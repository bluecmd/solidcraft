//! Bounded PNG encoding shared by native image executors.
use super::{CancelToken, ErrorCode, ImageRequest, ImageWork, JobError, Phase, Progress, Result, TaskPoll, encoding::PngEncoding};

/// Owns admitted RGBA bytes and yields after at most four scanlines.
pub struct PngTask {
    rgba: Vec<u8>,
    height: usize,
    encoder: PngEncoding,
}
impl PngTask {
    pub fn new(request: &ImageRequest, rgba: Vec<u8>, cancel: CancelToken) -> Result<Self> {
        let pixels = u64::from(request.width) * u64::from(request.height);
        if !(1..=4096).contains(&request.width) || !(1..=4096).contains(&request.height) || pixels > 4_194_304 || rgba.len() as u64 != pixels * 4 {
            return Err(JobError::new(ErrorCode::InvalidRequest, "invalid PNG dimensions or RGBA length"));
        }
        let encoder = PngEncoding::new(request.width, request.height, pixels * 8 + 65_536, cancel)?;
        Ok(Self { rgba, height: request.height as usize, encoder })
    }
}
impl ImageWork for PngTask {
    fn step(&mut self, _units: usize) -> Result<TaskPoll> {
        for _ in 0..4 {
            if self.encoder.row() == self.height {
                break;
            }
            self.encoder.step(&self.rgba)?;
        }
        if self.encoder.row() != self.height {
            return Ok(TaskPoll::Pending(Progress { phase: Phase::Encoding, done: self.encoder.row() as u64, total: self.height as u64 }));
        }
        match self.encoder.step(&self.rgba)? {
            Some(png) => {
                self.rgba = Vec::new();
                Ok(TaskPoll::Complete(png))
            }
            None => Err(JobError::new(ErrorCode::Internal, "PNG encoder did not finish after its last row")),
        }
    }
}
