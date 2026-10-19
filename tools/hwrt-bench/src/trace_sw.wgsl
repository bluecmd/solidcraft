// The renderer's traversal as its paths and shadows call it outside media.
fn bench_trace(o: vec3<f32>, lo: vec3<f32>, d: vec3<f32>, shadow: bool, excluded: u32, tmin: f32, tmax: f32) -> Hit {
    if shadow { return trace_precise_within(o, lo, d, false, excluded, tmin, false, tmax, tmax); }
    return trace_precise_within(o, lo, d, false, excluded, tmin, false, tmax, -1.0);
}
