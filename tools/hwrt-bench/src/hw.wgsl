enable wgpu_ray_query;
struct BenchRay { o: vec4<f32>, d: vec4<f32>, x: vec4<u32>, lo: vec4<f32> };
@group(0) @binding(0) var scene: acceleration_structure;
@group(0) @binding(1) var<storage, read_write> rays: array<BenchRay>;
@group(0) @binding(2) var<storage, read_write> hits: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> bench: vec4<u32>;

@compute @workgroup_size(64)
fn hw_trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if i >= bench.x { return; }
    let r = rays[i]; if r.d.w < 0.0 { return; }
    var flags = 0u;
    if r.x.y != 0u { flags = RAY_FLAG_TERMINATE_ON_FIRST_HIT; }
    var q: ray_query;
    rayQueryInitialize(&q, scene, RayDesc(flags, 0xffu, r.o.w, r.d.w, r.o.xyz + r.lo.xyz, r.d.xyz));
    while rayQueryProceed(&q) {}
    let c = rayQueryGetCommittedIntersection(&q);
    if c.kind == RAY_QUERY_INTERSECTION_NONE { hits[i] = vec4<f32>(r.d.w, bitcast<f32>(0xffffffffu), 0.0, 0.0); }
    else { hits[i] = vec4<f32>(c.t, bitcast<f32>(c.primitive_index), c.barycentrics.x, c.barycentrics.y); }
}
