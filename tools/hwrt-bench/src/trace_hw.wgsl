@group(2) @binding(0) var scene_as: acceleration_structure;
fn bench_trace(o: vec3<f32>, lo: vec3<f32>, d: vec3<f32>, shadow: bool, excluded: u32, tmin: f32, tmax: f32) -> Hit {
    var flags = 0u;
    if shadow { flags = RAY_FLAG_TERMINATE_ON_FIRST_HIT; }
    var q: ray_query;
    rayQueryInitialize(&q, scene_as, RayDesc(flags, 0xffu, tmin, tmax, o + lo, d));
    while rayQueryProceed(&q) {}
    let c = rayQueryGetCommittedIntersection(&q);
    if c.kind == RAY_QUERY_INTERSECTION_NONE { return Hit(tmax, 0xffffffffu, vec3<f32>(0.0)); }
    return Hit(c.t, c.primitive_index, vec3<f32>(1.0-c.barycentrics.x-c.barycentrics.y, c.barycentrics.x, c.barycentrics.y));
}
