// Trace-only path kernel: the renderer's ray pattern with no shading and no ray I/O.
// Per pixel: camera ray, then per surface vertex one shadow ray per area light and a
// cosine bounce, for bench.z vertices. Traversal is bench_trace (software or hardware).
@compute @workgroup_size(64)
fn paths(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.x; if pixel >= u.image.x*u.image.y { return; }
    var s = hash(pixel ^ bench.y);
    let xy = vec2<f32>(f32(pixel % u.image.x)+rnd(&s), f32(pixel / u.image.x)+rnd(&s));
    let uv = vec2<f32>(xy.x/f32(u.image.x)*2.0-1.0, 1.0-xy.y/f32(u.image.y)*2.0);
    let camera = primary_ray(uv, vec2<f32>(rnd(&s), rnd(&s)));
    var o = camera.origin; var lo = vec3<f32>(0.0); var d = camera.direction;
    var excluded = 0xffffffffu;
    var traced = 0u; var vertices = 0u; var occluded = 0u;
    for (var b = 0u; b < bench.z; b += 1u) {
        let h = bench_trace(o, lo, d, false, excluded, u.lighting.x, FAR);
        traced += 1u;
        if h.triangle == 0xffffffffu { break; }
        vertices += 1u;
        let t = triangle(h.triangle);
        // Opaque launch exactly as sample_path: offset along the geometric normal, no low part.
        let position = surface_point(t, h.bary);
        var n = normalize(cross(t.p1.xyz-t.p0.xyz, t.p2.xyz-t.p0.xyz));
        if dot(n, d) > 0.0 { n = -n; }
        let launch = position + n*(u.lighting.x*2.0);
        // direct_studio: one shadow ray per area light that faces the surface.
        for (var li = 0u; li < u32(u.light_count.w); li += 1u) {
            let l = u.area_lights[li];
            let q = l.center.xyz + l.u.xyz*(rnd(&s)*2.0-1.0) + l.v.xyz*(rnd(&s)*2.0-1.0);
            let delta = q - position;
            if dot(delta, n) <= 0.0 { continue; }
            let shadow = q - launch; let dist = length(shadow);
            let blocked = bench_trace(launch, vec3<f32>(0.0), shadow/dist, true, 0xffffffffu, u.lighting.x, dist);
            traced += 1u;
            occluded += select(0u, 1u, blocked.triangle != 0xffffffffu);
        }
        let a = 2.0*PI*rnd(&s); let r2 = rnd(&s);
        d = from_local(vec3<f32>(cos(a)*sqrt(r2), sin(a)*sqrt(r2), sqrt(1.0-r2)), n);
        o = launch; lo = vec3<f32>(0.0); excluded = 0xffffffffu;
    }
    hits[pixel] = vec4<f32>(bitcast<f32>(traced), bitcast<f32>(vertices), bitcast<f32>(occluded), 0.0);
}
