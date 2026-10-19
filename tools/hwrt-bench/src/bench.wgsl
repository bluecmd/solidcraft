// Appended to SolidCraft's path.wgsl: the software side traces with the renderer's own
// f32 traversal (trace_ordered_within, the fast hybrid path) over the renderer's own BVH.
struct BenchRay { o: vec4<f32>, d: vec4<f32>, x: vec4<u32>, lo: vec4<f32> }; // o.w tmin, d.w tmax (<0 inactive), x.x excluded, x.y any-hit, lo.xyz origin low part
@group(1) @binding(0) var<storage, read_write> rays: array<BenchRay>;
@group(1) @binding(1) var<storage, read_write> hits: array<vec4<f32>>; // t, triangle, bary.y, bary.z
@group(1) @binding(2) var<storage, read_write> verts: array<f32>;
@group(1) @binding(3) var<uniform> bench: vec4<u32>; // ray count, seed, kind, -

fn hash(x: u32) -> u32 {
    var v = x; v ^= v >> 16u; v *= 0x7feb352du; v ^= v >> 15u; v *= 0x846ca68bu; v ^= v >> 16u; return v;
}
fn rnd(s: ptr<function,u32>) -> f32 { *s = hash(*s); return f32(*s >> 8u) / 16777216.0; }

@compute @workgroup_size(64)
fn export_vertices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if i >= geometry[0] { return; }
    let t = triangle(i);
    let p = array<vec3<f32>,3>(t.p0.xyz, t.p1.xyz, t.p2.xyz);
    for (var k = 0u; k < 3u; k += 1u) { verts[i*9u+k*3u] = p[k].x; verts[i*9u+k*3u+1u] = p[k].y; verts[i*9u+k*3u+2u] = p[k].z; }
}

@compute @workgroup_size(64)
fn primary_rays(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.x; if pixel >= u.image.x*u.image.y { return; }
    var s = hash(pixel ^ bench.y);
    let xy = vec2<f32>(f32(pixel % u.image.x)+rnd(&s), f32(pixel / u.image.x)+rnd(&s));
    let uv = vec2<f32>(xy.x/f32(u.image.x)*2.0-1.0, 1.0-xy.y/f32(u.image.y)*2.0);
    let r = primary_ray(uv, vec2<f32>(rnd(&s), rnd(&s)));
    rays[pixel] = BenchRay(vec4<f32>(r.origin, u.lighting.x), vec4<f32>(r.direction, FAR), vec4<u32>(0xffffffffu, 0u, 0u, 0u), vec4<f32>(0.0));
}

// From the previous hits (SW results): kind 0 = cosine diffuse bounce (closest hit),
// kind 1 = shadow ray to a point on area light 0 (any hit, bounded).
@compute @workgroup_size(64)
fn secondary_rays(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if i >= bench.x { return; }
    let r = rays[i]; let h = hits[i];
    let tri = bitcast<u32>(h.y);
    if r.d.w < 0.0 || tri == 0xffffffffu { rays[i].d.w = -1.0; return; }
    let t = triangle(tri);
    // As the renderer launches from a surface: a vertex anchor plus a local offset.
    let point = surface_point_precise(t, vec3<f32>(1.0-h.z-h.w, h.z, h.w));
    let p = point.hi + point.lo;
    var n = normalize(cross(t.p1.xyz-t.p0.xyz, t.p2.xyz-t.p0.xyz));
    if dot(n, r.d.xyz) > 0.0 { n = -n; }
    var s = hash(i ^ bench.y ^ 0x5bd1e995u);
    if bench.z == 0u {
        let a = 2.0*PI*rnd(&s); let b = rnd(&s);
        let local = vec3<f32>(cos(a)*sqrt(b), sin(a)*sqrt(b), sqrt(1.0-b));
        rays[i] = BenchRay(vec4<f32>(point.hi, u.lighting.x), vec4<f32>(from_local(local, n), FAR), vec4<u32>(tri, 0u, 0u, 0u), vec4<f32>(point.lo, 0.0));
    } else {
        let l = u.area_lights[0];
        let q = l.center.xyz + l.u.xyz*(rnd(&s)*2.0-1.0) + l.v.xyz*(rnd(&s)*2.0-1.0);
        let delta = q - p; let dist = length(delta);
        rays[i] = BenchRay(vec4<f32>(point.hi, u.lighting.x), vec4<f32>(delta/dist, dist*(1.0-1e-4)), vec4<u32>(tri, 1u, 0u, 0u), vec4<f32>(point.lo, 0.0));
    }
}

// Fast f32 traversal from the rounded origin (hi+lo); any-hit where requested.
@compute @workgroup_size(64)
fn sw_trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if i >= bench.x { return; }
    let r = rays[i]; if r.d.w < 0.0 { return; }
    let h = trace_ordered_within(r.o.xyz + r.lo.xyz, r.d.xyz, r.x.y != 0u, r.x.x, r.o.w, false, r.d.w, -1.0);
    hits[i] = vec4<f32>(h.distance, bitcast<f32>(h.triangle), h.bary.y, h.bary.z);
}

// What the renderer runs today: trace_precise_within from the anchored origin. Shadows
// are closest-hit with the opaque-blocker early-out, as shadows.wgsl calls it outside media.
@compute @workgroup_size(64)
fn sw_precise(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if i >= bench.x { return; }
    let r = rays[i]; if r.d.w < 0.0 { return; }
    var h: Hit;
    if r.x.y != 0u { h = trace_precise_within(r.o.xyz, r.lo.xyz, r.d.xyz, false, r.x.x, r.o.w, false, r.d.w, r.d.w); }
    else { h = trace_precise_within(r.o.xyz, r.lo.xyz, r.d.xyz, false, r.x.x, r.o.w, false, FAR, -1.0); }
    hits[i] = vec4<f32>(h.distance, bitcast<f32>(h.triangle), h.bary.y, h.bary.z);
}

// Memory floor: read the ray, write a result, trace nothing.
@compute @workgroup_size(64)
fn io_only(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if i >= bench.x { return; }
    let r = rays[i]; if r.d.w < 0.0 { return; }
    hits[i] = vec4<f32>(r.d.w, bitcast<f32>(r.x.x), r.lo.x, 0.0);
}
