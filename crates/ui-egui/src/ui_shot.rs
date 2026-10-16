//! Test-only: a picture of the whole window as a person sees it (panels, the viewport drawn by
//! its GPU pipelines, the overlays and their text), rendered offscreen through egui-wgpu from a
//! [`Harness`] frame. `None` without a GPU adapter.

use egui_wgpu::wgpu;

use crate::gpu::{GpuTarget, Renderer};
use crate::scenario::Harness;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth24PlusStencil8;

/// The viewport target a shot draws into: let the harness build GPU scenes for it.
pub(crate) fn enable_viewport(h: &mut Harness) {
    h.app.viewport.gpu = Some(GpuTarget { format: FORMAT, depth: Some(DEPTH), samples: 1, tess: false });
}

/// Runs one frame and renders it: straight RGBA, the window's size in pixels at 1 point = 1 pixel.
pub(crate) fn rgba(h: &mut Harness) -> Option<(u32, u32, Vec<u8>)> {
    let (_, device, queue) = crate::gpu::offscreen_device()?;
    // Each shot draws on a renderer of its own: make the frame hand it every scene afresh.
    h.app.viewport.rebuild_scenes();
    let shapes = h.frame_shapes();
    let size = h.size();
    let (w, hgt) = (size.x as u32, size.y as u32);
    let mut egui_renderer = egui_wgpu::Renderer::new(
        &device,
        FORMAT,
        egui_wgpu::RendererOptions { depth_stencil_format: Some(DEPTH), msaa_samples: 1, ..Default::default() },
    );
    egui_renderer.callback_resources.insert(Renderer::new(&device, GpuTarget { format: FORMAT, depth: Some(DEPTH), samples: 1, tess: false }, None));
    for (id, deltas) in &h.textures.set {
        for delta in deltas {
            egui_renderer.update_texture(&device, &queue, *id, delta);
        }
    }
    let prims = h.ctx.tessellate(shapes, 1.0);
    let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: [w, hgt], pixels_per_point: 1.0 };
    let texture = |format, usage| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ui shot"),
            size: wgpu::Extent3d { width: w, height: hgt, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let color = texture(FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
    let depth = texture(DEPTH, wgpu::TextureUsages::RENDER_ATTACHMENT);
    let (color_view, depth_view) = (color.create_view(&Default::default()), depth.create_view(&Default::default()));
    let mut encoder = device.create_command_encoder(&Default::default());
    let extra = egui_renderer.update_buffers(&device, &queue, &mut encoder, &prims, &screen);
    {
        let mut pass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ui shot"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0), store: wgpu::StoreOp::Store }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            })
            .forget_lifetime();
        egui_renderer.render(&mut pass, &prims, &screen);
    }
    let row = (w * 4).div_ceil(256) * 256;
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ui shot read"),
        size: u64::from(row * hgt),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        color.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &read,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(hgt) },
        },
        wgpu::Extent3d { width: w, height: hgt, depth_or_array_layers: 1 },
    );
    queue.submit(extra.into_iter().chain([encoder.finish()]));
    let slice = read.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    let mapped = slice.get_mapped_range().ok()?.to_vec();
    let mut out = Vec::with_capacity((w * hgt * 4) as usize);
    for y in 0..hgt as usize {
        out.extend_from_slice(mapped.get(y * row as usize..y * row as usize + w as usize * 4)?);
    }
    Some((w, hgt, out))
}

/// [`rgba`] written as a PNG.
pub(crate) fn png(h: &mut Harness, path: &std::path::Path) -> Option<()> {
    let (w, hgt, rgba) = rgba(h)?;
    let bytes = solidcraft_engine::render::png(w as usize, hgt as usize, rgba)?;
    std::fs::write(path, bytes).ok()
}
