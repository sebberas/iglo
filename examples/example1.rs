use glam::uvec2;
use iglo::os::Window;
use iglo::rhi::vulkan::*;
use iglo::rhi::Format;

const VERTEX_SHADER: &[u8] = include_bytes!("./shader.vert.spv");
const PIXEL_SHADER: &[u8] = include_bytes!("./shader.frag.spv");

fn main() {
    let mut window = Window::new("Iglo Application").unwrap();
    window.show();

    let instance = Instance::new(true).unwrap();
    let surface = unsafe { instance.new_surface(&window) }.unwrap();

    let adapters: Vec<_> = instance.enumerate_adapters().collect();

    let device_props = DeviceProps {
        surface: Some(&surface),
        adapter: Some(adapters[0].clone()),
        ..Default::default()
    };

    let device = instance.new_device(device_props).unwrap();
    let mut swapchain = instance.new_swapchain(&device, surface).unwrap();

    let mut queue = device.queue(Operations::Graphics).unwrap();
    let mut pool = device.new_command_pool(&queue).unwrap();
    let mut list = device.new_command_list(&mut pool).unwrap();

    let vertex_shader = device.new_shader(VERTEX_SHADER).unwrap();
    let pixel_shader = device.new_shader(PIXEL_SHADER).unwrap();

    let attachments = [Attachment {
        format: Format::R8G8B8A8Unorm,
        load_op: AttachmentLoadOp::Clear,
        store_op: AttachmentStoreOp::Store,
        stencil_load_op: AttachmentLoadOp::DontCare,
        stencil_store_op: AttachmentStoreOp::DontCare,
    }];

    let render_pass_props = RenderPassProps {
        attachments: &attachments,
    };

    let render_pass = device.new_render_pass(&render_pass_props).unwrap();

    let pipeline_props = PipelineProps {
        stages: &[
            PipelineShaderStage {
                stage: ShaderStage::Vertex,
                shader: &vertex_shader,
                entrypoint: "main",
            },
            PipelineShaderStage {
                stage: ShaderStage::Pixel,
                shader: &pixel_shader,
                entrypoint: "main",
            },
        ],
        render_pass: &render_pass,
        layout: PipelineLayout {},
    };

    let pipeline = device.new_pipeline(&pipeline_props).unwrap();

    let signal = device.new_semaphore().unwrap();
    let mut fence = device.new_fence(false).unwrap();

    while !window.should_close() {
        let image_view = swapchain.image(None, Some(&mut fence)).unwrap().unwrap();
        fence.wait().unwrap();
        let mut framebuffer = device.new_framebuffer(&image_view, &render_pass).unwrap();

        window.poll_events();

        // Render loop
        unsafe { list.begin_unchecked(&pool) }.unwrap();
        unsafe { list.begin_render_pass_unchecked(&render_pass, &mut framebuffer) };

        unsafe { list.bind_pipeline_unchecked(&pipeline) };

        unsafe { list.set_viewport_unchecked(uvec2(0, 0), uvec2(640, 480), 0.0..=1.0) };
        unsafe { list.set_scissor_unchecked(uvec2(0, 0), uvec2(640, 480)) };

        unsafe { list.draw_unchecked(6, 0) };

        unsafe { list.end_render_pass_unchecked() };
        unsafe { list.end_unchecked() }.unwrap();

        unsafe { queue.submit_unchecked(&list, None, &signal, None) }.unwrap();
        unsafe { queue.wait_idle() }.unwrap();
        // unsafe { fence.reset() };

        swapchain.present(&mut queue, &image_view).unwrap();
        unsafe { queue.wait_idle() }.unwrap();

        fence.reset().unwrap();

        // let _ = swapchain.present(&mut queue);
    }
}
