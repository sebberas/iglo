use std::time::*;

use ::futures::executor::*;
use ::glam::*;
use iglo::os::*;
use iglo::rhi::vulkan::ShaderStage;
use iglo::rhi::*;

const VERTEX_SHADER_CODE: &[u8] = include_bytes!("./shader.vert.spv");
const PIXEL_SHADER_CODE: &[u8] = include_bytes!("./shader.frag.spv");

fn main() {
    let mut pool = LocalPool::new();
    let future = async {
        let mut window = Window::new("iglo").unwrap();
        let mut renderer = Renderer::new(&window);

        window.show();
        while !window.should_close() {
            let _ = window.poll_events();
            renderer.update(window.extent());
        }
    };

    pool.run_until(future);
}

struct Renderer {
    instance: Instance,
    device: Device,
    swapchain: DSwapchain,
    render_pass: RenderPass,
    pipeline: Pipeline,
}

impl Renderer {
    pub fn new(window: &Window) -> Self {
        let instance = Instance::new(Backend::Vulkan, true).unwrap();

        let mut adapters = instance.enumerate_adapters();
        let adapter = adapters.next().unwrap();

        let surface = instance.new_surface(window).unwrap();

        let device_props = DeviceProps {
            adapter: Some(adapter),
            surface: Some(&surface),
            ..Default::default()
        };
        let device = instance.new_device(device_props).unwrap();

        drop(adapters);

        let render_pass = {
            let attachments = [Attachment::Color {
                format: Format::R8G8B8A8Unorm,
                load_op: AttachmentLoadOp::DontCare,
                store_op: AttachmentStoreOp::Store,
                layout: Layout::Present,
            }];

            device.new_render_pass(&attachments).unwrap()
        };

        let swapchain = instance.new_dswapchain(&device, surface).unwrap();

        let vertex_shader = device.new_shader(VERTEX_SHADER_CODE).unwrap();
        let pixel_shader = device.new_shader(PIXEL_SHADER_CODE).unwrap();

        let pipeline_state = PipelineState {
            viewport: None,
            multisample: MultisampleState {
                samples: Samples::default(),
            },
        };

        let pipeline = device
            .new_pipeline(
                &pipeline_state,
                &[
                    (vertex_shader, ShaderStage::Vertex),
                    (pixel_shader, ShaderStage::Pixel),
                ],
                &render_pass,
            )
            .unwrap();

        Self {
            instance,
            device,
            swapchain,
            render_pass,
            pipeline,
        }
    }

    pub fn update(&mut self, extent: UVec2) {
        let Self { device, .. } = self;

        let start = std::time::Instant::now();

        let mut queue = device.queue(operations::Graphics).unwrap();
        let mut command_pool = device.new_command_pool(&queue).unwrap();
        let mut command_list = device.new_command_list(&mut command_pool).unwrap();

        let mut fence = device.new_fence(false).unwrap();
        let image = unsafe {
            self.swapchain
                .image_unchecked(&mut fence, Duration::from_secs(2))
        }
        .unwrap()
        .unwrap();

        fence.wait(Duration::from_secs(12)).unwrap();
        fence.reset().unwrap();

        let mut framebuffer = device
            .new_framebuffer(&self.render_pass, &[image.clone()], extent)
            .unwrap();

        let command_list = command_list.as_mut();
        unsafe {
            command_list.begin_unchecked(command_pool.as_mut()).unwrap();
            command_list.begin_render_pass_unchecked(&self.render_pass, &mut framebuffer);

            command_list.bind_pipeline_unchecked(&self.pipeline);
            command_list.set_viewport_unchecked(&ViewportState {
                position: vec2(0.0, 0.0),
                extent: extent.as_vec2(),
                depth: 0.0..=1.0,
                scissor: Some(ScissorState {
                    offset: uvec2(0, 0),
                    extent,
                }),
            });

            command_list.draw_unchecked(3, 1);

            command_list.end_render_pass_unchecked();
            command_list.end_unchecked().unwrap();
        }

        let submit_info = [SubmitInfo {
            command_lists: vec![command_list],
            wait_semaphores: vec![],
            signal_semaphores: vec![],
        }];

        unsafe {
            queue
                .as_mut()
                .submit_unchecked(submit_info.into_iter(), Some(&mut fence))
                .unwrap()
        };

        fence.wait(Duration::from_secs_f32(10.0)).unwrap();

        self.swapchain.present(&image).unwrap();

        let end = std::time::Instant::now();
        println!("{:?}", end - start);
    }
}
