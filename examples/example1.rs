use std::num::NonZeroUsize;
use std::time::*;

use ::futures::executor::*;
use ::glam::*;
use iglo::os::*;
use iglo::rhi::vulkan::ShaderStage;
use iglo::rhi::*;

fn main() {
    let mut pool = LocalPool::new();
    let future = async {
        let mut window = Window::new("iglo").unwrap();
        let mut renderer = Renderer::new(&window);

        window.show();
        while !window.should_close() {
            let _ = window.poll_events();
            renderer.update(window.inner_extent());
        }
    };

    pool.run_until(future);
}

struct Frame {
    framebuffer: Framebuffer,
    command_list: CommandList,
    image_available: Semaphore,
    render_finished: Semaphore,
    in_flight: Fence,
}

struct Renderer {
    device: Device,
    queue: Queue,
    command_pool: CommandPool,
    command_lists: [CommandList; 3],
    render_pass: RenderPass,
    pipeline: Pipeline,

    swapchain: DSwapchain,

    synchronization: Vec<(BinarySemaphore, BinarySemaphore, Fence)>,
    framebuffers: Vec<Framebuffer>,
    frame: usize,
}

impl Renderer {
    const VERTEX_SHADER_CODE: &[u8] = include_bytes!("./shader.vert.spv");
    const PIXEL_SHADER_CODE: &[u8] = include_bytes!("./shader.frag.spv");

    const MAX_FRAMES_IN_FLIGHT: usize = 3;

    pub fn new(window: &Window) -> Self {
        let (device, swapchain) = Self::setup_device_and_swapchain(window);
        let (queue, command_pool, command_lists) = Self::setup_queue_and_command_objects(&device);
        let render_pass = Self::setup_render_pass(&device);
        let pipeline = Self::setup_pipeline(&device, &render_pass);

        let mut framebuffers = Vec::with_capacity(3);
        for i in 0..3 {
            let framebuffer = device.new_framebuffer(
                &render_pass,
                &[unsafe { swapchain.image_unchecked(i) }],
                window.inner_extent(),
            );

            framebuffers.push(framebuffer.unwrap())
        }

        let mut synchronization = Vec::with_capacity(3);
        for _ in 0..3 {
            let semaphore1 = device.new_binary_semaphore().unwrap();
            let semaphore2 = device.new_binary_semaphore().unwrap();
            let fence = device.new_fence(false).unwrap();

            synchronization.push((semaphore1, semaphore2, fence));
        }

        Self {
            device,
            queue,
            command_pool,
            command_lists,
            swapchain,
            render_pass,
            pipeline,

            framebuffers,
            synchronization,
            frame: 0,
        }
    }

    pub fn update(&mut self, extent: UVec2) {
        let (image_available, render_finished, in_flight) = &mut self.synchronization[self.frame];

        in_flight.wait(Duration::MAX).unwrap();
        in_flight.reset().unwrap();

        let image_view_i = unsafe {
            self.swapchain
                .next_image_i_unchecked(image_available, None, Duration::MAX)
                .unwrap()
                .unwrap()
        };

        let image_view = unsafe { self.swapchain.image_unchecked(image_view_i) };

        let dcommand_list = self.command_lists[self.frame].as_mut();
        unsafe {
            dcommand_list
                .reset_unchecked(self.command_pool.as_mut())
                .unwrap();
            dcommand_list
                .begin_unchecked(self.command_pool.as_mut())
                .unwrap();

            dcommand_list.begin_render_pass_unchecked(
                &self.render_pass,
                &mut self.framebuffers[image_view_i],
            );

            dcommand_list.bind_pipeline_unchecked(&self.pipeline);
            dcommand_list.set_viewport_unchecked(&ViewportState {
                position: vec2(0.0, 0.0),
                extent: extent.as_vec2(),
                depth: 0.0..=1.0,
                scissor: Some(ScissorState { offset: uvec2(0, 0), extent }),
            });

            dcommand_list.draw_unchecked(3, 1);

            dcommand_list.end_render_pass_unchecked();
            dcommand_list.end_unchecked().unwrap();
        }

        let submit_info = SubmitInfo {
            command_lists: vec![dcommand_list],
            wait_semaphores: vec![SemaphoreSubmitInfo::Binary(image_available)],
            signal_semaphores: vec![SemaphoreSubmitInfo::Binary(render_finished)],
        };

        let dqueue = self.queue.as_mut();
        unsafe {
            dqueue
                .submit_unchecked([submit_info], Some(in_flight))
                .unwrap()
        };

        self.swapchain
            .present(&image_view, [render_finished])
            .unwrap();

        self.frame = (self.frame + 1) % Self::MAX_FRAMES_IN_FLIGHT;
    }

    fn setup_device_and_swapchain(window: &Window) -> (Device, DSwapchain) {
        let instance = Instance::new(Backend::Vulkan, false).unwrap();
        let adapter = instance.enumerate_adapters().next().unwrap();

        let surface = instance.new_surface(window).unwrap();

        let device_props = DeviceProps {
            adapter: Some(adapter),
            surface: Some(&surface),
            graphics_queues: None,
            compute_queues: None,
            transfer_queues: None,
        };

        let device = instance.new_device(device_props).unwrap();

        let swapchain_props = SwapchainProps {
            device: &device,
            surface,
            images: NonZeroUsize::new(3).unwrap(),
            image_format: Format::R8G8B8A8Unorm,
            image_extent: window.inner_extent(),
            present_mode: PresentMode::Immediate,
        };

        let swapchain = instance.new_dswapchain(swapchain_props).unwrap();
        (device, swapchain)
    }

    fn setup_queue_and_command_objects(device: &Device) -> (Queue, CommandPool, [CommandList; 3]) {
        let queue = device.queue(ops::Graphics).unwrap();
        let mut command_pool = device.new_command_pool(&queue).unwrap();
        let command_lists = [
            device.new_command_list(&mut command_pool).unwrap(),
            device.new_command_list(&mut command_pool).unwrap(),
            device.new_command_list(&mut command_pool).unwrap(),
        ];

        (queue, command_pool, command_lists)
    }

    fn setup_render_pass(device: &Device) -> RenderPass {
        let attachments = [Attachment::Color {
            format: Format::R8G8B8A8Unorm,
            load_op: AttachmentLoadOp::DontCare,
            store_op: AttachmentStoreOp::Store,
            layout: Layout::Present,
        }];

        device.new_render_pass(&attachments).unwrap()
    }

    fn setup_pipeline(device: &Device, render_pass: &RenderPass) -> Pipeline {
        let vertex_shader = device.new_shader(Self::VERTEX_SHADER_CODE).unwrap();
        let pixel_shader = device.new_shader(Self::PIXEL_SHADER_CODE).unwrap();

        let shaders = [
            (vertex_shader, ShaderStage::Vertex),
            (pixel_shader, ShaderStage::Pixel),
        ];

        let state = PipelineState {
            viewport: None,
            multisample: MultisampleState { samples: Samples::ONE },
        };

        device.new_pipeline(&state, &shaders, render_pass).unwrap()
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        self.device.wait_idle();
    }
}
