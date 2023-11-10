#![feature(const_size_of_val)]

use std::num::NonZeroUsize;
use std::time::*;

use ::futures::executor::*;
use ::glam::*;
use futures::task::SpawnExt;
use iglo::os::*;
use iglo::rhi::vulkan::ShaderStage;
use iglo::rhi::*;

const EXTENT: UVec2 = uvec2(1366, 768);

fn main() {
    let pool = ThreadPool::new().unwrap();
    let future = async {
        let mut window = Window::new("iglo").unwrap();
        let mut renderer = Renderer::new(&window);

        window.show();
        while !window.should_close() {
            let _ = window.poll_events();
            renderer.update(EXTENT);
        }
    };

    let handle = pool.spawn_with_handle(future).unwrap();
    block_on(handle);
}

struct TransferObjects {
    queue: TransferQueue,
    command_pool: TransferCommandPool,
    command_list: TransferCommandList,
    semaphore: BinarySemaphore,
}

struct Renderer {
    device: Device,
    queue: GraphicsQueue,
    command_pool: GraphicsCommandPool,
    command_lists: [GraphicsCommandList; 3],
    render_pass: RenderPass,
    pipeline: Pipeline,
    vertex_buffer: DBuffer,
    uniform_buffer: DBuffer,
    transfer_semaphore: BinarySemaphore,

    swapchain: DSwapchain,

    synchronization: Vec<(BinarySemaphore, BinarySemaphore, Fence)>,
    framebuffers: Vec<Framebuffer>,
    frame: usize,
}

impl Renderer {
    const MAX_FRAMES_IN_FLIGHT: usize = 3;

    const VERTEX_SHADER_CODE: &[u8] = include_bytes!("./vertex_shader.spirv");
    const PIXEL_SHADER_CODE: &[u8] = include_bytes!("./pixel_shader.spirv");

    const VERTICES: [(Vec4, Vec4); 6] = [
        // left
        (vec4(-0.5, -0.5, 0.0, 1.0), vec4(1.0, 0.0, 0.0, 1.0)),
        (vec4(0.5, 0.5, 0.0, 1.0), vec4(0.0, 1.0, 0.0, 1.0)),
        (vec4(-0.5, 0.5, 0.0, 1.0), vec4(0.0, 0.0, 1.0, 1.0)),
        // right
        (vec4(0.5, 0.5, 0.0, 1.0), vec4(0.0, 1.0, 0.0, 1.0)),
        (vec4(-0.5, -0.5, 0.0, 1.0), vec4(1.0, 0.0, 0.0, 1.0)),
        (vec4(0.5, -0.5, 0.0, 1.0), vec4(0.0, 0.0, 1.0, 1.0)),
    ];

    const VERTICES_SIZE: usize = std::mem::size_of_val(&Self::VERTICES);

    pub fn new(window: &Window) -> Self {
        let (device, swapchain) = Self::setup_device_and_swapchain(window);
        let (queue, command_pool, command_lists) = Self::setup_queue_and_command_objects(&device);
        let render_pass = Self::setup_render_pass(&device);
        let pipeline = Self::setup_pipeline(&device, &render_pass);
        let (vertex_buffer, transfer_semaphore) = Self::setup_vertex_buffer(&device);

        let mut framebuffers = Vec::with_capacity(3);
        for i in 0..3 {
            let framebuffer = device.new_framebuffer(&render_pass, &[unsafe { swapchain.image_unchecked(i) }], EXTENT);

            framebuffers.push(framebuffer.unwrap())
        }

        let mut synchronization = Vec::with_capacity(3);
        for _ in 0..3 {
            let semaphore1 = device.new_binary_semaphore().unwrap();
            let semaphore2 = device.new_binary_semaphore().unwrap();
            let fence = device.new_fence(false).unwrap();

            synchronization.push((semaphore1, semaphore2, fence));
        }

        let uniform_buffer = device
            .new_dbuffer(&DBufferProps {
                size: NonZeroUsize::new(std::mem::size_of::<Mat4>() * 3).unwrap(),
                usage: Usage::Uniform,
            })
            .unwrap();

        let matrices: [Mat4; 3] = [Mat4::IDENTITY, Mat4::IDENTITY, Mat4::IDENTITY];
        let data = unsafe { uniform_buffer.map_unchecked() }.unwrap();
        unsafe { (data.as_mut_ptr() as *mut Mat4).copy_from(matrices.as_ptr(), 3) };
        unsafe { uniform_buffer.unmap_unchecked() };

        let descriptor_pool = device.new_descriptor_pool().unwrap();

        Self {
            device,
            queue,
            command_pool,
            command_lists,
            swapchain,
            render_pass,
            pipeline,
            vertex_buffer,
            uniform_buffer,
            transfer_semaphore,

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
            dcommand_list.reset_unchecked(self.command_pool.as_mut()).unwrap();
            dcommand_list.begin_unchecked(self.command_pool.as_mut()).unwrap();

            dcommand_list.begin_render_pass_unchecked(&self.render_pass, &mut self.framebuffers[image_view_i]);

            dcommand_list.bind_pipeline_unchecked(&self.pipeline);

            dcommand_list.bind_vertex_buffers_unchecked([&self.vertex_buffer]);

            dcommand_list.set_viewport_unchecked(&ViewportState {
                position: vec2(0.0, 0.0),
                extent: extent.as_vec2(),
                depth: 0.0..=1.0,
                scissor: Some(ScissorState { offset: uvec2(0, 0), extent }),
            });

            dcommand_list.draw_unchecked(Self::VERTICES.len(), 1);

            dcommand_list.end_render_pass_unchecked();
            dcommand_list.end_unchecked().unwrap();
        }

        let submit_info = SubmitInfo {
            command_lists: vec![dcommand_list],
            wait_semaphores: vec![(
                SemaphoreSubmitInfo::Binary(image_available),
                PipelineStage::ColorAttachmentOutput,
            )],
            signal_semaphores: vec![SemaphoreSubmitInfo::Binary(render_finished)],
        };

        let dqueue = self.queue.as_mut();
        unsafe { dqueue.submit_unchecked([submit_info], Some(in_flight)).unwrap() };

        self.swapchain.present(&image_view, [render_finished]).unwrap();

        self.frame = (self.frame + 1) % Self::MAX_FRAMES_IN_FLIGHT;
    }
}

impl Renderer {
    fn setup_device_and_swapchain(window: &Window) -> (Device, DSwapchain) {
        let instance = Instance::new(Backend::Vulkan, true).unwrap();
        let adapter = instance.iter_adapters().next().unwrap();

        let surface = instance.new_surface(window).unwrap();

        let device_props = DeviceProps {
            adapter: Some(adapter),
            surface: Some(&surface),
            graphics_queues: None,
            compute_queues: todo!(),
            transfer_queues: todo!(),
        };

        let device = instance.new_device(device_props).unwrap();

        let swapchain_props = SwapchainProps {
            device: &device,
            surface,
            images: NonZeroUsize::new(3).unwrap(),
            image_format: Format::R8G8B8A8Unorm,
            image_extent: EXTENT,
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

    fn setup_transfer_objects(device: &Device) -> TransferObjects {
        let queue = device.queue(ops::Transfer).unwrap();
        let mut command_pool = device.new_command_pool(&queue).unwrap();
        let command_list = device.new_command_list(&mut command_pool).unwrap();
        let semaphore = device.new_binary_semaphore().unwrap();

        TransferObjects {
            queue,
            command_pool,
            command_list,
            semaphore,
        }
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

        let shaders = [(vertex_shader, ShaderStage::Vertex), (pixel_shader, ShaderStage::Pixel)];

        let state = PipelineState {
            vertex_input: Some(VertexInputState {
                bindings: vec![VertexInputBinding {
                    binding: 0,
                    stride: std::mem::size_of::<(Vec4, Vec4)>(),
                    rate: VertexInputRate::Vertex,
                }],
                attributes: vec![
                    VertexInputAttribute {
                        location: 0,
                        binding: 0,
                        format: Format::R32G32B32A32Float,
                        offset: 0,
                    },
                    VertexInputAttribute {
                        location: 1,
                        binding: 0,
                        format: Format::R32G32B32A32Float,
                        offset: std::mem::size_of::<Vec4>(),
                    },
                ],
            }),
            viewport: None,
            multisample: MultisampleState { samples: Samples::ONE },
        };

        device.new_pipeline(&state, &shaders, render_pass).unwrap()
    }

    fn setup_staging_buffer(device: &Device) -> DBuffer {
        let staging_buffer = device
            .new_dbuffer(&DBufferProps {
                size: NonZeroUsize::new(Self::VERTICES_SIZE).unwrap(),
                usage: Usage::Vertex,
            })
            .unwrap();

        let data = unsafe { staging_buffer.map_unchecked() }.unwrap();
        let ptr = data.as_mut_ptr();
        for (i, vertex) in Self::VERTICES.iter().enumerate() {
            unsafe { std::ptr::copy(vertex, (ptr as *mut (Vec4, Vec4)).add(i), 1) };
        }

        unsafe { staging_buffer.unmap_unchecked() };
        staging_buffer
    }

    fn setup_vertex_buffer(device: &Device) -> (DBuffer, BinarySemaphore) {
        let TransferObjects {
            mut queue,
            mut command_pool,
            mut command_list,
            semaphore,
        } = Self::setup_transfer_objects(device);

        let staging_buffer = Self::setup_staging_buffer(device);
        let vertex_buffer = device
            .new_dbuffer(&DBufferProps {
                size: NonZeroUsize::new(Self::VERTICES_SIZE).unwrap(),
                usage: Usage::Vertex,
            })
            .unwrap();

        let dcommand_list = command_list.as_mut();
        unsafe {
            dcommand_list.begin_unchecked(command_pool.as_mut()).unwrap();

            dcommand_list.copy_buffer_unchecked(
                command_pool.as_mut(),
                &staging_buffer,
                &vertex_buffer,
                Self::VERTICES_SIZE,
            );

            dcommand_list.end_unchecked().unwrap();
        }

        let submit_info = SubmitInfo {
            command_lists: vec![dcommand_list],
            wait_semaphores: vec![],
            signal_semaphores: vec![SemaphoreSubmitInfo::Binary(&semaphore)],
        };

        let mut fence = device.new_fence(false).unwrap();
        unsafe {
            queue
                .as_mut()
                .submit_unchecked([submit_info], Some(&mut fence))
                .unwrap()
        };

        fence.wait(Duration::MAX).unwrap();

        (vertex_buffer, semaphore)
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        self.device.wait_idle().unwrap();
    }
}
