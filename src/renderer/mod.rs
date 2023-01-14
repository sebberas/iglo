use std::num::NonZeroUsize;

use crate::os::Window;
use crate::rhi::*;

pub struct RendererProps {
    pub window: *const Window,
    pub backend: Backend,
}

impl AsRef<RendererProps> for RendererProps {
    fn as_ref(&self) -> &RendererProps {
        self
    }
}

impl RendererProps {
    pub fn builder(window: *const Window) -> RendererPropsBuilder {
        RendererPropsBuilder(Self {
            window,
            backend: Backend::DX12,
        })
    }
}

pub struct RendererPropsBuilder(RendererProps);

impl RendererPropsBuilder {
    pub fn backend(&mut self, backend: Backend) -> &mut Self {
        self.0.backend = backend;
        self
    }
}

impl AsRef<RendererProps> for RendererPropsBuilder {
    fn as_ref(&self) -> &RendererProps {
        &self.0
    }
}

pub struct Renderer {
    instance: Instance,
    device: Device,
    swapchain: Swapchain<format::R8G8B8A8Unorm>,
}

impl Renderer {
    ///
    /// # Safety
    ///
    /// `window` must outlive the renderer.
    pub unsafe fn new(props: impl AsRef<RendererProps>) -> Self {
        let props = props.as_ref();

        let instance = Instance::new(props.backend, false).unwrap();
        let surface = instance.new_surface(props.window).unwrap();

        let device_props = DeviceProps {
            surface: Some(&surface),
            ..Default::default()
        };

        let device = instance.new_device(&device_props).unwrap();

        let swapchain = instance
            .new_swapchain(SwapchainProps {
                device: &device,
                surface,
                width: None,
                height: None,
                format: format::R8G8B8A8Unorm,
                backbuffers: NonZeroUsize::new(1).unwrap(),
            })
            .unwrap();

        Self {
            instance,
            device,
            swapchain,
        }
    }

    pub fn render(&mut self) {
        let Self { device, .. } = self;

        let queue = device.new_command_queue(queue::Graphics).unwrap().unwrap();
        let mut pool = device.new_command_pool(&queue).unwrap();
        let mut list = device.new_command_list(&mut pool).unwrap();

        self.swapchain.present().unwrap();
    }
}
