use std::num::NonZeroUsize;

use iglo::os::*;
use iglo::rhi::*;

fn main() {
    let mut window = Window::new("iglo").unwrap();
    let instance = Instance::new(Backend::DX12, true).unwrap();
    let surface = unsafe { instance.new_surface(&window) }.unwrap();

    let props = DeviceProps {
        surface: Some(&surface),
        ..Default::default()
    };

    let device = instance.new_device(&props).unwrap();
    let mut swapchain = instance.new_swapchain(&device, surface).unwrap();

    let mut props = ImageProps::builder()
        .width(NonZeroUsize::new(256).unwrap())
        .height(NonZeroUsize::new(256).unwrap())
        .format(format::R8G8B8A8Unorm)
        .usage(usage::DepthStencil);

    let depth = device.new_image(&mut props).unwrap();

    let graphics_queue = device.new_command_queue(queue::Graphics).unwrap().unwrap();

    window.show();
    while !window.should_close() {
        window.poll_events();
    }
}
