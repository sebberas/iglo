use iglo::os::*;
use iglo::rhi::vulkan::*;
// use iglo::rhi::*;

const VERTEX_SHADER: &[u8] = include_bytes!("./shader.vert.spv");
const PIXEL_SHADER: &[u8] = include_bytes!("./shader.frag.spv");

fn main() {
    let window = Window::new("iglo").unwrap();

    let instance = Instance::new(true).unwrap();
    let adapter = instance.enumerate_adapters().next();
    let surface = instance.new_surface(&window).unwrap();

    let device = instance
        .new_device(DeviceProps {
            adapter,
            surface: Some(&surface),
            ..Default::default()
        })
        .unwrap();

    let mut swapchain = instance.new_swapchain(&device, surface).unwrap();

    let mut fence = device.new_fence(false).unwrap();

    let _ = swapchain.image(None, Some(&mut fence));

    while !window.should_close() {
        let _ = window.poll_events();
    }
}
