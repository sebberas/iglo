use iglo::os::Window;
use iglo::rhi::vulkan::*;
use iglo::rhi::SwapchainProps;

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
    let swapchain = instance.new_swapchain(&device, surface).unwrap();

    let queue = device.queue(Operations::Graphics).unwrap();
    let mut pool = device.new_command_pool(&queue).unwrap();
    let mut list = device.new_command_list(&mut pool).unwrap();

    unsafe { list.begin_unchecked(&pool) }.unwrap();

    unsafe { list.end_unchecked() }.unwrap();

    while !window.should_close() {
        window.poll_events();
    }
}
