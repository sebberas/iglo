use iglo::os::*;
use iglo::rhi::*;

fn main() {
    let mut window = Window::new("iglo").unwrap();
    let instance = Instance::new(Backend::DX12, true).unwrap();
    let surface = unsafe { instance.new_surface(&window) }.unwrap();
    let device = instance.new_device(Some(&surface)).unwrap();
    let mut swapchain = instance.new_swapchain(&device, surface).unwrap();

    let command_queue = device.new_command_queue().unwrap();
    let mut command_pool = device.new_command_pool().unwrap();
    let mut command_list = device.new_command_list(&mut command_pool).unwrap();

    swapchain.present().unwrap();
    window.show();

    loop {
        window.poll_events();
    }
}
