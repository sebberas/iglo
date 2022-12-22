use iglo::os::*;
use iglo::rhi::*;

fn main() {
    let mut window = Window::new("iglo").unwrap();
    let instance = Instance::new(Backend::DX12, true).unwrap();
    let surface = unsafe { instance.new_surface(&window) }.unwrap();
    let device = instance.new_device(Some(&surface)).unwrap();
    let swapchain = instance.new_swapchain(&device, surface).unwrap();

    window.show();

    loop {
        window.poll_events()
    }
}
