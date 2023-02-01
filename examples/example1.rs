use std::ops::Range;

use iglo::os::Window;
use iglo::rhi::vulkan::*;

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

    while !window.should_close() {
        window.poll_events();
    }
}
