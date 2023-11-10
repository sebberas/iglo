use ::iglo::os::*;
use ::iglo::rhi::*;

fn main() {
    let mut window = Window::new("Iglo").expect("Failed to create window");
    window.show();

    let instance = Instance::new(iglo::rhi::Backend::Vulkan, true).unwrap();

    for adapter in instance.iter_adapters() {
        println!("{}: {:?}", adapter.name(), adapter.kind());
    }

    let adapter = instance.iter_adapters().nth(0).unwrap();
    let surface = instance.new_surface(&window).unwrap();

    let device_props = DeviceProps {
        adapter: Some(adapter),
        surface: Some(&surface),
        ..Default::default()
    };

    let device = instance.new_device(device_props).unwrap();
    loop {}
}
