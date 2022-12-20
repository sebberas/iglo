use iglo::os::*;
use iglo::rhi::*;

fn main() {
    let mut window = Window::new("iglo").unwrap();
    let instance = Instance::new(Backend::D3D12, true).unwrap();
    let device = instance.new_device().unwrap();

    window.show();

    loop {
        window.poll_events()
    }
}
