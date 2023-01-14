use iglo::os::Window;
use iglo::renderer::{Renderer, RendererProps};
use iglo::rhi::Backend;

fn main() {
    let mut window = Window::new("Iglo Application").unwrap();

    let mut renderer = unsafe {
        Renderer::new(RendererProps {
            backend: Backend::DX12,
            window: &window,
        })
    };

    window.show();

    while !window.should_close() {
        window.poll_events();

        renderer.render();
    }
}
