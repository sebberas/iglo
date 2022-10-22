#![feature(async_closure)]
use iglo::core::*;
use iglo::os::*;

fn main() {
    let f = async {
        let mut window = Window::new("iglo").unwrap();
        window.show();
    };

    let mut executor = Executor::new();
    executor.run(f);
}
