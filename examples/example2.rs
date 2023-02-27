#![feature(pin_macro)]
use std::{pin::pin, future::Future};

use iglo::rhi::sync::*;

fn main() {
    let future = pin!(GpuFuture::new(|| {}, Fence));

    let vtable = std::task::RawWakerVTable::new(, wake, wake_by_ref, drop);
    let waker = std::task::RawWaker::new(std::ptr::null(), )
    let cx = std::task::Context::from_waker(waker)
    loop {
        match future.poll() {
            std::task::Poll::Ready(v) => {}
            std::task::Poll::Pending => {}
        }
    }
}
