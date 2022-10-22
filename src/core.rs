use std::future::*;
use std::pin::*;
use std::task::*;

use futures::prelude::*;

pub struct Executor {}

impl Executor {
    pub fn new() -> Self {
        Self {}
    }

    pub fn run<F: Future>(&mut self, future: F) -> F::Output {
        const VTABLE: RawWakerVTable = RawWakerVTable::new(
            |data: *const ()| RawWaker::new(data, &VTABLE),
            |_: *const ()| {},
            |_: *const ()| {},
            |_: *const ()| {},
        );

        let waker = RawWaker::new(std::ptr::null(), &VTABLE);
        let waker = unsafe { Waker::from_raw(waker) };
        let mut cx = Context::from_waker(&waker);

        let mut pinned = pin!(future);
        loop {
            if let Poll::Ready(o) = pinned.as_mut().poll(&mut cx) {
                return o;
            }
        }
    }
}
