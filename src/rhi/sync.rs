use std::future::*;
use std::marker::PhantomData;
use std::mem::*;
use std::pin::*;
use std::task::*;

pub struct Fence;

impl Fence {
    pub fn signaled(&self) -> bool {
        false
    }
}

/// Represents an asynchronous operation on the GPU.
///
/// Calling `.await` on a `GpuFuture` will cause the CPU to block until the
/// operation has completed. Just like regular futures, no work is started on
/// the GPU until `.await` has been called on the future.
pub struct GpuFuture<T> {
    f: Option<Box<dyn FnOnce() -> T>>,
    data: MaybeUninit<T>,
    fence: Option<Fence>,
}

impl<T> GpuFuture<T> {
    pub fn new(f: impl FnOnce() -> T + 'static, fence: Fence) -> Self {
        Self {
            f: Some(Box::new(f)),
            data: MaybeUninit::uninit(),
            fence: Some(fence),
        }
    }
}

impl<T> Future for GpuFuture<T> {
    type Output = (T, Fence);

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY:
        let Self { f, data, fence, .. } = unsafe { &mut self.get_unchecked_mut() };
        if let Some(f) = f.take() {
            *data = MaybeUninit::new(f.call_once(()));
        } else if fence.is_some() {
            if fence.as_ref().map(|e| e.signaled()).unwrap_or(false) {
                let mut value = MaybeUninit::uninit();
                std::mem::swap(&mut value, data);

                // SAFETY:
                // - We know that the value is available when the fence has been signaled.
                // - We swap `data` with the uninitialized bytes in`value` leaving `data`
                //   uninitialized.
                return Poll::Ready((unsafe { value.assume_init() }, fence.take().unwrap()));
            }
        } else {
            unreachable!("GpuFuture has already returned Poll::Ready");
        }

        Poll::Pending
    }
}

pub struct GpuRef<T> {
    _marker: PhantomData<T>,
}

pub struct GpuRefMut<T> {
    _marker: PhantomData<T>,
}
