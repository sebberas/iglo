use std::future::*;
use std::marker::PhantomData;
use std::mem::*;
use std::pin::*;
use std::task::*;

use crate::rhi;
use crate::rhi::*;

pub trait SemaphoreApi: Send + Sync {}

/// Semaphores are used for synchronizing primarily between device queues, but
/// also provides mechanisms for blocking on the CPU.
///
/// Semaphores can represent a wide range of values depending on the current
/// state of them.
///
/// This semaphore is the most commonly used one, but there also exists a binary
/// semaphore that can only represent 0 or 1.
pub enum Semaphore {
    Vulkan(vulkan::Semaphore),
}

impl Semaphore {
    /// Waits until the semaphore has reached `value` or until `timeout` has
    /// elapsed.
    ///
    /// # Returns
    ///
    /// Returns true if this semaphore reached `value` before the timeout.
    ///
    /// # Platform Specific
    /// - **Vulkan:** Timeouts longer than 584.55 years are clamped down.
    pub fn wait(&mut self, value: u64, timeout: Duration) -> Result<bool, Error> {
        match self {
            Self::Vulkan(e) => e.wait(value, timeout),
        }
    }

    /// Sets the value of this semaphore to `value`.
    ///
    /// # Panics
    ///
    /// Panics if `value` is less than the current value.
    pub fn signal(&mut self, value: u64) -> Result<(), Error> {
        match self {
            Self::Vulkan(e) => e.signal(value),
        }
    }

    /// Resets the value of this semaphore to `value`.
    pub fn reset(&mut self, value: u64) -> Result<(), Error> {
        match self {
            Self::Vulkan(e) => e.reset(value),
        }
    }

    /// Executes `f` when the value of this semaphore changes.
    pub fn on_signal(&mut self, f: impl Fn(u64) + 'static) {
        match self {
            Self::Vulkan(e) => e.on_signal(f),
        }
    }

    /// Executes `f` when the value of this semaphore reaches `value`.
    pub fn on_value(&mut self, value: u64, f: impl FnOnce() + 'static) {
        match self {
            Self::Vulkan(e) => e.on_value(value, f),
        }
    }
}

impl SemaphoreApi for Semaphore {}

pub trait FenceApi: Sized + Send + Sync {
    /// Waits until this fence has been signaled or until `timeout`
    /// has elapsed.
    /// # Returns
    ///
    /// Returns true if this semaphore reached `value` before the timeout.
    ///
    /// # Platform Specific
    /// - **Vulkan:** Timeouts longer than 584.55 years are clamped down.
    fn wait(&self, timeout: Duration) -> Result<bool, Error>;

    /// Returns whether this fence is in a signaled state.
    fn signaled(&self) -> Result<bool, Error>;

    /// Resets this fence back to being unsignaled.
    fn reset(&mut self) -> Result<(), Error>;

    /// Leaks the internal fence handle without waiting on the CPU.
    ///
    /// If the fence is already signaled no memory leak will occur and the fence
    /// will be destroyed correctly.
    ///
    /// This means that the handle will never be returned to the API essentially
    /// causing a memory leak in the Driver/GPU.
    fn leak(self);

    /// Attaches a callback to the fence that is executed when it is signaled.
    fn on_signal(&mut self, f: impl Fn()) {}

    /// Attaches a callback to the fence that is executed when it is reset.
    fn on_reset(&mut self, f: impl Fn()) {}
}

/// **WARNING** Dropping a fence that is still in use by the GPU will cause the
/// thread that is dropping the fence to block until the GPU has completed its
/// operation and signaled the fence.
pub enum Fence {
    Vulkan(vulkan::Fence),
}

impl FenceApi for Fence {
    fn wait(&self, timeout: Duration) -> Result<bool, Error> {
        match self {
            Self::Vulkan(fence) => fence.wait(timeout),
        }
    }

    fn signaled(&self) -> Result<bool, Error> {
        match self {
            Self::Vulkan(fence) => fence.signaled(),
        }
    }

    fn reset(&mut self) -> Result<(), Error> {
        match self {
            Self::Vulkan(fence) => fence.reset(),
        }
    }

    fn leak(self) {
        match self {
            Self::Vulkan(fence) => fence.leak(),
        }
    }
}

pub enum BinarySemaphore {
    Vulkan(vulkan::BinarySemaphore),
}

pub struct GpuFuture<'a, T, S = Semaphore, F = Fence>
where
    T: Send + Sync,
    S: SemaphoreApi,
    F: FenceApi,
{
    f: Option<Box<dyn FnOnce(&mut [u8]) + Send + Sync + 'a>>,
    data: Vec<u8>,
    fence: Option<F>,
    semaphores: Vec<S>,
    _marker: PhantomData<T>,
}

impl<'a, T, S, F> GpuFuture<'a, T, S, F>
where
    T: Send + Sync + 'a,
    S: SemaphoreApi,
    F: FenceApi,
{
    pub fn new(f: Box<dyn FnOnce() -> T + Send + Sync>, fence: F) -> Self {
        let data = vec![0; std::mem::align_of::<T>() + std::mem::size_of::<T>()];
        println!("{}", data.len());

        let f = Box::new(|data: &mut [u8]| {
            // SAFETY:
            // This call is safe because `data` is aligned to the same multiple as
            // `T`
            unsafe { std::ptr::copy(&f(), data.as_mut_ptr() as *mut T, 1) };
        });

        Self {
            f: Some(f),
            data,
            fence: Some(fence),
            semaphores: Vec::default(),
            _marker: PhantomData,
        }
    }

    pub fn cast<T2, S2, F2>(self) -> GpuFuture<'a, T2, S2, F2>
    where
        T: Into<T2>,
        S: Into<S2>,
        F: Into<F2>,
        T2: Send + Sync,
        S2: SemaphoreApi,
        F2: FenceApi,
    {
        GpuFuture::<T2, S2, F2> {
            f: self.f,
            data: self.data,
            fence: self.fence.map(Into::into),
            semaphores: self.semaphores.into_iter().map(Into::into).collect(),
            _marker: PhantomData,
        }
    }
}

// impl<T, S, F> Future for GpuFuture<T, S, F>
// where
//     Self: Send + Sync,
//     S: rhi::SemaphoreApi,
//     F: rhi::FenceApi,
// {
//     type Output = (T, F);

//     fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>
// {         let Self { f, data, fence, .. } = unsafe { self.get_unchecked_mut()
// };

//         if let Some(f) = f.take() {
//             *data = MaybeUninit::new(f.call_once(()));
//         } else if fence.is_some() {
//             if fence
//                 .as_ref()
//                 .map(|e| e.signaled().unwrap())
//                 .unwrap_or(false)
//             {
//                 let mut value = MaybeUninit::uninit();
//                 std::mem::swap(&mut value, data);

//                 // SAFETY:
//                 // - We know that the value is available when the fence has
// been signaled.                 // - We swap `data` with the uninitialized
// bytes in`value` leaving `data`                 //   uninitialized.
//                 return Poll::Ready((unsafe { value.assume_init() },
// fence.take().unwrap()));             }
//         } else {
//             unreachable!("GpuFuture has already returned Poll::Ready");
//         }

//         Poll::Pending
//     }
// }

// pub struct GpuFutureWithTimeout<T, F = rhi::Fence> {
//     f: Option<Box<dyn FnOnce() -> T>>,
//     data: MaybeUninit<T>,
//     fence: Option<F>,
// }

// impl<T, F: FenceApi> Future for GpuFutureWithTimeout<T, F> {
//     type Output = Option<T>;

//     fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>
// {         todo!()
//     }
// }

pub mod tests {
    use super::*;
    // use crate::rhi::{FenceApi, SemaphoreApi};

    struct Fence;
    impl FenceApi for Fence {
        fn wait(&self, timeout: std::time::Duration) -> Result<bool, rhi::Error> {
            todo!()
        }

        fn signaled(&self) -> Result<bool, rhi::Error> {
            todo!()
        }

        fn reset(&mut self) -> Result<(), rhi::Error> {
            todo!()
        }

        fn leak(self) {
            todo!()
        }
    }

    struct Semaphore;
    impl SemaphoreApi for Semaphore {}

    #[test]
    fn test_gpu_future() {
        let fence = Fence;
        let mut future = GpuFuture::<_, Semaphore, _>::new(Box::new(|| String::from("Hej")), fence);

        future.f.unwrap()(&mut future.data);
        println!("{:?}", &future.data);
        let v = unsafe { &*(future.data.as_ptr() as *const String) };
        assert_eq!(v, &String::from("Hej"));
    }
}
