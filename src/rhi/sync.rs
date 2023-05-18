use std::future::*;
use std::marker::PhantomData;
use std::mem::*;
use std::pin::*;
use std::task::*;

use crate::rhi;

pub struct GpuFuture<'a, T, S = rhi::Semaphore, F = rhi::Fence>
where
    T: Send + Sync,
    S: rhi::SemaphoreApi,
    F: rhi::FenceApi,
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
    S: rhi::SemaphoreApi,
    F: rhi::FenceApi,
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
        S2: rhi::SemaphoreApi,
        F2: rhi::FenceApi,
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
    use crate::rhi::{FenceApi, SemaphoreApi};

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
