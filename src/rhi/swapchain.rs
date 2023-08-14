use std::num::*;

use crate::rhi::*;

pub enum PresentMode {
    Immediate,
    Mailbox,
    Fifo,
    FifoRelaxed,
}

pub struct SwapchainProps<'a, D = rhi::Device, S = rhi::Surface> {
    pub device: &'a D,
    pub surface: S,
    pub images: NonZeroUsize,
    pub image_format: Format,
    pub image_extent: UVec2,
    pub present_mode: PresentMode,
}

pub enum DSwapchain {
    Vulkan(vulkan::DSwapchain),
}

impl DSwapchain {
    /// Returns the format of this swapchain.
    pub fn format(&self) -> Format {
        todo!()
    }

    /// Returns the amount of backbuffers of this swapchain.
    pub fn backbuffers(&self) -> usize {
        todo!()
    }

    /// Returns an immutable view into a swapchain image.
    pub unsafe fn image_unchecked(&self, i: usize) -> DImageView2D {
        match self {
            Self::Vulkan(swapchain) => swapchain.image_unchecked(i).into(),
        }
    }

    pub fn image_mut_unchecked(&self, i: Fence) -> ViewMut<DImage2D> {
        todo!()
    }

    /// Returns the index of the next available image from the swapchain.
    ///
    /// Users should call `image_mut_unchecked` to get a mutable view
    /// into the image.
    ///
    /// It is safe to read from the image while it is still in use by the
    /// swapchain.
    ///
    /// # Safety
    ///
    /// It is undefined behaviour to write to the image before `fence` has been
    /// signaled since it may still be read from by the swapchain.
    // TODO: Replace usize with an id.
    pub unsafe fn next_image_i_unchecked(
        &mut self,
        semaphore: &mut BinarySemaphore,
        fence: Option<&mut Fence>,
        timeout: Duration,
    ) -> Result<Option<usize>, Error> {
        match self {
            Self::Vulkan(swapchain) => {
                let semaphore = semaphore.try_into().unwrap();
                let fence = fence.map(|f| f.try_into().unwrap());
                swapchain.next_image_i_unchecked(semaphore, fence, timeout)
            }
        }
    }

    /// Acquires an image from the swapchain signaling `fence` when it is no
    /// longer used by the swapchain.
    ///
    /// If there is no available image before `timeout` it returns `None`.
    ///
    /// This is an unsafe version of [`DSwapchain::image`](DSwapchain).
    ///
    /// # Arguments
    ///
    /// - `fence` - The fence to signal when the image is available.
    ///
    /// # Safety
    ///
    /// It is undefined behaviour to use the image before `fence` has been
    /// signaled since it may still be read from by the swapchain.
    // pub unsafe fn next_image_unchecked(
    //     &mut self,
    //     fence: &mut Fence,
    //     timeout: Duration,
    // ) -> Result<Option<DImageView2D>, Error> {
    //     match self {
    //         Self::Vulkan(swapchain) => {
    //             let fence = fence.try_into().unwrap();
    //             let image = swapchain.image_unchecked(fence, timeout);
    //             image.map(|image| image.map(Into::into))
    //         }
    //     }
    // }

    /// Enumerates all the images in the swapchain.
    pub fn enumerate_images(&mut self) -> impl ExactSizeIterator<Item = &DImage2D> + '_ {
        const V: &[DImage2D] = &[];
        V.iter()
    }

    pub fn present<'a>(
        &mut self,
        image_view: &DImageView2D,
        wait_semaphores: impl IntoIterator<Item = &'a mut BinarySemaphore>,
    ) -> Result<(), Error> {
        match self {
            Self::Vulkan(swapchain) => {
                let image_view = image_view.try_into().unwrap();
                let wait_semaphores = wait_semaphores.into_iter().map(|s| s.try_into().unwrap());
                swapchain.present(image_view, wait_semaphores)
            }
        }
    }
}
