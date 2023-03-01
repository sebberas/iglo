use std::borrow::Cow;
use std::marker::PhantomData;
use std::num::NonZeroUsize;

use self::operations::{OperationsError, OperationsType};
use self::sync::GpuFuture;
use crate::os::Window;

pub mod sync;

// pub mod dx12;
pub mod vulkan;

pub mod spirv;

mod macros {
    /// Implements [`From`] and [`Into`] for converting away from the
    /// API-specific object, into the common object.
    ///
    /// # Examples
    ///
    /// ```rs
    /// impl_into_rhi!(Vulkan, Instance);
    /// impl_into_rhi!(Vulkan, Buffer<T: BufferLayout>);
    /// ```
    macro_rules! impl_into_rhi {
        ($match:ident, $type:ident $(<$($arg:ident $(: $bound:ident)?),*>)?) => {
            impl<$($($arg $(: $bound)?),*)?> From<$type$(<$($arg),*>)?> for $crate::rhi::$type$(<$($arg),*>)? {
                fn from(o: $type) -> Self {
                    $crate::rhi::$type::$match(o)
                }
            }
        };
    }

    macro_rules! impl_try_from_rhi {
        ($match:ident, $type:ident $(<$($arg:ident $(: $bound:ident)?),*>)?) => {
            impl $(<$($arg $(: $bound)?),*>)? TryFrom <$crate::rhi::$type$(<$($arg),*>)?> for $type$(<$($arg),*>)? {
                type Error = $crate::rhi::BackendError;

                fn try_from(o: $crate::rhi::$type$(<$($arg),*>)?) -> std::result::Result<Self, Self::Error> {
                    match o {
                        $crate::rhi::$type::$match(o) => Ok(o),
                        _ => Err(Self::Error::Mismatch),
                    }
                }
            }
        };
    }

    macro_rules! impl_try_from_rhi_ref {
        ($match:ident, $type:ident $(<$($arg:ident $(: $bound:ident)?),*>)?) => {
            impl <'a, $($($arg $(: $bound)?),*)?> TryFrom <&'a $crate::rhi::$type$(<$($arg),*>)?> for &'a $type$(<$($arg),*>)? {
                type Error = $crate::rhi::BackendError;

                fn try_from(o: &'a $crate::rhi::$type$(<$($arg),*>)?) -> std::result::Result<Self, Self::Error> {
                    match o {
                        $crate::rhi::$type::$match(o) => Ok(o),
                        _ => Err(Self::Error::Mismatch),
                    }
                }
            }
        };
    }

    macro_rules! impl_try_from_rhi_mut {
        ($match:ident, $type:ident $(<$($arg:ident $(: $bound:ident)?),*>)?) => {
            impl <'a, $($($arg $(: $bound)?),*)?> TryFrom <&'a mut $crate::rhi::$type$(<$($arg),*>)?> for &'a mut $type$(<$($arg),*>)? {
                type Error = $crate::rhi::BackendError;

                fn try_from(o: &'a mut $crate::rhi::$type$(<$($arg),*>)?) -> std::result::Result<Self, Self::Error> {
                    match o {
                        $crate::rhi::$type::$match(o) => Ok(o),
                        _ => Err(Self::Error::Mismatch),
                    }
                }
            }
        };
    }

    macro_rules! impl_try_from_rhi_all {
        ($match:ident, $type:ident $(<$($arg:ident $(: $bound:ident)?),*>)?) => {
            $crate::rhi::macros::impl_into_rhi!($match, $type $(<$($arg $(: $bound)?),*>)?);
            $crate::rhi::macros::impl_try_from_rhi!($match, $type $(<$($arg $(: $bound)?),*>)?);
            $crate::rhi::macros::impl_try_from_rhi_ref!($match, $type $(<$($arg $(: $bound)?),*>)?);
            $crate::rhi::macros::impl_try_from_rhi_mut!($match, $type $(<$($arg $(: $bound)?),*>)?);
        };
    }

    pub(crate) use {
        impl_into_rhi, impl_try_from_rhi, impl_try_from_rhi_all, impl_try_from_rhi_mut,
        impl_try_from_rhi_ref,
    };
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    OutOfHostMemory,
    OutOfDeviceMemory,
    TooManyObjects,
    LayerNotPresent,
    ExtensionNotPresent,
    FeatureNotPresent,
    NotSupported,
    Unknown,

    DeviceLost,
    Timeout,

    SurfaceLost,
    SurfaceOutdated,

    Other(Cow<'static, String>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    Mismatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    Vulkan,
    #[cfg(target_os = "windows")]
    DX12,
    #[cfg(target_os = "macos")]
    Metal,
}

pub enum Instance {
    Vulkan(vulkan::Instance),
}

impl Instance {
    /// Creates a new instance. This functions acts as the entrypoint for
    /// interacting with the GPU.
    ///
    /// # Arguments
    ///
    /// # Returns
    /// Returns an instance if the call was successfull. Otherwise returns an
    /// error.
    ///
    /// If the selected backend is not supported on this device
    /// [`Error::NotSupported`] is returned.
    pub fn new(backend: Backend, debug: bool) -> Result<Self, Error> {
        match backend {
            #[cfg(any(target_os = "windows", target_os = "linux"))]
            Backend::Vulkan => vulkan::Instance::new(debug).map(Self::Vulkan),
            #[cfg(target_os = "windows")]
            Backend::DX12 => todo!(),
            #[cfg(target_os = "macos")]
            Backend::Metal => todo!(),
        }
    }

    /// Returns what backend this instance is running.
    pub fn backend(&self) -> Backend {
        match self {
            Self::Vulkan(_) => Backend::Vulkan,
        }
    }

    /// Returns an iterator over all the physical pieces of hardware that
    /// supports the selected backend.
    ///
    /// The instance creation is guaranteed to fail if no adapters are
    /// available which means that the returned iterator will never be empty and
    /// at least contain one element.
    pub fn enumerate_adapters(&self) -> impl Iterator<Item = Adapter> + '_ {
        match &self {
            Self::Vulkan(i) => i.enumerate_adapters().map(Adapter::Vulkan),
        }
    }

    /// Creates a new surface.
    ///
    /// If the window is dropped before the surface or a swapchain created with
    /// this surface, any other failable call may return a
    /// [surface lost](Error::SurfaceLost) error.
    pub fn new_surface(&self, window: &Window) -> Result<Surface, Error> {
        match &self {
            Self::Vulkan(i) => i.new_surface(window).map(Surface::Vulkan),
        }
    }

    /// Creates a new device. The device is the primary interface for creating
    /// objects and otherwise interacting with the GPU.
    ///
    /// # Panics
    ///
    /// This function will panic if both an adapter and a surface is supplied
    /// and the adapter doesn't support presentation to the surface.
    pub fn new_device(&self) -> Result<Device, Error> {
        todo!()
    }

    pub fn new_dswapchain(&self) -> Result<DSwapchain, Error> {
        todo!()
    }
}

pub enum Adapter {
    Vulkan(vulkan::Adapter),
}

pub enum Surface {
    Vulkan(vulkan::Surface),
}

pub enum Device {
    Vulkan(vulkan::Device),
}

impl Device {
    /// Retrieves a queue from the device
    ///
    /// Queues are used for submitting commands to the GPU.
    ///
    /// Queues are not an infinite resource, so the device is allowed to
    /// return `None`, if no more queues of the requested type are
    /// available.
    ///
    /// # Arguments
    ///
    /// - `_operations` - The operations this queue should support.
    ///
    /// # Returns
    ///
    /// Returns the queue if more of the requested type are available. Otherwise
    /// returns None.
    pub fn queue<O>(&self, _operations: O) -> Option<Queue<O>>
    where
        O: OperationsType,
    {
        self.dqueue(O::OPERATIONS).map(|v| Queue(v, PhantomData))
    }

    /// Retrieves a queue from the device where the supported operations of that
    /// queue is not known at compile-time.
    ///
    /// For more information see [`Device::queue`].
    ///
    /// # Arguments
    ///
    /// - `operations` - The operations this queue should support.
    pub fn dqueue(&self, operations: Operations) -> Option<DQueue> {
        match &self {
            Self::Vulkan(d) => d.queue(operations).map(DQueue::Vulkan),
        }
    }

    /// Creates a new command pool.
    ///
    /// The command pool is an allocator that manages all the memory used by
    /// different command lists.
    ///
    /// # Arguments
    ///
    /// - `queue` -
    ///
    /// # Returns
    ///
    /// Returns the newly created command pool if successfull. Otherwise returns
    /// an error.
    pub fn new_command_pool<O>(&self, queue: &Queue<O>) -> Result<CommandPool<O>, Error>
    where
        O: OperationsType,
    {
        let dcommand_pool = self.new_dcommand_pool(queue)?;
        Ok(CommandPool(dcommand_pool, PhantomData))
    }

    /// Creates a new command pool where the supported operations of that
    /// command pool is not known at compile-time.
    ///
    /// For more information see [`Device::new_command_pool`].
    pub fn new_dcommand_pool(&self, queue: &impl AsRef<DQueue>) -> Result<DCommandPool, Error> {
        match &self {
            Self::Vulkan(d) => {
                let queue = queue.as_ref().try_into().unwrap();
                d.new_command_pool(queue).map(DCommandPool::Vulkan)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operations {
    Graphics,
    Compute,
    Transfer,
}

pub mod operations {
    use super::Operations;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct OperationsError {
        pub expected: Operations,
        pub found: Operations,
    }

    pub trait OperationsType {
        const OPERATIONS: Operations;
    }

    /// Supports all operations.
    pub struct Graphics;
    impl OperationsType for Graphics {
        const OPERATIONS: Operations = Operations::Graphics;
    }

    /// Supports compute and transfer operations.
    pub struct Compute;
    impl OperationsType for Compute {
        const OPERATIONS: Operations = Operations::Compute;
    }

    /// Supports transfer operations.
    pub struct Transfer;
    impl OperationsType for Transfer {
        const OPERATIONS: Operations = Operations::Transfer;
    }
}

pub enum DQueue {
    Vulkan(vulkan::DQueue),
}

impl DQueue {
    pub fn operations(&self) -> Operations {
        match &self {
            Self::Vulkan(q) => q.operations(),
        }
    }

    pub fn submit(&mut self) -> GpuFuture<()> {
        todo!()
    }
}

pub struct Queue<O: OperationsType>(DQueue, PhantomData<O>);

impl<O: OperationsType> Queue<O> {
    /// Creates a new queue where the supported operations are encoded in the
    /// type.
    pub fn new(dqueue: DQueue) -> Result<Self, OperationsError> {
        let operations = dqueue.operations();
        if operations == O::OPERATIONS {
            Ok(unsafe { Self::new_unchecked(dqueue) })
        } else {
            Err(OperationsError {
                expected: O::OPERATIONS,
                found: operations,
            })
        }
    }

    /// Creates a new queue that is guaranteed to support `O` operations.
    ///
    /// For the safe variant see [`Self::new`].
    ///
    /// # Safety
    /// Undefined behaviour will occur if `dqueue` doesn't support `O`
    /// operations.
    pub unsafe fn new_unchecked(dqueue: DQueue) -> Self {
        Self(dqueue, PhantomData)
    }
}

impl<O: OperationsType> AsRef<DQueue> for Queue<O> {
    fn as_ref(&self) -> &DQueue {
        &self.0
    }
}

impl<O: OperationsType> AsMut<DQueue> for Queue<O> {
    fn as_mut(&mut self) -> &mut DQueue {
        &mut self.0
    }
}

pub enum DCommandPool {
    Vulkan(vulkan::DCommandPool),
}

pub struct CommandPool<O: OperationsType>(DCommandPool, PhantomData<O>);

pub enum DCommandList {
    Vulkan(vulkan::DCommandList),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {}

pub enum DSwapchain {}
