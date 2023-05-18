//! Rendering Hardware Interface (RHI)
//!
//! # Using Multiple Backends
//!
//! Mixing of objects that are not using the same [backend][Backend] will cause
//! the program to panic with a [`BackendError`]. For example supplying an
//! [adapter][Adapter] which is using [`Backend::DX12`] to a
//! [instance][Instance] that is running [vulkan][`Backend::Vulkan`] will cause
//! one of these panics.
//!
//! This behaviour is not documented under a panics section since it
//! applies to all functions/methods that takes any of the backend-agnostic
//! objects as an argument.

use std::borrow::Cow;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::ops::{Range, RangeInclusive};
use std::time::Duration;

use ::futures::{Future, FutureExt};
use ::glam::*;

use self::operations::{OperationsError, OperationsType};
use self::sync::GpuFuture;
use self::vulkan::{Shader, ShaderStage};
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
    /// impl_into_rhi!(Vulkan, Pipeline<'a>);
    /// ```
    macro_rules! impl_into_rhi {
        ($match:ident, $type:ident $(<$($arg:tt $(: $bound:tt)?),*>)?) => {
            impl<$($($arg $(: $bound)?),*)?> From<$type$(<$($arg),*>)?> for $crate::rhi::$type$(<$($arg),*>)? {
                fn from(o: $type$(<$($arg),*>)?) -> Self {
                    $crate::rhi::$type::$match(o)
                }
            }
        };
    }

    /// Implements [`TryFrom`] and [`TryInto`] for converting into an
    /// API-specific object, from the common object.
    ///
    /// # Examples
    ///
    /// ```rs
    /// impl_into_rhi!(Vulkan, Instance);
    /// impl_into_rhi!(Vulkan, Buffer<T: BufferLayout>);
    /// impl_into_rhi!(Vulkan, Pipeline<'a>);
    /// ```
    macro_rules! impl_try_from_rhi {
        ($match:ident, $type:ident $(<$($arg:tt $(: $bound:tt)?),*>)?) => {
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

    /// Implements [`TryFrom`] and [`TryInto`] for converting into a reference
    /// of a API-specific object, from a reference of the common object.
    ///
    /// # Examples
    ///
    /// ```rs
    /// impl_try_from_rhi_ref!(Vulkan, Instance);
    /// impl_try_from_rhi_ref!(Vulkan, Buffer<T: BufferLayout>);
    /// impl_try_from_rhi_ref!(Vulkan, Pipeline<'a>);
    /// ```
    macro_rules! impl_try_from_rhi_ref {
        ($match:ident, $type:ident $(<$($arg:tt $(: $bound:tt)?),*>)?) => {
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

    /// Implements [`TryFrom`] and [`TryInto`] for converting into a mutable
    /// reference of a API-specific object, from a mutable reference of the
    /// common object.
    ///
    /// # Examples
    ///
    /// ```rs
    /// impl_try_from_rhi_mut!(Vulkan, Instance);
    /// impl_try_from_rhi_mut!(Vulkan, Buffer<T: BufferLayout>);
    /// impl_try_from_rhi_mut!(Vulkan, Pipeline<'a>);
    /// ```
    macro_rules! impl_try_from_rhi_mut {
        ($match:ident, $type:ident $(<$($arg:tt $(: $bound:tt)?),*>)?) => {
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
        ($match:ident, $type:ident $(<$($arg:tt $(: $bound:tt)?),*>)?) => {
            $crate::rhi::macros::impl_into_rhi!($match, $type $(<$($arg $(: $bound)?),*>)?);
            $crate::rhi::macros::impl_try_from_rhi!($match, $type $(<$($arg $(: $bound)?),*>)?);
            $crate::rhi::macros::impl_try_from_rhi_ref!($match, $type $(<$($arg $(: $bound)?),*>)?);
            $crate::rhi::macros::impl_try_from_rhi_mut!($match, $type $(<$($arg $(: $bound)?),*>)?);
        };
    }

    pub(super) use {
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
    /// If the selected backend is not supported on this hardware
    /// [`Error::NotSupported`] is returned.
    pub fn new(backend: Backend, debug: bool) -> Result<Self, Error> {
        match backend {
            #[cfg(any(target_os = "windows", target_os = "linux"))]
            Backend::Vulkan => vulkan::Instance::new(debug).map(Self::Vulkan),
            #[cfg(target_os = "windows")]
            Backend::DX12 => todo!(),
            #[cfg(target_os = "macos")]
            Backend::Metal => todo!(),
            _ => Err(Error::NotSupported),
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
    /// - This function will panic if both an adapter and a surface is supplied
    /// and the adapter doesn't support presentation to the surface.
    ///
    /// - Panics if [`DeviceProps::adapter`](DeviceProps) or
    ///   [`DeviceProps::surface`] in [DeviceProps] is not using the same
    ///   backend as the
    /// instance.
    pub fn new_device(&self, mut props: DeviceProps) -> Result<Device, Error> {
        match &self {
            Self::Vulkan(i) => {
                let surface = props.surface.take().map(|e| e.try_into().unwrap());
                let adapter = props.adapter.take().map(|e| e.try_into().unwrap());
                i.new_device(surface, adapter, props).map(Into::into)
            }
        }
    }

    /// Creates a new swapchain where the format is not known at compile time.
    ///
    /// The size of the swapchain images are the current size of `surface`.
    pub fn new_dswapchain(&self, device: &Device, surface: Surface) -> Result<DSwapchain, Error> {
        match &self {
            Self::Vulkan(i) => {
                let device = device.try_into().unwrap();
                let surface = surface.try_into().unwrap();
                i.new_swapchain(device, surface).map(Into::into)
            }
        }
    }
}

pub enum Adapter {
    Vulkan(vulkan::Adapter),
}

pub enum Surface {
    Vulkan(vulkan::Surface),
}

#[derive(Default)]
pub struct DeviceProps<'a> {
    pub adapter: Option<Adapter>,
    pub surface: Option<&'a Surface>,
    pub graphics_queues: Option<Range<usize>>,
    pub compute_queues: Option<Range<usize>>,
    pub transfer_queues: Option<Range<usize>>,
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
        let dqueue = self.dqueue(O::OPERATIONS)?;
        Some(Queue(dqueue, PhantomData))
    }

    /// Retrieves a queue from the device where the supported operations of that
    /// queue is not known at compile-time.
    ///
    /// For more information see [Device::queue].
    ///
    /// # Arguments
    ///
    /// - `operations` - The operations this queue should support.
    pub fn dqueue(&self, operations: Operations) -> Option<DQueue> {
        match &self {
            Self::Vulkan(device) => device.queue(operations).map(Into::into),
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
            Self::Vulkan(device) => {
                let queue = queue.as_ref().try_into().unwrap();
                device.new_command_pool(queue).map(Into::into)
            }
        }
    }

    pub fn new_command_list<O>(&self, pool: &mut CommandPool<O>) -> Result<CommandList<O>, Error>
    where
        O: OperationsType,
    {
        let dcommand_list = self.new_dcommand_list(pool.as_mut())?;
        Ok(CommandList(dcommand_list, PhantomData))
    }

    pub fn new_dcommand_list<P>(&self, pool: &mut P) -> Result<DCommandList, Error>
    where
        P: AsMut<DCommandPool>,
    {
        match self {
            Self::Vulkan(device) => {
                let pool = pool.as_mut().try_into().unwrap();
                device.new_command_list(pool).map(Into::into)
            }
        }
    }

    /// Creates a new semaphore that is initialized with `value`.
    pub fn new_semaphore(&self, value: u64) -> Result<Semaphore, Error> {
        match &self {
            Self::Vulkan(device) => device.new_semaphore(value).map(Into::into),
        }
    }

    pub fn new_fence(&self, signaled: bool) -> Result<Fence, Error> {
        match self {
            Self::Vulkan(device) => device.new_fence(signaled).map(Into::into),
        }
    }

    /// Creates a new render pass.
    pub fn new_render_pass(&self, attachments: &[Attachment]) -> Result<RenderPass, Error> {
        match self {
            Self::Vulkan(device) => device.new_render_pass(attachments).map(Into::into),
        }
    }

    pub fn new_framebuffer(
        &self,
        render_pass: &RenderPass,
        attachments: &[DImageView2D],
        extent: UVec2,
    ) -> Result<Framebuffer, Error> {
        match self {
            Self::Vulkan(device) => {
                let render_pass = render_pass.try_into().unwrap();
                let attachments = attachments.iter().map(|a| a.try_into().unwrap());

                device
                    .new_framebuffer(render_pass, attachments, extent)
                    .map(Into::into)
            }
        }
    }

    pub fn new_shader(&self, bytecode: &[u8]) -> Result<Shader, Error> {
        match self {
            Self::Vulkan(device) => device.new_shader(bytecode).map(Into::into),
        }
    }

    /// Creates a new pipeline
    pub fn new_pipeline(
        &self,
        state: &PipelineState,
        shaders: &[(Shader, ShaderStage)],
        render_pass: &RenderPass,
    ) -> Result<Pipeline, Error> {
        match self {
            Self::Vulkan(device) => {
                let render_pass = render_pass.try_into().unwrap();
                device
                    .new_pipeline(state, shaders, render_pass)
                    .map(Into::into)
            }
        }
    }

    /// Blocks on the current thread until the device has completed all pending
    /// work
    pub fn wait_idle(&self) -> Result<(), Error> {
        match self {
            Self::Vulkan(device) => device.wait_idle(),
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

pub struct SubmitInfo<'a> {
    pub command_lists: Vec<&'a DCommandList>,
    pub wait_semaphores: Vec<(&'a Semaphore, u64)>,
    pub signal_semaphores: Vec<(&'a Semaphore, u64)>,
}

pub enum DQueue {
    Vulkan(vulkan::DQueue),
}

impl DQueue {
    /// Returns the operations supported by this queue.
    pub fn operations(&self) -> Operations {
        match &self {
            Self::Vulkan(q) => q.operations(),
        }
    }

    /// Submits commands to the GPU.
    ///
    /// # Safety
    ///
    /// It is undefined behaviour to drop any of the the command lists before
    /// the device is finished executing them.
    ///
    /// # Panics
    ///
    /// This function panics if one of the `infos` doesn't have any command
    /// lists attached with it.
    ///
    /// This function panics if any of the command lists belong to another
    /// queue.
    // TODO Measure overhead of type conversions. Maybe use iterators with generics
    // instead.
    pub unsafe fn submit_unchecked<'a>(
        &mut self,
        infos: impl Iterator<Item = SubmitInfo<'a>>,
        fence: Option<&mut Fence>,
    ) -> Result<(), Error> {
        match self {
            Self::Vulkan(queue) => {
                let infos: Vec<_> = infos
                    .into_iter()
                    .map(TryInto::try_into)
                    .collect::<Result<_, _>>()
                    .unwrap();

                let fence = fence.map(TryInto::try_into).transpose().unwrap();
                queue.submit_unchecked(infos.as_slice(), fence)
            }
        }
    }

    /// Blocks on the calling-thread until the queue has completed all pending
    /// work.
    pub fn wait_idle(&mut self) -> Result<(), Error> {
        match self {
            Self::Vulkan(queue) => queue.wait_idle(),
        }
    }
}

impl AsRef<DQueue> for DQueue {
    fn as_ref(&self) -> &DQueue {
        self
    }
}

impl AsMut<DQueue> for DQueue {
    fn as_mut(&mut self) -> &mut DQueue {
        self
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
    /// It is undefined behaviour if `dqueue` doesn't support `O`
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

impl AsMut<DCommandPool> for DCommandPool {
    fn as_mut(&mut self) -> &mut DCommandPool {
        self
    }
}

pub struct CommandPool<O: OperationsType>(DCommandPool, PhantomData<O>);

impl<O: OperationsType> AsMut<DCommandPool> for CommandPool<O> {
    fn as_mut(&mut self) -> &mut DCommandPool {
        &mut self.0
    }
}

pub enum DCommandList {
    Vulkan(vulkan::DCommandList),
}

impl DCommandList {
    /// Begins recording on this command list.
    ///
    /// This sets the internal state of the command list to recording.
    ///
    /// # Safety
    ///
    /// This command list must be in the initial state.
    ///
    /// Only a single command list in a pool can be recording at any given
    /// time. This means that until `end_unchecked` is called, `command_pool`
    /// must not begin recording for any of its
    /// other command lists.
    ///
    /// # Panics
    ///
    /// Panics if `command_pool` is not the command pool as the one that was
    /// passed in when creating this command list initially.
    pub unsafe fn begin_unchecked(&mut self, command_pool: &mut DCommandPool) -> Result<(), Error> {
        match self {
            Self::Vulkan(command_list) => {
                let command_pool: &mut _ = command_pool.try_into().unwrap();
                command_list.begin_unchecked(command_pool)
            }
        }
    }

    /// Ends recording for this command list.
    ///
    /// After this has been called it is again safe to use the command pool for
    /// recording another command list.
    ///
    ///  # Safety
    ///
    /// It is undefined behaviour if this command list is not in the recording
    /// state.
    pub unsafe fn end_unchecked(&mut self) -> Result<(), Error> {
        match self {
            Self::Vulkan(command_list) => command_list.end_unchecked(),
        }
    }

    /// Begins a render pass on this command list.
    ///
    /// # Safety
    ///
    /// It is undefined behaviour if any of the following are broken:
    /// - This command list must support graphics operations and be in the
    ///   recording state.
    /// - No other render passes are currently recording meaning you can't call
    ///   `begin_render_pass_unchecked` two times in a row without calling
    ///   `end_render_pass_unchecked` in between.
    pub unsafe fn begin_render_pass_unchecked(
        &mut self,
        render_pass: &RenderPass,
        framebuffer: &mut Framebuffer,
    ) {
        match self {
            Self::Vulkan(command_list) => {
                let render_pass = render_pass.try_into().unwrap();
                let framebuffer = framebuffer.try_into().unwrap();
                command_list.begin_render_pass_unchecked(render_pass, framebuffer);
            }
        }
    }

    /// Ends a render pass on this command list.
    ///
    /// After calling this method it is again safe to call
    /// `begin_render_pass_unchecked`.
    ///
    /// # Safety
    ///
    /// It is undefined behaviour if any of the following are broken:
    /// - This command list must support graphics operations and be in the
    ///   recording state.
    /// - This command list is not currently recording a render pass.
    pub unsafe fn end_render_pass_unchecked(&mut self) {
        match self {
            Self::Vulkan(command_list) => {
                command_list.end_render_pass_unchecked();
            }
        }
    }

    /// Binds a pipeline to this command list.
    ///
    ///
    /// # Safety
    ///
    /// If `pipeline` is a graphics pipeline, this command list must be
    /// currently recording a render pass
    ///
    /// If `pipeline` is a compute pipeline, this command list must be currently
    /// recording a compute pass.
    pub unsafe fn bind_pipeline_unchecked(&mut self, pipeline: &Pipeline) {
        match self {
            Self::Vulkan(command_list) => {
                let pipeline = pipeline.try_into().unwrap();
                command_list.bind_pipeline_unchecked(pipeline)
            }
        }
    }

    /// Sets the viewport state
    ///
    /// # Panics
    ///
    /// - Panics if the pipeline doesn't support updating the viewport state
    ///   dynamically.
    ///
    /// # Safety
    ///
    /// A pipeline must have been bound
    pub unsafe fn set_viewport_unchecked(&mut self, viewport: &ViewportState) {
        match self {
            Self::Vulkan(command_list) => command_list.set_viewport_unchecked(&viewport),
        }
    }

    /// Draw
    ///
    /// # Safety
    ///
    /// This command list must be recording a render pass.
    pub unsafe fn draw_unchecked(&mut self, vertices: usize, instances: usize) {
        match self {
            Self::Vulkan(command_list) => command_list.draw_unchecked(vertices, instances),
        }
    }
}

pub struct CommandList<O: OperationsType>(DCommandList, PhantomData<O>);

impl<O: OperationsType> AsMut<DCommandList> for CommandList<O> {
    fn as_mut(&mut self) -> &mut DCommandList {
        &mut self.0
    }
}

pub trait SemaphoreApi: Send + Sync {}

/// Semaphores are used for synchronizing primarily between device queues, but
/// also provides mechanisms for blocking on the CPU.
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

/// **NOTE** Dropping a fence that is still in use by the GPU will cause the
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

pub enum DImage2D
where
    Self: Send + Sync,
{
    Vulkan(vulkan::DImage2D),
}

#[derive(Debug, Clone)]
pub enum DImageView2D {
    Vulkan(vulkan::DImageView2D),
}

/// A render pass is an object that specifiies how multiple different
/// attachments will be used during rendering.
pub enum RenderPass {
    Vulkan(vulkan::RenderPass),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    R8G8B8A8Unorm,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum AttachmentLoadOp {
    Load,
    Clear,
    #[default]
    DontCare,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum AttachmentStoreOp {
    Store,
    #[default]
    DontCare,
}

pub enum Layout {
    Undefined,
    Preinitialized,
    General,
    Present,
}

pub enum Attachment {
    Color {
        format: Format,
        load_op: AttachmentLoadOp,
        store_op: AttachmentStoreOp,
        layout: Layout,
    },
    DepthStencil {
        format: Format,
        depth_load_op: Option<AttachmentLoadOp>,
        depth_store_op: Option<AttachmentStoreOp>,
        stencil_load_op: Option<AttachmentLoadOp>,
        stencil_store_op: Option<AttachmentStoreOp>,
    },
}

pub enum Framebuffer {
    Vulkan(vulkan::Framebuffer),
}

pub struct ScissorState {
    pub offset: UVec2,
    pub extent: UVec2,
}

pub struct ViewportState {
    pub position: Vec2,
    pub extent: Vec2,
    pub depth: RangeInclusive<f32>,
    pub scissor: Option<ScissorState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Samples(usize);

impl Samples {
    pub const ONE: Samples = Samples(1);
    pub const TWO: Samples = Samples(2);
    pub const FOUR: Samples = Samples(4);

    pub fn new(count: usize) -> Option<Samples> {
        match count {
            1 | 2 | 4 | 16 | 32 | 64 => Some(Self(count)),
            _ => None,
        }
    }
}

impl Default for Samples {
    fn default() -> Self {
        Self::ONE
    }
}

pub struct MultisampleState {
    pub samples: Samples,
}

pub struct PipelineState {
    pub viewport: Option<ViewportState>,
    pub multisample: MultisampleState,
}

pub struct PipelineProps {}

pub enum Pipeline {
    Vulkan(vulkan::Pipeline),
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

    /// Acquires an image from the swapchain.
    pub fn image(&self, fence: Fence) -> GpuFuture<View<DImage2D>> {
        todo!()
    }

    pub fn image_mut(&self, fence: Fence) -> GpuFuture<ViewMut<DImage2D>> {
        todo!()
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
    pub unsafe fn image_unchecked(
        &mut self,
        fence: &mut Fence,
        timeout: Duration,
    ) -> Result<Option<DImageView2D>, Error> {
        match self {
            Self::Vulkan(swapchain) => {
                let fence = fence.try_into().unwrap();
                let image = swapchain.image_unchecked(fence, timeout);
                image.map(|image| image.map(Into::into))
            }
        }
    }

    /// Enumerates all the images in the swapchain.
    pub fn enumerate_images(&mut self) -> impl ExactSizeIterator<Item = &DImage2D> + '_ {
        const V: &[DImage2D] = &[];
        V.iter()
    }

    pub fn present(&mut self, image_view: &DImageView2D) -> Result<(), Error> {
        match self {
            Self::Vulkan(swapchain) => {
                let image_view = image_view.try_into().unwrap();
                swapchain.present(image_view)
            }
        }
    }
}

pub struct View<T> {
    _marker: PhantomData<T>,
}

pub struct ViewMut<T> {
    _marker: PhantomData<T>,
}
