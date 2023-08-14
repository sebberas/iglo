//! Rendering Hardware Interface (RHI)
//!
//! # Using Multiple Backends
//!
//! Mixing of objects that are not using the same [backend][Backend] will cause
//! the program to panic with a [`BackendError`]. As an example supplying an
//! [adapter][Adapter] which is using [`Backend::DX12`] to a
//! [instance][Instance] that is running [vulkan][`Backend::Vulkan`] will cause
//! one of these panics.
//!
//! This behaviour is not documented under a separate panics section for the
//! individual items since it applies to all functions/methods that takes any of
//! the backend-agnostic objects as an argument.
//!
//! # Example
//!
//! ```
//! let opengl = Instance::new(Backend::OpenGL_460).unwrap();
//! let adapter = opengl.emumerate_adapters().next().unwrap();
//!
//! let vulkan = Instance::new(Backend::Vulkan).unwrap();
//!
//!  // Panic with BackendError::Mismatch
//! let device = vulkan.new_device(DeviceProps {
//!     adapter: Some(&adapter),
//!     ..Default::default()
//! });
//! ```

use std::borrow::Cow;
use std::marker::PhantomData;
use std::ops::{Range, RangeInclusive};
use std::time::Duration;

use ::glam::*;

pub use self::backend::*;
pub use self::queue::*;
pub use self::resources::*;
pub use self::swapchain::*;
use self::vulkan::{Shader, ShaderStage}; // TEMPORARY
use crate::os::Window;
use crate::rhi;

pub mod spirv;
pub mod sync;

mod backend;
mod queue;
mod resources;
mod swapchain;

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
    pub fn new_dswapchain(&self, props: SwapchainProps) -> Result<DSwapchain, Error> {
        match &self {
            Self::Vulkan(instance) => {
                let props = SwapchainProps {
                    device: props.device.try_into().unwrap(),
                    surface: props.surface.try_into().unwrap(),
                    images: props.images,
                    image_format: props.image_format,
                    image_extent: props.image_extent,
                    present_mode: props.present_mode,
                };

                instance.new_swapchain(props).map(Into::into)
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
    ///
    /// # Examples
    ///
    /// ```
    /// // Get a graphics queue with the type known at compile-time.
    /// let queue = device.queue(ops::Graphics).unwrap();
    /// ```
    pub fn queue<O>(&self, _operations: O) -> Option<Queue<O>>
    where
        O: OperationsType,
    {
        let dqueue = self.dqueue(O::OPERATIONS)?;
        Some(unsafe { Queue::new_unchecked(dqueue) })
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
        let command_pool = self.new_dcommand_pool(queue)?;
        Ok(unsafe { CommandPool::new_unchecked(command_pool) })
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

    pub fn new_command_list<O>(
        &self,
        command_pool: &mut CommandPool<O>,
    ) -> Result<CommandList<O>, Error>
    where
        O: OperationsType,
    {
        let command_list = self.new_dcommand_list(command_pool.as_mut())?;
        Ok(unsafe { CommandList::new_unchecked(command_list) })
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

    pub fn new_binary_semaphore(&self) -> Result<BinarySemaphore, Error> {
        match &self {
            Self::Vulkan(device) => device.new_binary_semaphore().map(Into::into),
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

    pub fn new_descriptor_pool(&self) -> Result<DescriptorPool, Error> {
        match self {
            Self::Vulkan(device) => device.new_descriptor_pool().map(Into::into),
        }
    }

    // pub fn new_ddescriptor(&self) -> Result<DDescriptor, Error> {}

    pub fn new_dbuffer(&self, props: &DBufferProps) -> Result<DBuffer, Error> {
        match self {
            Self::Vulkan(device) => device.new_buffer(props).map(Into::into),
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

pub mod ops {
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

pub enum SemaphoreSubmitInfo<'a, S = rhi::Semaphore, B = rhi::BinarySemaphore> {
    Default(&'a S, u64),
    Binary(&'a B),
}

pub struct SubmitInfo<
    'a,
    C = rhi::DCommandList,
    S = rhi::Semaphore,
    P = rhi::PipelineStage,
    B = rhi::BinarySemaphore,
> {
    pub command_lists: Vec<&'a C>,
    pub wait_semaphores: Vec<(SemaphoreSubmitInfo<'a, S, B>, P)>,
    pub signal_semaphores: Vec<SemaphoreSubmitInfo<'a, S, B>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    Transfer,
    ColorAttachmentOutput,
}

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
    R8G8B8A8Srgb,

    R32G32B32A32Float,
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

pub enum VertexInputRate {
    Vertex,
    Instance,
}

pub struct VertexInputBinding {
    pub binding: usize,
    pub stride: usize,
    pub rate: VertexInputRate,
}

pub struct VertexInputAttribute {
    pub location: usize,
    pub binding: usize,
    pub format: Format,
    pub offset: usize,
}

pub struct VertexInputState {
    pub bindings: Vec<VertexInputBinding>,
    pub attributes: Vec<VertexInputAttribute>,
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
    pub vertex_input: Option<VertexInputState>,
    pub viewport: Option<ViewportState>,
    pub multisample: MultisampleState,
}

pub struct PipelineProps {}

pub enum Pipeline {
    Vulkan(vulkan::Pipeline),
}

pub enum DescriptorPool {
    Vulkan(vulkan::DescriptorPool),
}

pub enum DescriptorSet {
    Vulkan(vulkan::DescriptorSet),
}

pub struct View<T> {
    _marker: PhantomData<T>,
}

pub struct ViewMut<T> {
    _marker: PhantomData<T>,
}
