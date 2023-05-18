//!
//! Instance
//! Surface
//! Adapter
//! Device
//! Swapchain
//! CommandQueue <Graphics | Compute | Transfer>
//! CommandPool  <Graphics | Compute | Transfer>
//! CommandList  <Graphics | Compute | Transfer>
//! RenderTarget
//! RenderPass
//! ComputePass
//! RenderPipeline
//! ComputePipeline
//! Memory
//! Buffer + BufferView
//! Image  + ImageView
//! Shader
//! DescriptorPool
//! DescriptorSet
//! Descriptor
//!
//! Fence
//! Barrier
//! Semaphore
//!
//! Raytracing

use std::borrow::Cow;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::time::Duration;

pub use self::format::Format;
use self::format::FormatType;
use self::queue::{Graphics, QueueKind};
use self::state::{Initial, Recording, State};
use self::usage::{BufferUsage, ImageUsage};
use crate::os::Window;

pub mod sync;

pub mod dx12;
pub mod vulkan;

pub mod spirv;

mod macros {
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

pub type Result<T> = std::result::Result<T, Error>;

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
    DX12,
    Vulkan,
}

pub enum Instance {
    DX12(dx12::Instance),
    Vulkan(vulkan::Instance),
}

impl Instance {
    pub fn new(backend: Backend, debug: bool) -> Result<Self> {
        match backend {
            // Backend::DX12 => dx12::Instance::new(debug).map(Self::DX12),
            Backend::Vulkan => vulkan::Instance::new(debug).map(Self::Vulkan),
            _ => unimplemented!(),
        }
    }

    /// Creates a new surface
    ///
    /// # Safety
    ///
    /// `window` must not be destroyed while the surface or swapchain exists. If
    /// the window is destroyed any function can fail with
    /// [`Error::SurfaceLost`]
    pub unsafe fn new_surface(&self, window: *const Window) -> Result<Surface> {
        match self {
            Self::DX12(instance) => instance.new_surface(window).map(Surface::DX12),
        }
    }

    /// Creates a new device
    ///
    /// The device is the primary interface for interacting with the GPU.
    pub fn new_device(&self, props: &DeviceProps) -> Result<Device> {
        match self {
            Self::DX12(i) => {
                let surface = props.surface.map(TryInto::try_into).transpose();
                i.new_device(surface.unwrap(), props).map(Device::DX12)
            }
        }
    }

    /// Creates a new swapchain with the specified format
    ///
    /// # Panics
    ///
    /// - Panics if the format is [`Format::Unknown`](Format)
    /// - Panics if [`SwapchainProps::width`] or [`SwapchainProps::height`] are
    /// larger than the size allowed by the API.
    pub fn new_swapchain<'a, P, F>(&self, props: P) -> Result<Swapchain<F>>
    where
        P: Into<SwapchainProps<'a, F>>,
        F: FormatType,
    {
        let props = props.into();
        match self {
            Self::DX12(i) => {
                let device = props.device.try_into().unwrap();
                let surface = props.surface.try_into().unwrap();
                i.new_swapchain(device, surface).map(Swapchain::DX12)
            }
            Self::Vulkan(i) => {
                let device = props.device.try_into().unwrap();
                let surface = props.surface.try_into().unwrap();
                i.new_swapchain(device, surface).map(Swapchain::Vulkan)
            }
        }
    }
}

pub enum Surface {
    DX12(dx12::Surface),
    Vulkan(vulkan::Surface),
}

#[derive(Default, Clone, Copy)]
pub struct DeviceProps<'a> {
    pub surface: Option<&'a Surface>,
    pub max_graphics_queues: Option<NonZeroUsize>,
    pub max_compute_queues: Option<NonZeroUsize>,
    pub max_transfer_queues: Option<NonZeroUsize>,
}

pub enum Device {
    DX12(dx12::Device),
}

impl Device {
    /// Creates a new queue that can be used for asynchronous execution on the
    /// GPU.
    ///
    /// Three different kinds of queues are available:
    ///
    /// **Graphics** - Which is able to execute all commands, graphics specific
    /// commands included.
    ///
    /// **Compute** - Which is able to execute compute
    /// and transfer commands.
    ///
    /// **Transfer** - Which is able to execute
    /// transfer commands.
    ///
    /// It is recommended to use a queue with the least amount of supported
    /// commands since the driver is able to optimize it
    /// better by parallelizing execution of different queue types. For
    /// a more in-depth explanation of these concepts see ...
    ///
    /// # Arguments
    ///
    /// - `kind` - The kind of queue to create.
    ///
    /// # Returns
    ///
    /// Queues are not an infinite resource, so the device is allowed to return
    /// `None`, if no more queues of the requested type are available, but the
    /// call was otherwise successfull.
    pub fn new_command_queue<K>(&self, kind: K) -> Result<Option<CommandQueue<K>>>
    where
        K: QueueKind,
    {
        match self {
            Self::DX12(d) => d.new_command_queue(kind).map(|q| q.map(CommandQueue::DX12)),
        }
    }

    /// Creates a new command pool linked with a supplied command queue.
    ///
    /// The command pool is an allocator that manages all the memory used by
    /// different command lists.
    pub fn new_command_pool<K>(&self, queue: &CommandQueue<K>) -> Result<CommandPool<K>>
    where
        K: QueueKind,
    {
        match self {
            Self::DX12(d) => {
                let queue = queue.try_into();
                d.new_command_pool(queue.unwrap()).map(CommandPool::DX12)
            }
        }
    }

    /// Creates a new command list.
    ///
    /// A command list is created in the initial state. The same command pool
    /// used when creating the command list, must be used for all further
    /// operations where a pool is needed.
    ///
    /// # Arguments
    ///
    /// - `pool` - The pool that should be backing this command list.
    pub fn new_command_list<K>(&self, pool: &mut CommandPool<K>) -> Result<CommandList<K, Initial>>
    where
        K: QueueKind,
    {
        match self {
            Self::DX12(d) => {
                let pool = pool.try_into();
                d.new_command_list(pool.unwrap()).map(CommandList::DX12)
            }
        }
    }

    /// Creates a new buffer
    ///
    /// If `props.memory` contains a block of memory, the buffer is
    /// created and assigned to that block. This mean the field always contains
    /// `None` after the function has returned.
    pub fn new_buffer<T, U>(&self, p: &mut impl AsMut<BufferProps<T, U>>) -> Result<Buffer<T, U>>
    where
        T: BufferLayout,
        U: BufferUsage,
    {
        match self {
            Self::DX12(d) => d.new_buffer().map(Buffer::DX12),
        }
    }

    /// Creates a new image
    ///
    /// If `create_info.memory` contains a block of memory, the image is
    /// created and assigned to that passed-in block. `create_info.memory` is
    /// being and left empty after the function has returned.
    /// In the case of no memory supplied, it is simply created without a memory
    /// block backing the image. Before you read/write to an image, it
    /// must have been assigned some memory.
    ///
    /// # Panics
    ///
    /// - Panics if the memory in `create_info` is incompatible with the
    /// type of image being created.
    ///
    /// - Panics if [`ImageProps::width`] or [`ImageProps::height`] are
    /// larger than the size allowed by the API. These constraints should be
    /// queried at runtime using [`Device::limits`].
    ///
    /// - Panics if the image is being created with an unknown format.
    ///
    /// # Returns
    ///
    /// Returns a new image with the specified format and usage if the operation
    /// is successfull. Otherwise it returns an error.
    pub fn new_image<F, U>(&self, props: &mut impl AsMut<ImageProps<F, U>>) -> Result<Image<F, U>>
    where
        F: FormatType,
        U: ImageUsage,
    {
        match self {
            Self::DX12(d) => d.new_image(props.as_mut()).map(Image::DX12),
        }
    }
}

pub struct SwapchainProps<'a, F: FormatType> {
    pub device: &'a Device,
    pub surface: Surface,
    pub width: Option<NonZeroUsize>,
    pub height: Option<NonZeroUsize>,
    pub backbuffers: NonZeroUsize,
    pub format: F,
}

impl<'a, F: FormatType> SwapchainProps<'a, F> {
    pub fn new(device: &'a Device, surface: Surface, format: F) -> Self {
        Self {
            device,
            surface,
            width: None,
            height: None,
            backbuffers: unsafe { NonZeroUsize::new_unchecked(2) },
            format,
        }
    }
}

impl<'a> SwapchainProps<'a, format::Unknown> {
    pub fn builder(device: &'a Device, surface: Surface) -> SwapchainPropsBuilder<format::Unknown> {
        SwapchainPropsBuilder(Self::new(device, surface, format::Unknown))
    }
}

impl<'a, F: FormatType> AsRef<SwapchainProps<'a, F>> for SwapchainProps<'a, F> {
    fn as_ref(&self) -> &SwapchainProps<'a, F> {
        self
    }
}

pub struct SwapchainPropsBuilder<'a, F: FormatType>(SwapchainProps<'a, F>);

impl<'a, F: FormatType> SwapchainPropsBuilder<'a, F> {
    pub fn width(mut self, width: NonZeroUsize) -> Self {
        self.0.width = Some(width);
        self
    }

    pub fn height(mut self, height: NonZeroUsize) -> Self {
        self.0.height = Some(height);
        self
    }

    pub fn backbuffers(mut self, n: NonZeroUsize) -> Self {
        self.0.backbuffers = n;
        self
    }

    pub fn format<F2: FormatType>(self, format: F2) -> SwapchainPropsBuilder<'a, F2> {
        SwapchainPropsBuilder(SwapchainProps {
            device: self.0.device,
            surface: self.0.surface,
            width: self.0.width,
            height: self.0.height,
            backbuffers: self.0.backbuffers,
            format,
        })
    }

    pub fn build(self) -> SwapchainProps<'a, F> {
        self.0
    }
}

impl<'a, F: FormatType> AsRef<SwapchainProps<'a, F>> for SwapchainPropsBuilder<'a, F> {
    fn as_ref(&self) -> &SwapchainProps<'a, F> {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fullscreen {
    Fullscreen,
    Windowed,
}

pub enum Swapchain<F: FormatType> {
    DX12(dx12::Swapchain<F>),
    Vulkan(vulkan::Swapchain),
}

impl<F: FormatType> Swapchain<F> {
    pub fn set_fullscreen(&mut self, state: Fullscreen) -> Result<()> {
        match self {
            Self::DX12(s) => s.set_fullscreen(state),
        }
    }

    pub fn image(&mut self, timeout: Duration) -> Result<Option<Image<F, usage::RenderTarget>>> {
        match self {
            Self::DX12(s) => s.image(timeout).map(|i| i.map(Image::DX12)),
        }
    }

    pub fn present(&mut self) -> Result<()> {
        match self {
            Self::DX12(s) => s.present(),
        }
    }
}

// TODO: Rename to commands (?).
pub mod queue {
    pub enum QueueType {
        Graphics,
        Compute,
        Transfer,
    }

    pub trait QueueKind {
        const QUEUE_TYPE: QueueType;
    }

    pub struct Graphics;
    impl QueueKind for Graphics {
        const QUEUE_TYPE: QueueType = QueueType::Graphics;
    }

    pub struct Compute;
    impl QueueKind for Compute {
        const QUEUE_TYPE: QueueType = QueueType::Compute;
    }

    pub struct Transfer;
    impl QueueKind for Transfer {
        const QUEUE_TYPE: QueueType = QueueType::Graphics;
    }
}

pub enum CommandQueue<K: QueueKind> {
    DX12(dx12::CommandQueue<K>),
}

pub enum CommandPool<K: QueueKind> {
    DX12(dx12::CommandPool<K>),
}

pub mod state {
    pub trait State {}

    pub struct Initial;
    impl State for Initial {}

    pub struct Recording;
    impl State for Recording {}

    pub struct Executable;
    impl State for Executable {}
}

pub enum CommandList<'a, K: QueueKind, S: State> {
    DX12(dx12::CommandList<'a, K, S>),
}

impl<'a> CommandList<'a, Graphics, Initial> {
    pub fn begin(self, pool: &mut CommandPool<Graphics>) -> CommandList<'a, Graphics, Recording> {
        match self {
            Self::DX12(cl) => {
                let pool = pool.try_into().unwrap();
                CommandList::DX12(cl.begin(pool))
            }
        }
    }
}

impl<'a> CommandList<'a, queue::Graphics, state::Recording> {
    pub fn bind_vertex_buffers<T>(&mut self, buf: &'a VertexBuffer<T>) -> &mut Self
    where
        T: BufferLayout + 'a,
    {
        match self {
            Self::DX12(cl) => {}
        }

        self
    }

    // fn bind_index_buffer_u8(&mut self, buf: &Buffer<u8, usage::IndexBuffer>) {}
    pub fn bind_index_buffer_u16(&mut self, buf: &IndexBuffer<u16>) -> &mut Self {
        match self {
            Self::DX12(cl) => {
                let buf = buf.try_into();
                cl.bind_index_buffer_u16(buf.unwrap());
                self
            }
        }
    }

    pub fn bind_index_buffer_u32(&mut self, buf: &IndexBuffer<u32>) -> &mut Self {
        match self {
            Self::DX12(cl) => {
                let buf = buf.try_into();
                cl.bind_index_buffer_u32(buf.unwrap());
                self
            }
        }
    }
}

pub struct RenderPassProps {}

pub struct BufferProps<T: BufferLayout, U: BufferUsage> {
    pub width: NonZeroUsize,
    pub usage: U,
    pub memory: Option<Memory>,
    _marker: PhantomData<T>,
}

impl<T: BufferLayout, U: BufferUsage> BufferProps<T, U> {
    pub fn new(width: NonZeroUsize, usage: U) -> Self {
        Self {
            width,
            usage,
            memory: None,
            _marker: PhantomData,
        }
    }

    pub fn new_with_memory(width: NonZeroUsize, usage: U, memory: Memory) -> Self {
        Self {
            width,
            usage,
            memory: Some(memory),
            _marker: PhantomData,
        }
    }
}

impl BufferProps<(), usage::Unknown> {
    pub fn builder() -> BufferPropsBuilder<(), usage::Unknown> {
        BufferPropsBuilder(Self::new(NonZeroUsize::MIN, usage::Unknown))
    }
}

impl<T: BufferLayout, U: BufferUsage> AsMut<BufferProps<T, U>> for BufferProps<T, U> {
    fn as_mut(&mut self) -> &mut BufferProps<T, U> {
        self
    }
}

pub struct BufferPropsBuilder<T: BufferLayout, U: BufferUsage>(BufferProps<T, U>);

impl<T: BufferLayout, U: BufferUsage> BufferPropsBuilder<T, U> {
    pub fn width(mut self, width: NonZeroUsize) -> Self {
        self.0.width = width;
        self
    }

    pub fn usage<U2: BufferUsage>(self, usage: U2) -> BufferPropsBuilder<T, U2> {
        BufferPropsBuilder(BufferProps {
            width: self.0.width,
            usage,
            memory: self.0.memory,
            _marker: PhantomData,
        })
    }

    pub fn memory(mut self, memory: Option<Memory>) -> Self {
        self.0.memory = memory;
        self
    }

    pub fn layout<T2: BufferLayout>(self) -> BufferPropsBuilder<T2, U> {
        BufferPropsBuilder(BufferProps {
            width: self.0.width,
            usage: self.0.usage,
            memory: self.0.memory,
            _marker: PhantomData,
        })
    }

    pub fn build(self) -> BufferProps<T, U> {
        self.0
    }
}

impl<T: BufferLayout, U: BufferUsage> AsMut<BufferProps<T, U>> for BufferPropsBuilder<T, U> {
    fn as_mut(&mut self) -> &mut BufferProps<T, U> {
        &mut self.0
    }
}

///
///
/// # Safety
pub unsafe trait BufferLayout: Copy {}

unsafe impl BufferLayout for () {}
unsafe impl BufferLayout for u8 {}
unsafe impl BufferLayout for u16 {}
unsafe impl BufferLayout for u32 {}

pub enum Buffer<T: BufferLayout, U: BufferUsage> {
    DX12(dx12::Buffer<T, U>),
}

pub type VertexBuffer<T> = Buffer<T, usage::VertexBuffer>;
pub type IndexBuffer<T> = Buffer<T, usage::IndexBuffer>;

/// This structure contains all the necessary information for creating an
/// image.
pub struct ImageProps<F: FormatType, U: ImageUsage> {
    pub width: NonZeroUsize,
    pub height: NonZeroUsize,
    pub format: F,
    pub usage: U,
    pub memory: Option<Memory>,
}

impl ImageProps<format::Unknown, usage::Unknown> {
    pub fn builder() -> ImagePropsBuilder<format::Unknown, usage::Unknown> {
        ImagePropsBuilder(Self {
            width: NonZeroUsize::MIN,
            height: NonZeroUsize::MIN,
            format: format::Unknown,
            usage: usage::Unknown,
            memory: None,
        })
    }
}

impl<F: FormatType, U: ImageUsage> AsMut<Self> for ImageProps<F, U> {
    fn as_mut(&mut self) -> &mut Self {
        self
    }
}

pub struct ImagePropsBuilder<F: FormatType, U: ImageUsage>(ImageProps<F, U>);

impl<F: FormatType, U: ImageUsage> ImagePropsBuilder<F, U> {
    pub fn width(mut self, width: NonZeroUsize) -> Self {
        self.0.width = width;
        self
    }

    pub fn height(mut self, height: NonZeroUsize) -> Self {
        self.0.height = height;
        self
    }

    pub fn format<F2: FormatType>(self, format: F2) -> ImagePropsBuilder<F2, U> {
        ImagePropsBuilder(ImageProps {
            width: self.0.width,
            height: self.0.height,
            format,
            usage: self.0.usage,
            memory: self.0.memory,
        })
    }

    pub fn usage<U2: ImageUsage>(self, usage: U2) -> ImagePropsBuilder<F, U2> {
        ImagePropsBuilder(ImageProps {
            width: self.0.width,
            height: self.0.height,
            format: self.0.format,
            usage,
            memory: self.0.memory,
        })
    }

    pub fn build(self) -> ImageProps<F, U> {
        self.0
    }
}

impl<F: FormatType, U: ImageUsage> AsMut<ImageProps<F, U>> for ImagePropsBuilder<F, U> {
    fn as_mut(&mut self) -> &mut ImageProps<F, U> {
        &mut self.0
    }
}

pub enum Image<F: FormatType, U: ImageUsage> {
    DX12(dx12::Image<F, U>),
}

pub enum ImageView<'a, F: FormatType> {
    DX12(dx12::ImageView<'a, F>),
}

pub enum Memory {}

pub mod usage {
    // TODO: Make BufferUsage and ImageUsage sealed.

    macro_rules! generate_usages {
        ($vis:vis enum $enum:ident : $trait:ident {
            $($inner:ident,)*
        }) => {
            $vis trait $trait {
                const USAGE_TYPE: $enum;
            }

            $vis enum $enum {
                Unknown,
                $($inner,)*
            }

            $(
                $vis struct $inner;
                impl $trait for $inner {
                    const USAGE_TYPE: $enum = $enum::$inner;
                }
            )*
        };
    }

    generate_usages! {
        pub enum BufferUsageType : BufferUsage {
            VertexBuffer,
            IndexBuffer,
        }
    }

    generate_usages! {
        pub enum ImageUsageType : ImageUsage {
            RenderTarget,
            DepthStencil,
        }
    }

    pub struct Unknown;
    impl BufferUsage for Unknown {
        const USAGE_TYPE: BufferUsageType = BufferUsageType::Unknown;
    }

    impl ImageUsage for Unknown {
        const USAGE_TYPE: ImageUsageType = ImageUsageType::Unknown;
    }
}

pub mod format {
    macro_rules! generate_formats {
        ($vis:vis enum $enum:ident { $($inner:ident,)* }) => {
            #[derive(Debug, Clone, Copy, PartialEq, Eq)]
            $vis enum $enum {
                $($inner,)*
            }

            $(
                $vis struct $inner;
                impl FormatType for $inner {
                    const FORMAT: $enum = $enum::$inner;
                }
            )*

        };
    }

    // TODO: Make this trait sealed.
    pub trait FormatType {
        const FORMAT: Format;
    }

    generate_formats! {
        pub enum Format {
            Unknown,

            R8,
            R8Unorm,
            R8Snorm,
            R8Uint,
            R8Sint,

            R16,
            R16Unorm,
            R16Snorm,
            R16Uint,
            R16Sint,
            R16Float,

            D16Unorm,

            R32,
            R32Uint,
            R32Sint,
            R32Float,

            D32Float,

            R8G8,
            R8G8Unorm,
            R8G8Snorm,
            R8G8Uint,
            R8G8Sint,

            R16G16,
            R16G16Unorm,
            R16G16Snorm,
            R16G16Uint,
            R16G16Sint,
            R16G16Float,

            D24UnormS8Uint,

            R32G32,
            R32G32Uint,
            R32G32Sint,
            R32G32Float,

            R11G11B10Float,

            R32G32B32,
            R32G32B32Uint,
            R32G32B32Sint,
            R32G32B32Float,

            R8G8B8A8,
            R8G8B8A8Unorm,
            R8G8B8A8Snorm,
            R8G8B8A8Uint,
            R8G8B8A8Sint,
            R8G8B8A8Srgb,

            R10G10B10A2,
            R10G10B10A2Unorm,
            R10G10B10A2Uint,

            R16G16B16A16,
            R16G16B16A16Unorm,
            R16G16B16A16Snorm,
            R16G16B16A16Uint,
            R16G16B16A16Sint,
            R16G16B16A16Float,

            R32G32B32A32,
            R32G32B32A32Uint,
            R32G32B32A32Sint,
            R32G32B32A32Float,
        }
    }
}