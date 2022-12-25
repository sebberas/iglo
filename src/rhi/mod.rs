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

use std::marker::PhantomData;
use std::num::NonZeroUsize;

pub use self::format::{Format, FormatType};
pub use self::usage::{BufferUsage, BufferUsageType, ImageUsage, ImageUsageType};
use crate::os::Window;

pub mod dx12;

pub mod spirv;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    BackendMismatch,
}

pub enum Backend {
    DX12,
}

pub enum Instance {
    DX12(dx12::Instance),
}

impl Instance {
    pub fn new(backend: Backend, debug: bool) -> Result<Self> {
        match backend {
            Backend::DX12 => dx12::Instance::new(debug).map(Self::DX12),
        }
    }

    ///
    ///
    /// # Safety
    pub unsafe fn new_surface(&self, window: *const Window) -> Result<Surface> {
        match self {
            Self::DX12(instance) => instance.new_surface(window).map(Surface::DX12),
        }
    }

    pub fn new_device(&self, surface: Option<&Surface>) -> Result<Device> {
        match self {
            Self::DX12(i) => {
                let surface = surface.map(TryInto::try_into).transpose();
                i.new_device(surface?).map(Device::DX12)
            }
        }
    }

    pub fn new_swapchain(&self, device: &Device, surface: Surface) -> Result<Swapchain> {
        match self {
            Self::DX12(i) => {
                let (device, surface) = (device.try_into(), surface.try_into());
                i.new_swapchain(device?, surface?).map(Swapchain::DX12)
            }
        }
    }
}

pub enum Surface {
    DX12(dx12::Surface),
}

pub struct DeviceInfo<'a> {
    pub surface: Option<&'a Surface>,
}

pub enum Device {
    DX12(dx12::Device),
}

impl Device {
    /// Creates a new queue that can be used for execution on the GPU.
    pub fn new_command_queue(&self) -> Result<CommandQueue> {
        match self {
            Self::DX12(d) => d.new_command_queue().map(CommandQueue::DX12),
        }
    }

    pub fn new_command_pool(&self) -> Result<CommandPool> {
        match self {
            Self::DX12(d) => d.new_command_pool().map(CommandPool::DX12),
        }
    }

    pub fn new_command_list(&self, pool: &mut CommandPool) -> Result<CommandList> {
        match self {
            Self::DX12(d) => {
                let pool = pool.try_into()?;
                d.new_command_list(pool).map(CommandList::DX12)
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
    /// - Panics if `ImageProps::width` or `ImageProps::height` are
    /// larger than the size allowed by the API. These constraints should be
    /// queried at runtime.
    ///
    /// # Returns
    ///
    /// Returns a new image with the specified format and usage if the operation
    /// is successfull. Otherwise it returns an error.
    pub fn new_image<F, U>(&self, p: &mut impl AsMut<ImageProps<F, U>>) -> Result<Image<F, U>>
    where
        F: Format,
        U: ImageUsage,
    {
        match self {
            Self::DX12(d) => d.new_image(p.as_mut()).map(Image::DX12),
        }
    }
}

pub struct SwapchainCreateInfo<F: Format> {
    pub surface: Surface,
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub backbuffers: usize,
    pub format: F,
}

pub enum Swapchain {
    DX12(dx12::Swapchain),
}

impl Swapchain {
    pub fn present(&mut self) -> Result<()> {
        match self {
            Self::DX12(s) => s.present(),
        }
    }
}

pub enum CommandQueue {
    DX12(dx12::CommandQueue),
}

pub enum CommandPool {
    DX12(dx12::CommandPool),
}

pub enum CommandList {
    DX12(dx12::CommandList),
}

pub struct BufferProps<T: BufferLayout, U: BufferUsage> {
    pub width: NonZeroUsize,
    pub usage: U,
    pub memory: Option<Memory>,
    _marker: PhantomData<T>,
}

impl<T: BufferLayout, U: BufferUsage> BufferProps<T, U> {
    pub fn new(width: NonZeroUsize, usage: U, memory: Option<Memory>) -> Self {
        Self {
            width,
            usage,
            memory,
            _marker: PhantomData,
        }
    }
}

impl BufferProps<(), usage::Unknown> {
    pub fn builder() -> BufferPropsBuilder<(), usage::Unknown> {
        BufferPropsBuilder(Self::new(NonZeroUsize::MIN, usage::Unknown, None))
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

pub unsafe trait BufferLayout: Sized {}

unsafe impl BufferLayout for () {}
unsafe impl BufferLayout for f32 {}

pub enum Buffer<T: BufferLayout, U: BufferUsage> {
    DX12(dx12::Buffer<T, U>),
}

/// This structure contains all the necessary information for creating an
/// image.
pub struct ImageProps<F: Format, U: ImageUsage> {
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

impl<F: Format, U: ImageUsage> AsMut<Self> for ImageProps<F, U> {
    fn as_mut(&mut self) -> &mut Self {
        self
    }
}

pub struct ImagePropsBuilder<F: Format, U: ImageUsage>(ImageProps<F, U>);

impl<F: Format, U: ImageUsage> ImagePropsBuilder<F, U> {
    pub fn width(mut self, width: NonZeroUsize) -> Self {
        self.0.width = width;
        self
    }

    pub fn height(mut self, height: NonZeroUsize) -> Self {
        self.0.height = height;
        self
    }

    pub fn format<F2: Format>(self, format: F2) -> ImagePropsBuilder<F2, U> {
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

impl<F: Format, U: ImageUsage> AsMut<ImageProps<F, U>> for ImagePropsBuilder<F, U> {
    fn as_mut(&mut self) -> &mut ImageProps<F, U> {
        &mut self.0
    }
}

pub enum Image<F: Format, U: ImageUsage> {
    DX12(dx12::Image<F, U>),
}

pub enum ImageView<'a, F: Format> {
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
            $vis enum $enum {
                $($inner,)*
            }

            $(
                $vis struct $inner;
                impl Format for $inner {
                    const FORMAT_TYPE: $enum = $enum::$inner;
                }
            )*

        };
    }

    // TODO: Make this trait sealed.
    pub trait Format {
        const FORMAT_TYPE: FormatType;
    }

    generate_formats! {
        pub enum FormatType {
            Unknown,

            R8Unorm,
            R8Snorm,
            R8Uint,
            R8Sint,

            R16Unorm,
            R16Snorm,
            R16Uint,
            R16Sint,
            R16Float,

            R32Uint,
            R32Sint,
            R32Float,

            R8G8Unorm,
            R8G8Snorm,
            R8G8Uint,
            R8G8Sint,

            R16G16Unorm,
            R16G16Snorm,
            R16G16Uint,
            R16G16Sint,
            R16G16Float,

            R32G32Uint,
            R32G32Sint,
            R32G32Float,

            R11G11B10Float,

            R32G32B32Uint,
            R32G32B32Sint,
            R32G32B32Float,

            R8G8B8A8Unorm,
            R8G8B8A8Snorm,
            R8G8B8A8Uint,
            R8G8B8A8Sint,
            R8G8B8A8Srgb,

            R10G10B10A2Unorm,
            R10G10B10A2Uint,

            R16G16B16A16Unorm,
            R16G16B16A16Snorm,
            R16G16B16A16Uint,
            R16G16B16A16Sint,
            R16G16B16A16Float,

            R32G32B32A32Uint,
            R32G32B32A32Sint,
            R32G32B32A32Float,
        }
    }
}
