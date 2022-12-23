use self::format::Format;
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
        let instance = match backend {
            Backend::DX12 => Self::DX12(dx12::Instance::new(debug)?),
        };

        Ok(instance)
    }

    ///
    ///
    /// # Safety
    pub unsafe fn new_surface(&self, window: *const Window) -> Result<Surface> {
        let surface = match self {
            Self::DX12(instance) => Surface::DX12(instance.new_surface(window)?),
            _ => unreachable!(),
        };

        Ok(surface)
    }

    pub fn new_device(&self, surface: Option<&Surface>) -> Result<Device> {
        let device = match self {
            Self::DX12(i) => {
                let surface = surface.map(TryInto::try_into).transpose();
                i.new_device(surface?).map(Device::DX12)?
            }
            _ => unreachable!(),
        };

        Ok(device)
    }

    pub fn new_swapchain(&self, device: &Device, surface: Surface) -> Result<Swapchain> {
        match self {
            Self::DX12(i) => {
                let (device, surface) = (device.try_into(), surface.try_into());
                i.new_swapchain(device?, surface?).map(Swapchain::DX12)
            }
            _ => unreachable!(),
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
}

pub struct SwapchainCreateInfo {
    pub surface: Surface,
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub backbuffers: usize,
    pub format: Format,
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

pub mod format {
    pub enum Format {
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

        R32Unorm,
        R32Snorm,
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
