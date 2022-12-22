use crate::os::Window;

pub mod dx12;

pub mod spirv;

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
    pub fn new(backend: Backend, debug: bool) -> Result<Self, Error> {
        let instance = match backend {
            Backend::DX12 => Self::DX12(dx12::Instance::new(debug)?),
        };

        Ok(instance)
    }

    pub unsafe fn new_surface(&self, window: *const Window) -> Result<Surface, Error> {
        let surface = match self {
            Self::DX12(instance) => Surface::DX12(instance.new_surface(window)?),
            _ => unreachable!(),
        };

        Ok(surface)
    }

    pub fn new_device(&self, surface: Option<&Surface>) -> Result<Device, Error> {
        let device = match self {
            Self::DX12(i) => Device::DX12(i.new_device(surface.map(|s| s.try_into().unwrap()))?),
            _ => unreachable!(),
        };

        Ok(device)
    }

    pub fn new_swapchain(&self, device: &Device, surface: Surface) -> Result<Swapchain, Error> {
        let swapchain = match self {
            Self::DX12(i) => {
                let (device, surface) = (device.try_into(), surface.try_into());
                Swapchain::DX12(i.new_swapchain(device?, surface?)?)
            }
            _ => unreachable!(),
        };

        Ok(swapchain)
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

pub enum Swapchain {
    DX12(dx12::Swapchain),
}
