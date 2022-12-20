pub mod d3d12;

pub mod spirv;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {}

pub enum Backend {
    D3D12,
}

pub enum Instance {
    D3D12(d3d12::Instance),
}

impl Instance {
    pub fn new(backend: Backend, debug: bool) -> Result<Self, Error> {
        let instance = match backend {
            Backend::D3D12 => Self::D3D12(d3d12::Instance::new(debug)?),
        };

        Ok(instance)
    }

    pub fn new_device(&self) -> Result<Device, Error> {
        let device = match self {
            Self::D3D12(instance) => Device::D3D12(instance.new_device()?),
            _ => unreachable!(),
        };

        Ok(device)
    }
}

pub enum Device {
    D3D12(d3d12::Device),
}
