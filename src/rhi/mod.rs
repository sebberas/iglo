use std::marker::PhantomData;

pub mod d3d12;

pub enum Backend {
    D3D12,
}

pub enum Instance {
    D3D12(d3d12::Instance),
}

impl Instance {
    pub fn new(backend: Backend, debug: bool) -> Self {
        match backend {
            Backend::D3D12 => Self::D3D12(d3d12::Instance::new(debug)),
        }
    }
}
