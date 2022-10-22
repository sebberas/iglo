use std::borrow::Cow;

use windows::core::*;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;

use crate::rhi;

#[derive(Clone)]
struct InstanceShared {
    factory: IDXGIFactory7,
}

pub struct Instance(InstanceShared);

impl Instance {
    pub fn new(debug: bool) -> Self {
        let flags = if debug { DXGI_CREATE_FACTORY_DEBUG } else { 0 };
        let factory: IDXGIFactory2 = unsafe { CreateDXGIFactory2(flags).unwrap() };
        let factory: IDXGIFactory7 = factory.cast().unwrap();
        Self(InstanceShared { factory })
    }
}
