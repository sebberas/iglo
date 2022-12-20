use std::sync::*;

use windows::core::{Interface, PCWSTR};
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;

use crate::rhi::{self, Error};

#[derive(Clone)]
struct InstanceShared {
    factory: IDXGIFactory7,
}

pub struct Instance(Arc<InstanceShared>);

impl Instance {
    const FEATURE_LEVEL: D3D_FEATURE_LEVEL = D3D_FEATURE_LEVEL_12_1;

    pub fn new(debug: bool) -> Result<Self, Error> {
        let flags = if debug { DXGI_CREATE_FACTORY_DEBUG } else { 0 };
        let factory: IDXGIFactory2 = unsafe { CreateDXGIFactory2(flags).unwrap() };
        let factory: IDXGIFactory7 = factory.cast().unwrap();

        let mut i = 0;
        const PREFERENCE: DXGI_GPU_PREFERENCE = DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE;
        while let Ok(adapter) = unsafe { factory.EnumAdapterByGpuPreference(i, PREFERENCE) } {
            let adapter: IDXGIAdapter = adapter;
            let DXGI_ADAPTER_DESC { Description, .. } = unsafe { adapter.GetDesc() }.unwrap();
            let name = PCWSTR::from_raw(Description.as_ptr());
            let name = unsafe { name.to_string() }.unwrap();
            println!("{name}");

            i += 1;
        }

        Ok(Self(Arc::new(InstanceShared { factory })))
    }

    pub fn new_device(&self) -> Result<Device, Error> {
        let mut handle = None;
        unsafe { D3D12CreateDevice(None, Self::FEATURE_LEVEL, &mut handle) }.unwrap();

        let handle: ID3D12Device = unsafe { handle.unwrap_unchecked() };
        let handle: ID3D12Device9 = handle.cast().unwrap();

        Ok(Device(Arc::new(DeviceShared {
            handle,
            _instance: Arc::clone(&self.0),
        })))
    }
}

struct DeviceShared {
    handle: ID3D12Device9,
    _instance: Arc<InstanceShared>,
}

pub struct Device(Arc<DeviceShared>);
