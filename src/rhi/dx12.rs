use std::ffi::*;
use std::sync::*;
use std::thread::JoinHandle;

use windows::core::{Interface, GUID, PCSTR, PCWSTR};
use windows::Win32::Foundation::{HINSTANCE, HWND};
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;

use crate::os::windows::WindowExt;
use crate::os::Window;
use crate::rhi::{self, Error};

struct Debugger {
    dxgi: (IDXGIDebug1, IDXGIInfoQueue),
    thread: Option<JoinHandle<()>>,
}

impl Drop for Debugger {
    fn drop(&mut self) {
        if let Some(thread) = self.thread.take() {
            thread.join();
        }
    }
}

struct IDXGIInfoQueueSend(IDXGIInfoQueue);

unsafe impl Send for IDXGIInfoQueueSend {}
unsafe impl Sync for IDXGIInfoQueueSend {}

struct InstanceShared {
    factory: IDXGIFactory7,
    debugger: Option<Debugger>,
}

pub struct Instance(Arc<InstanceShared>);

impl Instance {
    const FEATURE_LEVEL: D3D_FEATURE_LEVEL = D3D_FEATURE_LEVEL_12_1;

    pub fn new(debug: bool) -> Result<Self, Error> {
        let flags = if debug { DXGI_CREATE_FACTORY_DEBUG } else { 0 };
        let factory: IDXGIFactory2 = unsafe { CreateDXGIFactory2(flags).unwrap() };
        let factory: IDXGIFactory7 = factory.cast().unwrap();

        let mut debugger = None;

        Ok(Self(Arc::new(InstanceShared { factory, debugger })))
    }

    /// Creates a new presentable surface.
    ///
    /// # Safety
    ///
    /// TODO
    pub unsafe fn new_surface(&self, window: *const Window) -> Result<Surface, Error> {
        let window = unsafe { &*window };

        Ok(Surface {
            hwnd: window.hwnd(),
            hinstance: window.hinstance(),
            _instance: Arc::clone(&self.0),
        })
    }

    pub fn new_device(&self, surface: Option<&Surface>) -> Result<Device, Error> {
        let InstanceShared { debugger, .. } = &*self.0;

        let mut device = None;
        unsafe { D3D12CreateDevice(None, Self::FEATURE_LEVEL, &mut device) }.unwrap();

        let device: ID3D12Device = unsafe { device.unwrap_unchecked() };
        let device: ID3D12Device8 = device.cast().unwrap();

        unsafe {
            let mut data = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
                Format: DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
                ..Default::default()
            };

            device.CheckFeatureSupport(
                D3D12_FEATURE_FORMAT_SUPPORT,
                &mut data as *mut _ as *mut _,
                std::mem::size_of_val(&data) as u32,
            );

            println!(
                "{:?}",
                (data.Support1 & D3D12_FORMAT_SUPPORT1_RENDER_TARGET)
                    == D3D12_FORMAT_SUPPORT1_RENDER_TARGET
            );
        }

        let present = {
            let desc = D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
                Flags: D3D12_COMMAND_QUEUE_FLAGS(0),
                NodeMask: 0,
            };

            unsafe { device.CreateCommandQueue(&desc) }.unwrap()
        };

        Ok(Device(Arc::new(DeviceShared {
            device,
            present,
            _instance: Arc::clone(&self.0),
        })))
    }

    pub fn new_swapchain(&self, device: &Device, surface: Surface) -> Result<Swapchain, Error> {
        let InstanceShared { factory, .. } = &*self.0;
        let DeviceShared { present, .. } = &*device.0;
        let Surface { hwnd, .. } = surface;

        let desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: 0,
            Height: 0,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            Stereo: false.into(),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: 2,
            Scaling: DXGI_SCALING_STRETCH,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
            AlphaMode: DXGI_ALPHA_MODE_UNSPECIFIED,
            Flags: 0,
        };

        let fdesc = DXGI_SWAP_CHAIN_FULLSCREEN_DESC {
            Windowed: true.into(),
            ..Default::default()
        };

        let handle = unsafe {
            factory
                .CreateSwapChainForHwnd(present, hwnd, &desc, Some(&fdesc), None)
                .unwrap()
        };

        let handle = handle.cast().unwrap();

        Ok(Swapchain(Arc::new(SwapchainShared {
            handle,
            _device: Arc::clone(&device.0),
            _instance: Arc::clone(&self.0),
        })))
    }
}

pub struct Surface {
    hwnd: HWND,
    hinstance: HINSTANCE,
    _instance: Arc<InstanceShared>,
}

impl TryFrom<rhi::Surface> for Surface {
    type Error = Error;

    fn try_from(value: rhi::Surface) -> Result<Self, Error> {
        match value {
            rhi::Surface::DX12(surface) => Ok(surface),
            _ => Err(Error::BackendMismatch),
        }
    }
}

impl<'a> TryFrom<&'a rhi::Surface> for &'a Surface {
    type Error = Error;

    fn try_from(value: &'a rhi::Surface) -> Result<Self, Error> {
        match value {
            rhi::Surface::DX12(surface) => Ok(surface),
            _ => Err(Error::BackendMismatch),
        }
    }
}

struct DeviceShared {
    device: ID3D12Device8,
    present: ID3D12CommandQueue,
    _instance: Arc<InstanceShared>,
}

pub struct Device(Arc<DeviceShared>);

impl Device {
    pub fn new_command_queue(&self) -> Result<CommandQueue, Error> {
        let DeviceShared { device, .. } = &*self.0;

        let desc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
            ..Default::default()
        };

        let queue = unsafe { device.CreateCommandQueue(&desc) }.unwrap();
        Ok(CommandQueue {
            queue,
            _device: Arc::clone(&self.0),
        })
    }

    pub fn new_command_pool(&self) -> Result<CommandPool, Error> {
        let DeviceShared { device, .. } = &*self.0;

        let allocator =
            unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }.unwrap();

        Ok(CommandPool {
            allocator,
            _device: Arc::clone(&self.0),
        })
    }

    pub fn new_command_list(&self, _pool: &mut CommandPool) -> Result<CommandList, Error> {
        let DeviceShared { device, .. } = &*self.0;

        let list = unsafe {
            device.CreateCommandList1(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                D3D12_COMMAND_LIST_FLAG_NONE,
            )
        }
        .unwrap();

        Ok(CommandList {
            list,
            _device: Arc::clone(&self.0),
        })
    }
}

impl TryFrom<rhi::Device> for Device {
    type Error = Error;

    fn try_from(value: rhi::Device) -> Result<Self, Error> {
        match value {
            rhi::Device::DX12(inner) => Ok(inner),
            _ => Err(Error::BackendMismatch),
        }
    }
}

impl<'a> TryFrom<&'a rhi::Device> for &'a Device {
    type Error = Error;

    fn try_from(value: &'a rhi::Device) -> Result<Self, Error> {
        match value {
            rhi::Device::DX12(inner) => Ok(inner),
            _ => Err(Error::BackendMismatch),
        }
    }
}

struct SwapchainShared {
    handle: IDXGISwapChain4,
    _device: Arc<DeviceShared>,
    _instance: Arc<InstanceShared>,
}

pub struct Swapchain(Arc<SwapchainShared>);

impl Swapchain {
    pub fn present(&mut self) -> Result<(), Error> {
        let SwapchainShared { handle, .. } = &*self.0;

        let params = DXGI_PRESENT_PARAMETERS {
            DirtyRectsCount: 0,
            pDirtyRects: std::ptr::null_mut(),
            pScrollRect: std::ptr::null_mut(),
            pScrollOffset: std::ptr::null_mut(),
        };

        unsafe { handle.Present1(1, DXGI_PRESENT_RESTART, &params) }.unwrap();
        Ok(())
    }
}

pub struct CommandQueue {
    queue: ID3D12CommandQueue,
    _device: Arc<DeviceShared>,
}

pub struct CommandPool {
    allocator: ID3D12CommandAllocator,
    _device: Arc<DeviceShared>,
}

impl<'a> TryFrom<&'a rhi::CommandPool> for &'a CommandPool {
    type Error = Error;

    fn try_from(value: &'a rhi::CommandPool) -> Result<Self, Self::Error> {
        match value {
            rhi::CommandPool::DX12(p) => Ok(p),
            _ => Err(Error::BackendMismatch),
        }
    }
}

impl<'a> TryFrom<&'a mut rhi::CommandPool> for &'a mut CommandPool {
    type Error = Error;

    fn try_from(value: &'a mut rhi::CommandPool) -> Result<Self, Self::Error> {
        match value {
            rhi::CommandPool::DX12(p) => Ok(p),
            _ => Err(Error::BackendMismatch),
        }
    }
}

pub struct CommandList {
    list: ID3D12GraphicsCommandList6,
    _device: Arc<DeviceShared>,
}
