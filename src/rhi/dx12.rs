use std::alloc::Layout;
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
    thread: JoinHandle<()>,
}

impl Drop for Debugger {
    fn drop(&mut self) {
        let _ = self.thread.join();
    }
}

struct IDXGIInfoQueueSend(IDXGIInfoQueue);

unsafe impl Send for IDXGIInfoQueueSend {}

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
        if debug {
            let controller: IDXGIDebug1 = unsafe { DXGIGetDebugInterface1(0) }.unwrap();
            let queue: IDXGIInfoQueue = unsafe { DXGIGetDebugInterface1(0) }.unwrap();
            unsafe {
                queue.SetBreakOnSeverity(
                    DXGI_DEBUG_ALL,
                    DXGI_INFO_QUEUE_MESSAGE_SEVERITY_ERROR,
                    true,
                )
            };

            unsafe {
                queue.SetBreakOnSeverity(
                    DXGI_DEBUG_ALL,
                    DXGI_INFO_QUEUE_MESSAGE_SEVERITY_CORRUPTION,
                    true,
                )
            };

            let thread = {
                let queue = IDXGIInfoQueueSend(queue.clone());
                std::thread::spawn(move || {
                    let queue = queue.0;
                    const PRODUCER: GUID = DXGI_DEBUG_DX;
                    let max = unsafe { queue.GetMessageCountLimit(PRODUCER) };
                    let mut i = 0;

                    loop {
                        i = if i == max { 0 } else { i };
                    }
                })
            };

            debugger = Some(Debugger {
                dxgi: (controller, queue),
                thread,
            })
        }

        Ok(Self(Arc::new(InstanceShared { factory, debugger })))
    }

    /// Creates a new presentable surface.
    pub unsafe fn new_surface(&self, window: *const Window) -> Result<Surface, Error> {
        let window = unsafe { &*window };

        Ok(Surface {
            hwnd: window.hwnd(),
            hinstance: window.hinstance(),
            _instance: Arc::clone(&self.0),
        })
    }

    pub fn new_device(&self, surface: Option<&Surface>) -> Result<Device, Error> {
        if self.0.debug {
            let mut debugger: Option<ID3D12Debug> = None;
            unsafe { D3D12GetDebugInterface(&mut debugger) }.unwrap();
            let debugger = debugger.unwrap();

            unsafe { debugger.EnableDebugLayer() };
        }

        let mut handle = None;
        unsafe { D3D12CreateDevice(None, Self::FEATURE_LEVEL, &mut handle) }.unwrap();

        let handle: ID3D12Device = unsafe { handle.unwrap_unchecked() };
        let handle: ID3D12Device8 = handle.cast().unwrap();

        let info_queue: ID3D12InfoQueue = handle.cast().unwrap();

        let present = {
            let desc = D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
                Flags: D3D12_COMMAND_QUEUE_FLAGS(0),
                NodeMask: 0,
            };

            unsafe { handle.CreateCommandQueue(&desc) }.unwrap()
        };

        Ok(Device(Arc::new(DeviceShared {
            handle,
            present,
            info_queue,
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
            Format: DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
            Stereo: false.into(),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: 1,
            Scaling: DXGI_SCALING_STRETCH,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
            AlphaMode: DXGI_ALPHA_MODE_UNSPECIFIED,
            Flags: 0,
        };

        Ok(Swapchain(Arc::new(SwapchainShared {
            handle: todo!(),
            _device: Arc::clone(&device.0),
            _instance: Arc::clone(&self.0),
        })))
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        if let Some(thread) = self.0.debug_thread {
            thread.join();
        }
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
    handle: ID3D12Device8,
    present: ID3D12CommandQueue,
    info_queue: ID3D12InfoQueue,
    _instance: Arc<InstanceShared>,
}

pub struct Device(Arc<DeviceShared>);

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
