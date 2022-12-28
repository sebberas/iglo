use std::ffi::*;
use std::marker::*;
use std::sync::*;
use std::thread::JoinHandle;

use windows::core::{Interface, PCSTR};
use windows::w;
use windows::Win32::Foundation::{HINSTANCE, HWND};
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;

use crate::os::windows::WindowExt;
use crate::os::Window;
use crate::rhi::queue::{QueueKind, QueueType};
use crate::rhi::state::{Executeable, Initial, Recording, State};
use crate::rhi::usage::{BufferUsage, ImageUsage, ImageUsageType};
use crate::rhi::{self, BufferLayout, DeviceProps, Error, Format, FormatType, ImageProps, Result};

struct InstanceDebugger {
    dxgi: (IDXGIDebug, IDXGIInfoQueue),
    thread: Option<JoinHandle<()>>,
}

impl Drop for InstanceDebugger {
    fn drop(&mut self) {
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct InstanceShared {
    factory: IDXGIFactory7,
    debugger: Option<InstanceDebugger>,
}

pub struct Instance(Arc<InstanceShared>);

unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

impl Instance {
    const FEATURE_LEVEL: D3D_FEATURE_LEVEL = D3D_FEATURE_LEVEL_12_1;

    pub fn new(debug: bool) -> Result<Self> {
        let debugger = if debug {
            let controller: IDXGIDebug = unsafe { DXGIGetDebugInterface1(0) }.unwrap();
            let queue = unsafe { DXGIGetDebugInterface1(0) }.unwrap();
            let thread = None;

            Some(InstanceDebugger {
                dxgi: (controller, queue),
                thread,
            })
        } else {
            None
        };

        let flags = if debug { DXGI_CREATE_FACTORY_DEBUG } else { 0 };
        let factory: IDXGIFactory2 = unsafe { CreateDXGIFactory2(flags).unwrap() };
        let factory: IDXGIFactory7 = factory.cast().unwrap();

        Ok(Self(Arc::new(InstanceShared { factory, debugger })))
    }

    /// Creates a new presentable surface.
    ///
    /// # Safety
    ///
    /// TODO
    pub unsafe fn new_surface(&self, window: *const Window) -> Result<Surface> {
        let window = unsafe { &*window };

        Ok(Surface {
            hwnd: window.hwnd(),
            hinstance: window.hinstance(),
            _instance: Arc::clone(&self.0),
        })
    }

    pub fn new_device(&self, surface: Option<&Surface>, props: &DeviceProps) -> Result<Device> {
        let InstanceShared { debugger, .. } = &*self.0;

        if debugger.is_some() {
            let mut controller: Option<ID3D12Debug1> = None;
            unsafe { D3D12GetDebugInterface(&mut controller) }.unwrap();
            if let Some(controller) = controller {
                println!("Enabled Device Debugging!");
                unsafe { controller.EnableDebugLayer() };
            }
        }

        let mut device = None;
        unsafe { D3D12CreateDevice(None, Self::FEATURE_LEVEL, &mut device) }.unwrap();

        let device: ID3D12Device = unsafe { device.unwrap_unchecked() };
        let device: ID3D12Device8 = device.cast().unwrap();

        if debugger.is_some() {
            if let Ok(queue) = device.cast::<ID3D12InfoQueue1>() {
                let mut cookie = 0;
                unsafe {
                    queue.RegisterMessageCallback(
                        Some(Self::debug_callback),
                        D3D12_MESSAGE_CALLBACK_IGNORE_FILTERS,
                        std::ptr::null(),
                        &mut cookie,
                    )
                }
                .unwrap();

                println!("Enabled Device MessageCallback!");
            }
        }

        let present: ID3D12CommandQueue = {
            let desc = D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
                Flags: D3D12_COMMAND_QUEUE_FLAGS(0),
                NodeMask: 0,
            };

            unsafe { device.CreateCommandQueue(&desc) }.unwrap()
        };

        let _ = unsafe { present.SetName(w!("Present ID3D12CommandQueue")) };

        Ok(Device(Arc::new(DeviceShared {
            device,
            present,
            _instance: Arc::clone(&self.0),
        })))
    }

    pub fn new_swapchain(&self, device: &Device, surface: Surface) -> Result<Swapchain> {
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

        let handle: IDXGISwapChain4 = handle.cast().unwrap();

        Ok(Swapchain(Arc::new(SwapchainShared {
            handle,
            present: present.clone(),
            backbuffers: 2,
            _device: Arc::clone(&device.0),
            _instance: Arc::clone(&self.0),
        })))
    }

    extern "system" fn debug_callback(
        _category: D3D12_MESSAGE_CATEGORY,
        severity: D3D12_MESSAGE_SEVERITY,
        _id: D3D12_MESSAGE_ID,
        description: PCSTR,
        _: *mut c_void,
    ) {
        if severity.0 <= 2 {
            let s = unsafe { description.to_string() }.unwrap();
            println!("{s}");
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

    fn try_from(value: rhi::Surface) -> Result<Self> {
        match value {
            rhi::Surface::DX12(surface) => Ok(surface),
            _ => Err(Error::BackendMismatch),
        }
    }
}

impl<'a> TryFrom<&'a rhi::Surface> for &'a Surface {
    type Error = Error;

    fn try_from(value: &'a rhi::Surface) -> Result<Self> {
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
    pub fn new_command_queue<K>(&self, _kind: K) -> Result<Option<CommandQueue<K>>>
    where
        K: QueueKind,
    {
        let DeviceShared { device, .. } = &*self.0;

        let desc = D3D12_COMMAND_QUEUE_DESC {
            Type: match K::QUEUE_TYPE {
                QueueType::Graphics => D3D12_COMMAND_LIST_TYPE_DIRECT,
                QueueType::Compute => D3D12_COMMAND_LIST_TYPE_COMPUTE,
                QueueType::Transfer => D3D12_COMMAND_LIST_TYPE_COPY,
            },
            Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
            ..Default::default()
        };

        let queue = unsafe { device.CreateCommandQueue(&desc) }.unwrap();
        Ok(Some(CommandQueue {
            queue,
            _device: Arc::clone(&self.0),
            _marker: PhantomData,
        }))
    }

    pub fn new_command_pool<K>(&self, queue: &CommandQueue<K>) -> Result<CommandPool<K>>
    where
        K: QueueKind,
    {
        let DeviceShared { device, .. } = &*self.0;

        let allocator =
            unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }.unwrap();

        Ok(CommandPool {
            allocator,
            _device: Arc::clone(&self.0),
            _marker: PhantomData,
        })
    }

    pub fn new_command_list<K>(&self, _: &mut CommandPool<K>) -> Result<CommandList<K, Initial>>
    where
        K: QueueKind,
    {
        let DeviceShared { device, .. } = &*self.0;

        let kind = D3D12_COMMAND_LIST_TYPE_DIRECT;
        let flags = D3D12_COMMAND_LIST_FLAG_NONE;

        let list: ID3D12CommandList = unsafe { device.CreateCommandList1(0, kind, flags) }.unwrap();
        let list = list.cast().unwrap();

        Ok(CommandList {
            list,
            _device: Arc::clone(&self.0),
            _marker: PhantomData,
        })
    }

    pub fn new_buffer<T, U>(&self) -> Result<Buffer<T, U>>
    where
        T: BufferLayout,
        U: BufferUsage,
    {
        let DeviceShared { device, .. } = &*self.0;

        let heap_props = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        };

        let heap_flags = D3D12_HEAP_FLAG_NONE;

        let buffer_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64,
            Width: D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };

        let mut resource: Option<ID3D12Resource> = None;
        unsafe {
            device
                .CreateCommittedResource(
                    &heap_props,
                    heap_flags,
                    &buffer_desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut resource,
                )
                .unwrap()
        };

        todo!()
    }

    pub fn new_image<F, U>(&self, info: &mut ImageProps<F, U>) -> Result<Image<F, U>>
    where
        F: Format,
        U: ImageUsage,
    {
        let DeviceShared { device, .. } = &*self.0;

        let mut resource: Option<ID3D12Resource> = None;
        if let Some(_memory) = info.memory.take() {
            todo!()
        } else {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };

            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Alignment: 0,
                Width: info.width.get() as u64,
                Height: info.height.get() as u32,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: F::FORMAT_TYPE.into(),
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: match U::USAGE_TYPE {
                    ImageUsageType::Unknown => {
                        panic!("Images cannot be created with an unknown format")
                    }
                    ImageUsageType::DepthStencil => D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                },
            };

            unsafe {
                let result = device.CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut resource,
                );

                result.unwrap();
            }
        }

        // SAFETY:
        let resource = unsafe { resource.unwrap_unchecked() };
        Ok(Image {
            resource,
            _marker: PhantomData,
        })
    }
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl TryFrom<rhi::Device> for Device {
    type Error = Error;

    fn try_from(value: rhi::Device) -> Result<Self> {
        match value {
            rhi::Device::DX12(inner) => Ok(inner),
            _ => Err(Error::BackendMismatch),
        }
    }
}

impl<'a> TryFrom<&'a rhi::Device> for &'a Device {
    type Error = Error;

    fn try_from(value: &'a rhi::Device) -> Result<Self> {
        match value {
            rhi::Device::DX12(inner) => Ok(inner),
            _ => Err(Error::BackendMismatch),
        }
    }
}

struct SwapchainShared {
    handle: IDXGISwapChain4,
    present: ID3D12CommandQueue,
    backbuffers: u32,
    _device: Arc<DeviceShared>,
    _instance: Arc<InstanceShared>,
}

pub struct Swapchain(Arc<SwapchainShared>);

impl Swapchain {
    pub fn present(&mut self) -> Result<()> {
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

pub struct RenderTarget {}

pub struct CommandQueue<K: QueueKind> {
    queue: ID3D12CommandQueue,
    _device: Arc<DeviceShared>,
    _marker: PhantomData<K>,
}

impl<'a, K: QueueKind> TryFrom<&'a rhi::CommandQueue<K>> for &'a CommandQueue<K> {
    type Error = Error;

    fn try_from(value: &'a rhi::CommandQueue<K>) -> Result<Self> {
        match value {
            rhi::CommandQueue::DX12(q) => Ok(q),
            _ => Err(Error::BackendMismatch),
        }
    }
}

pub struct CommandPool<K: QueueKind> {
    allocator: ID3D12CommandAllocator,
    _device: Arc<DeviceShared>,
    _marker: PhantomData<K>,
}

impl<'a, K: QueueKind> TryFrom<&'a rhi::CommandPool<K>> for &'a CommandPool<K> {
    type Error = Error;

    fn try_from(value: &'a rhi::CommandPool<K>) -> Result<Self> {
        match value {
            rhi::CommandPool::DX12(p) => Ok(p),
            _ => Err(Error::BackendMismatch),
        }
    }
}

impl<'a, K: QueueKind> TryFrom<&'a mut rhi::CommandPool<K>> for &'a mut CommandPool<K> {
    type Error = Error;

    fn try_from(value: &'a mut rhi::CommandPool<K>) -> Result<Self> {
        match value {
            rhi::CommandPool::DX12(p) => Ok(p),
            _ => Err(Error::BackendMismatch),
        }
    }
}

pub struct CommandList<K: QueueKind, S: State> {
    list: ID3D12GraphicsCommandList6,
    _device: Arc<DeviceShared>,
    _marker: PhantomData<(K, S)>,
}

pub struct Buffer<T: BufferLayout, U: BufferUsage> {
    resource: ID3D12Resource2,
    _marker: PhantomData<(T, U)>,
}

pub struct Image<F: Format, U: ImageUsage> {
    resource: ID3D12Resource,
    _marker: PhantomData<(F, U)>,
}

pub struct ImageView<'a, F: Format> {
    _marker: PhantomData<(&'a (), F)>,
}

impl From<FormatType> for DXGI_FORMAT {
    fn from(value: FormatType) -> Self {
        use FormatType::*;

        match value {
            Unknown => DXGI_FORMAT_UNKNOWN,

            R8Unorm => DXGI_FORMAT_R8_UNORM,
            R8Snorm => DXGI_FORMAT_R8_SNORM,
            R8Uint => DXGI_FORMAT_R8_UINT,
            R8Sint => DXGI_FORMAT_R8_SINT,

            R16Unorm => DXGI_FORMAT_R16_UNORM,
            R16Snorm => DXGI_FORMAT_R16_SNORM,
            R16Uint => DXGI_FORMAT_R16_UINT,
            R16Sint => DXGI_FORMAT_R16_SINT,
            R16Float => DXGI_FORMAT_R16_FLOAT,

            D16Unorm => DXGI_FORMAT_D16_UNORM,

            R32Uint => DXGI_FORMAT_R32_UINT,
            R32Sint => DXGI_FORMAT_R32_SINT,
            R32Float => DXGI_FORMAT_R32_FLOAT,

            D32Float => DXGI_FORMAT_D32_FLOAT,

            R8G8Unorm => DXGI_FORMAT_R8G8_UNORM,
            R8G8Snorm => DXGI_FORMAT_R8G8_SNORM,
            R8G8Uint => DXGI_FORMAT_R8G8_UINT,
            R8G8Sint => DXGI_FORMAT_R8G8_SINT,

            R16G16Unorm => DXGI_FORMAT_R16G16_UNORM,
            R16G16Snorm => DXGI_FORMAT_R16G16_SNORM,
            R16G16Uint => DXGI_FORMAT_R16G16_UINT,
            R16G16Sint => DXGI_FORMAT_R16G16_SINT,
            R16G16Float => DXGI_FORMAT_R16G16_FLOAT,

            D24UnormS8Uint => DXGI_FORMAT_D24_UNORM_S8_UINT,

            R32G32Uint => DXGI_FORMAT_R32G32_UINT,
            R32G32Sint => DXGI_FORMAT_R32G32_SINT,
            R32G32Float => DXGI_FORMAT_R32G32_FLOAT,

            R11G11B10Float => DXGI_FORMAT_R11G11B10_FLOAT,

            R32G32B32Uint => DXGI_FORMAT_R32G32B32_UINT,
            R32G32B32Sint => DXGI_FORMAT_R32G32B32_SINT,
            R32G32B32Float => DXGI_FORMAT_R32G32B32_FLOAT,

            R8G8B8A8Unorm => DXGI_FORMAT_R8G8B8A8_UNORM,
            R8G8B8A8Snorm => DXGI_FORMAT_R8G8B8A8_SNORM,
            R8G8B8A8Uint => DXGI_FORMAT_R8G8B8A8_UINT,
            R8G8B8A8Sint => DXGI_FORMAT_R8G8B8A8_SINT,
            R8G8B8A8Srgb => DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,

            R10G10B10A2Unorm => DXGI_FORMAT_R10G10B10A2_UNORM,
            R10G10B10A2Uint => DXGI_FORMAT_R10G10B10A2_UINT,

            R16G16B16A16Unorm => DXGI_FORMAT_R16G16B16A16_UNORM,
            R16G16B16A16Snorm => DXGI_FORMAT_R16G16B16A16_SNORM,
            R16G16B16A16Uint => DXGI_FORMAT_R16G16B16A16_UINT,
            R16G16B16A16Sint => DXGI_FORMAT_R16G16B16A16_SINT,
            R16G16B16A16Float => DXGI_FORMAT_R16G16B16A16_FLOAT,

            R32G32B32A32Uint => DXGI_FORMAT_R32G32B32A32_UINT,
            R32G32B32A32Sint => DXGI_FORMAT_R32G32B32A32_SINT,
            R32G32B32A32Float => DXGI_FORMAT_R32G32B32A32_FLOAT,
        }
    }
}
