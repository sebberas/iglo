use std::alloc::*;
use std::borrow::Cow;
use std::ffi::*;
use std::marker::*;
use std::sync::*;
use std::thread::JoinHandle;
use std::time::Duration;

use windows::core::{Interface, GUID, PCSTR};
use windows::Win32::Foundation::{HINSTANCE, HWND, *};
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::{s, w};

use super::queue::Graphics;
use super::Fullscreen;
use crate::os::windows::WindowExt;
use crate::os::Window;
use crate::rhi::macros::*;
use crate::rhi::queue::{self, QueueKind, QueueType};
use crate::rhi::state::{Executable, Initial, Recording, State};
use crate::rhi::usage::{self, BufferUsage, ImageUsage, ImageUsageType};
use crate::rhi::{self, BufferLayout, DeviceProps, Error, Format, FormatType, ImageProps, Result};

impl From<windows::core::Error> for Error {
    fn from(value: windows::core::Error) -> Self {
        match value.code() {
            E_OUTOFMEMORY => Error::OutOfHostMemory,
            E_NOINTERFACE | E_NOTIMPL => Error::NotSupported,
            E_ABORT | E_FAIL | E_UNEXPECTED => Error::Unknown,
            _ => Error::Other(Cow::Owned(value.to_string())),
        }
    }
}

struct InstanceDebugger {
    dxgi: (IDXGIDebug, Arc<IDXGIInfoQueueSync>),
    thread: Option<JoinHandle<()>>,
}

unsafe impl Send for InstanceDebugger {}
unsafe impl Sync for InstanceDebugger {}

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

struct IDXGIInfoQueueSync(IDXGIInfoQueue);

unsafe impl Send for IDXGIInfoQueueSync {}
unsafe impl Sync for IDXGIInfoQueueSync {}

pub struct Instance(Arc<InstanceShared>);

unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

impl Instance {
    const FEATURE_LEVEL: D3D_FEATURE_LEVEL = D3D_FEATURE_LEVEL_12_1;

    pub fn new(debug: bool) -> Result<Self> {
        let debugger = if debug {
            const PRODUCER: GUID = DXGI_DEBUG_ALL;
            let controller: IDXGIDebug = unsafe { DXGIGetDebugInterface1(0) }?;
            let queue: IDXGIInfoQueue = unsafe { DXGIGetDebugInterface1(0) }?;

            let filter = DXGI_INFO_QUEUE_FILTER {
                ..Default::default()
            };

            unsafe {
                let _ = queue.SetBreakOnSeverity(
                    PRODUCER,
                    DXGI_INFO_QUEUE_MESSAGE_SEVERITY_CORRUPTION,
                    true,
                );

                let _ = queue.SetBreakOnSeverity(
                    PRODUCER,
                    DXGI_INFO_QUEUE_MESSAGE_SEVERITY_ERROR,
                    true,
                );

                let _ = queue.AddStorageFilterEntries(PRODUCER, &filter);
            }

            let queue = Arc::new(IDXGIInfoQueueSync(queue));

            let thread = {
                let queue = Arc::downgrade(&queue);
                Some(std::thread::spawn(move || {
                    const TIMEOUT: f32 = 1.0 / 60.0;

                    let max = unsafe { queue.upgrade().unwrap().0.GetMessageCountLimit(PRODUCER) };
                    let mut i = 0;

                    while let Some(queue) = queue.upgrade() {
                        let queue = &queue.0;

                        let stored = unsafe { queue.GetNumStoredMessages(PRODUCER) };
                        // println!("STORED {stored}");
                        while stored <= max && i < stored {
                            let mut size = 0;
                            // SAFETY:
                            let _ = unsafe { queue.GetMessage(PRODUCER, i, None, &mut size) };

                            let (msg, msg_layout): (*mut DXGI_INFO_QUEUE_MESSAGE, _) = {
                                let align = std::mem::align_of::<DXGI_INFO_QUEUE_MESSAGE>();
                                let layout = Layout::from_size_align(size, align).unwrap();
                                let msg = unsafe { std::alloc::alloc_zeroed(layout) as *mut _ };
                                (msg, layout)
                            };

                            // SAFETY:
                            let r = unsafe { queue.GetMessage(PRODUCER, i, Some(msg), &mut size) };
                            if r.is_ok() {
                                let s = unsafe { CStr::from_ptr((*msg).pDescription as *mut _) };
                                println!("{:?}", s.to_str().unwrap());
                            }

                            // SAFETY
                            unsafe { std::alloc::dealloc(msg as *mut _, msg_layout) };

                            i += 1;
                        }

                        std::thread::park_timeout(Duration::from_secs_f32(TIMEOUT));
                    }
                }))
            };

            Some(InstanceDebugger {
                dxgi: (controller, queue),
                thread,
            })
        } else {
            None
        };

        let flags = if debug { DXGI_CREATE_FACTORY_DEBUG } else { 0 };
        let factory: IDXGIFactory2 = unsafe { CreateDXGIFactory2(flags) }?;
        let factory: IDXGIFactory7 = factory.cast()?;

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
            unsafe { D3D12GetDebugInterface(&mut controller) }?;
            if let Some(controller) = controller {
                println!("Enabled Device Debugging!");
                unsafe { controller.EnableDebugLayer() };
            }
        }

        let mut device = None;
        unsafe { D3D12CreateDevice(None, Self::FEATURE_LEVEL, &mut device) }?;

        let device: ID3D12Device = unsafe { device.unwrap_unchecked() };
        let device: ID3D12Device8 = device.cast()?;

        let present: ID3D12CommandQueue = {
            let desc = D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
                Flags: D3D12_COMMAND_QUEUE_FLAGS(0),
                NodeMask: 0,
            };

            unsafe { device.CreateCommandQueue(&desc) }?
        };

        let _ = unsafe { present.SetName(w!("Present ID3D12CommandQueue")) };

        Ok(Device(Arc::new(DeviceShared {
            device,
            present,
            _instance: Arc::clone(&self.0),
        })))
    }

    pub fn new_swapchain<F>(&self, device: &Device, surface: Surface) -> Result<Swapchain<F>>
    where
        F: Format,
    {
        let InstanceShared { factory, .. } = &*self.0;
        let DeviceShared { present, .. } = &*device.0;
        let Surface { hwnd, .. } = surface;

        let desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: 0,
            Height: 0,
            Format: match F::FORMAT_TYPE {
                FormatType::Unknown => panic!("Unable to create swapchain with an unknown format"),
                FormatType::R8G8B8A8Srgb => DXGI_FORMAT_R8G8B8A8_UNORM,
                format => format.into(),
            },
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

        let handle =
            unsafe { factory.CreateSwapChainForHwnd(present, hwnd, &desc, Some(&fdesc), None)? };

        let handle: IDXGISwapChain4 = handle.cast()?;

        Ok(Swapchain(Arc::new(SwapchainShared {
            handle,
            present: present.clone(),
            backbuffers: 2,
            _device: Arc::clone(&device.0),
            _instance: Arc::clone(&self.0),
            _marker: PhantomData,
        })))
    }
}

pub struct Surface {
    hwnd: HWND,
    hinstance: HINSTANCE,
    _instance: Arc<InstanceShared>,
}

impl_try_from_rhi_all!(DX12, Surface);

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

    pub fn new_command_pool<K>(&self, _queue: &CommandQueue<K>) -> Result<CommandPool<K>>
    where
        K: QueueKind,
    {
        let DeviceShared { device, .. } = &*self.0;

        let allocator = unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }?;

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

        let list: ID3D12CommandList = unsafe { device.CreateCommandList1(0, kind, flags)? };
        let list: ID3D12GraphicsCommandList6 = list.cast()?;

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
            device.CreateCommittedResource(
                &heap_props,
                heap_flags,
                &buffer_desc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut resource,
            )?
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
                    ImageUsageType::DepthStencil => D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                    ImageUsageType::RenderTarget => D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                    ImageUsageType::Unknown => {
                        panic!("Images cannot be created with an unknown usage")
                    }
                },
            };

            unsafe {
                device.CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut resource,
                )?;
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

impl_try_from_rhi_all!(DX12, Device);

struct SwapchainShared<F: Format> {
    handle: IDXGISwapChain4,
    present: ID3D12CommandQueue,
    backbuffers: u32,
    _device: Arc<DeviceShared>,
    _instance: Arc<InstanceShared>,
    _marker: PhantomData<F>,
}

pub struct Swapchain<F: Format>(Arc<SwapchainShared<F>>);

impl<F: Format> Swapchain<F> {
    pub fn set_fullscreen(&mut self, state: Fullscreen) -> Result<()> {
        let SwapchainShared { handle, .. } = &*self.0;

        let fullscreen = matches!(state, Fullscreen::Fullscreen);
        unsafe { handle.SetFullscreenState(fullscreen, None) }?;
        unsafe { handle.ResizeBuffers(0, 0, 0, DXGI_FORMAT_UNKNOWN, 0) }?;
        Ok(())
    }

    pub fn image(&mut self, _timeout: Duration) -> Result<Option<Image<F, usage::RenderTarget>>> {
        let SwapchainShared { handle, .. } = &*self.0;

        let resource: ID3D12Resource = unsafe { handle.GetBuffer(0) }?;

        Ok(Some(Image {
            resource,
            _marker: PhantomData,
        }))
    }

    pub fn present(&mut self) -> Result<()> {
        let SwapchainShared { handle, .. } = &*self.0;

        let params = DXGI_PRESENT_PARAMETERS {
            DirtyRectsCount: 0,
            pDirtyRects: std::ptr::null_mut(),
            pScrollRect: std::ptr::null_mut(),
            pScrollOffset: std::ptr::null_mut(),
        };

        // TODO: Avoid unwrap
        unsafe { handle.Present1(1, DXGI_PRESENT_RESTART, &params) }.unwrap();
        Ok(())
    }
}

pub struct CommandQueue<K: QueueKind> {
    queue: ID3D12CommandQueue,
    _device: Arc<DeviceShared>,
    _marker: PhantomData<K>,
}

impl_try_from_rhi_all!(DX12, CommandQueue<K: QueueKind>);

pub struct CommandPool<K: QueueKind> {
    allocator: ID3D12CommandAllocator,
    _device: Arc<DeviceShared>,
    _marker: PhantomData<K>,
}

impl_try_from_rhi_all!(DX12, CommandPool<K: QueueKind>);

pub struct CommandList<'a, K: QueueKind, S: State> {
    list: ID3D12GraphicsCommandList6,
    _device: Arc<DeviceShared>,
    _marker: PhantomData<(&'a (), K, S)>,
}

impl<'a> CommandList<'a, Graphics, Initial> {
    pub fn begin(self, pool: &mut CommandPool<Graphics>) -> CommandList<'a, Graphics, Recording> {
        unsafe { self.list.Reset(&pool.allocator, None) }.unwrap();

        CommandList {
            list: self.list,
            _device: self._device,
            _marker: PhantomData,
        }
    }
}

impl<'a> CommandList<'a, queue::Graphics, Recording> {
    pub fn bind_vertex_buffer<T>(&mut self, slot: usize, buf: &'a Buffer<T, usage::VertexBuffer>)
    where
        T: BufferLayout,
    {
    }

    pub fn bind_index_buffer_u16(&mut self, buf: &Buffer<u16, usage::IndexBuffer>) {
        let view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: todo!(),
            SizeInBytes: todo!(),
            Format: DXGI_FORMAT_R16_UINT,
        };

        unsafe { self.list.IASetIndexBuffer(Some(&view)) };
    }

    pub fn bind_index_buffer_u32(&mut self, buf: &Buffer<u32, usage::IndexBuffer>) {
        let view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: todo!(),
            SizeInBytes: todo!(),
            Format: DXGI_FORMAT_R32_UINT,
        };

        unsafe { self.list.IASetIndexBuffer(Some(&view)) };
    }
}

impl<'a> CommandList<'a, queue::Graphics, Executable> {}
pub struct RenderPass {}

pub struct RenderTarget<F: Format> {
    image: Image<F, usage::RenderTarget>,
}

pub struct Buffer<T: BufferLayout, U: BufferUsage> {
    resource: ID3D12Resource2,
    _marker: PhantomData<(T, U)>,
}

impl_try_from_rhi_all!(DX12, Buffer<T: BufferLayout, U: BufferUsage>);

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

            R8 => DXGI_FORMAT_R8_TYPELESS,
            R8Unorm => DXGI_FORMAT_R8_UNORM,
            R8Snorm => DXGI_FORMAT_R8_SNORM,
            R8Uint => DXGI_FORMAT_R8_UINT,
            R8Sint => DXGI_FORMAT_R8_SINT,

            R16 => DXGI_FORMAT_R16_TYPELESS,
            R16Unorm => DXGI_FORMAT_R16_UNORM,
            R16Snorm => DXGI_FORMAT_R16_SNORM,
            R16Uint => DXGI_FORMAT_R16_UINT,
            R16Sint => DXGI_FORMAT_R16_SINT,
            R16Float => DXGI_FORMAT_R16_FLOAT,

            D16Unorm => DXGI_FORMAT_D16_UNORM,

            R32 => DXGI_FORMAT_R32_TYPELESS,
            R32Uint => DXGI_FORMAT_R32_UINT,
            R32Sint => DXGI_FORMAT_R32_SINT,
            R32Float => DXGI_FORMAT_R32_FLOAT,

            D32Float => DXGI_FORMAT_D32_FLOAT,

            R8G8 => DXGI_FORMAT_R8G8_TYPELESS,
            R8G8Unorm => DXGI_FORMAT_R8G8_UNORM,
            R8G8Snorm => DXGI_FORMAT_R8G8_SNORM,
            R8G8Uint => DXGI_FORMAT_R8G8_UINT,
            R8G8Sint => DXGI_FORMAT_R8G8_SINT,

            R16G16 => DXGI_FORMAT_R16G16_TYPELESS,
            R16G16Unorm => DXGI_FORMAT_R16G16_UNORM,
            R16G16Snorm => DXGI_FORMAT_R16G16_SNORM,
            R16G16Uint => DXGI_FORMAT_R16G16_UINT,
            R16G16Sint => DXGI_FORMAT_R16G16_SINT,
            R16G16Float => DXGI_FORMAT_R16G16_FLOAT,

            D24UnormS8Uint => DXGI_FORMAT_D24_UNORM_S8_UINT,

            R32G32 => DXGI_FORMAT_R32G32_TYPELESS,
            R32G32Uint => DXGI_FORMAT_R32G32_UINT,
            R32G32Sint => DXGI_FORMAT_R32G32_SINT,
            R32G32Float => DXGI_FORMAT_R32G32_FLOAT,

            R11G11B10Float => DXGI_FORMAT_R11G11B10_FLOAT,

            R32G32B32 => DXGI_FORMAT_R32G32B32_TYPELESS,
            R32G32B32Uint => DXGI_FORMAT_R32G32B32_UINT,
            R32G32B32Sint => DXGI_FORMAT_R32G32B32_SINT,
            R32G32B32Float => DXGI_FORMAT_R32G32B32_FLOAT,

            R8G8B8A8 => DXGI_FORMAT_R8G8B8A8_TYPELESS,
            R8G8B8A8Unorm => DXGI_FORMAT_R8G8B8A8_UNORM,
            R8G8B8A8Snorm => DXGI_FORMAT_R8G8B8A8_SNORM,
            R8G8B8A8Uint => DXGI_FORMAT_R8G8B8A8_UINT,
            R8G8B8A8Sint => DXGI_FORMAT_R8G8B8A8_SINT,
            R8G8B8A8Srgb => DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,

            R10G10B10A2 => DXGI_FORMAT_R10G10B10A2_TYPELESS,
            R10G10B10A2Unorm => DXGI_FORMAT_R10G10B10A2_UNORM,
            R10G10B10A2Uint => DXGI_FORMAT_R10G10B10A2_UINT,

            R16G16B16A16 => DXGI_FORMAT_R16G16B16A16_TYPELESS,
            R16G16B16A16Unorm => DXGI_FORMAT_R16G16B16A16_UNORM,
            R16G16B16A16Snorm => DXGI_FORMAT_R16G16B16A16_SNORM,
            R16G16B16A16Uint => DXGI_FORMAT_R16G16B16A16_UINT,
            R16G16B16A16Sint => DXGI_FORMAT_R16G16B16A16_SINT,
            R16G16B16A16Float => DXGI_FORMAT_R16G16B16A16_FLOAT,

            R32G32B32A32 => DXGI_FORMAT_R32G32B32A32_TYPELESS,
            R32G32B32A32Uint => DXGI_FORMAT_R32G32B32A32_UINT,
            R32G32B32A32Sint => DXGI_FORMAT_R32G32B32A32_SINT,
            R32G32B32A32Float => DXGI_FORMAT_R32G32B32A32_FLOAT,
        }
    }
}
