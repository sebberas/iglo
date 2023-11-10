use std::ffi::*;
use std::fmt::{Debug, Formatter};
use std::hint::unreachable_unchecked;
use std::num::NonZeroUsize;
use std::ops::{Range, RangeInclusive};
use std::sync::*;
use std::time::Duration;

use ::ash::extensions::*;
use ::ash::vk::{MemoryType, Viewport};
use ::ash::{vk, Entry};
use ::glam::UVec2;
#[cfg(target_os = "windows")]
use ::windows::Win32::Foundation::*;

use crate::rhi::macros::*;
use crate::rhi::{self, *};

impl From<vk::Result> for Error {
    fn from(value: vk::Result) -> Self {
        use vk::Result;

        match value {
            Result::ERROR_OUT_OF_HOST_MEMORY => Error::OutOfHostMemory,
            Result::ERROR_OUT_OF_DEVICE_MEMORY => Error::OutOfDeviceMemory,
            Result::ERROR_INITIALIZATION_FAILED => Error::Unknown,
            Result::ERROR_DEVICE_LOST => Error::DeviceLost,
            Result::ERROR_LAYER_NOT_PRESENT => Error::LayerNotPresent,
            Result::ERROR_EXTENSION_NOT_PRESENT => Error::ExtensionNotPresent,
            Result::ERROR_FEATURE_NOT_PRESENT => Error::FeatureNotPresent,
            Result::ERROR_INCOMPATIBLE_DRIVER => Error::NotSupported,
            Result::ERROR_UNKNOWN => Error::Unknown,

            Result::TIMEOUT => Error::Timeout,

            _ => Error::Unknown,
        }
    }
}

type Result<T> = std::result::Result<T, Error>;

struct InstanceShared {
    entry: Entry,
    instance: ash::Instance,
    physical_devices: Vec<vk::PhysicalDevice>,
}

impl InstanceShared {
    fn raw(&self) -> &ash::Instance {
        &self.instance
    }

    unsafe extern "system" fn debug_callback(severity: vk::DebugUtilsMessageSeverityFlagsEXT, kind: vk::DebugUtilsMessageTypeFlagsEXT, data: *const vk::DebugUtilsMessengerCallbackDataEXT, _: *mut c_void) -> u32 {
        let severity = match severity {
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "VERBOSE",
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "INFO",
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "WARNING",
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "ERROR",
            _ => unreachable!()
        };

        let s = CStr::from_ptr((*data).p_message);
        let s = CString::from(s);

        println!("[VULKAN/{severity}] {}", s.to_str().unwrap());
        vk::FALSE
    }

    fn surface_extension(&self) -> khr::Surface {
        khr::Surface::new(&self.entry, self.raw())
    }

    #[cfg(target_os = "linux")]
    fn xcb_surface_extension(&self) -> khr::XcbSurface {
        khr::XcbSurface::new(&self.entry, self.raw())
    }
    
    #[cfg(target_os = "windows")]
    fn win32_surface_extension(&self) -> khr::Win32Surface {
        khr::Win32Surface::new(&self.entry, self.raw())
    }
}

impl Debug for InstanceShared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InstanceShared")
            .field("physical_devices", &self.physical_devices)
            .finish_non_exhaustive()
    }
}

pub struct Instance(Arc<InstanceShared>);

impl Instance {
    pub fn raw(&self) -> &ash::Instance {
        &self.0.instance
    }
}

impl Instance {
    const ENGINE_NAME: &'static CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"iglo\0") };
    const VALIDATION_NAME: &'static CStr =
        unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };

    pub fn new(debug: bool) -> Result<Self> {
        let entry = Entry::linked();

        let layers = [Self::VALIDATION_NAME].map(CStr::as_ptr);

        let extensions = {
            let extensions = [
                khr::Surface::name(),
                ext::DebugUtils::name(),
                #[cfg(target_os = "linux")]
                {
                    khr::XcbSurface::name()
                },
                #[cfg(target_os = "windows")]
                {
                    khr::Win32Surface::name()
                },
            ];

            extensions.map(CStr::as_ptr)
        };

        let application_info = vk::ApplicationInfo::builder()
            .application_version(0)
            .engine_name(Self::ENGINE_NAME)
            .engine_version(0)
            .api_version(vk::API_VERSION_1_3);

        let create_info = vk::InstanceCreateInfo::builder()
            .flags(vk::InstanceCreateFlags::empty())
            .application_info(&application_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);

        // SAFETY: TODO
        let instance = unsafe { entry.create_instance(&create_info, None)? };

        let physical_devices = {
            let mut physical_devices = Vec::default();

            // SAFETY: TODO
            for physical_device in unsafe { instance.enumerate_physical_devices() }? {
                // SAFETY: TODO
                let (properties, memory_properties, features, layers, extensions) = unsafe {
                    (
                        instance.get_physical_device_properties(physical_device),
                        instance.get_physical_device_memory_properties(physical_device),
                        instance.get_physical_device_features(physical_device),
                        instance.enumerate_device_layer_properties(physical_device)?,
                        instance.enumerate_device_extension_properties(physical_device)?,
                    )
                };

                physical_devices.push(physical_device);
                // TODO: Check Raytracing
            }

            physical_devices
        };

        unsafe {
            use ext::DebugUtils;
            use vk::{DebugUtilsMessengerCreateInfoEXT, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessengerCreateFlagsEXT, DebugUtilsMessageTypeFlagsEXT};

            DebugUtils::new(&entry, &instance).create_debug_utils_messenger(
                &DebugUtilsMessengerCreateInfoEXT {
                    flags: DebugUtilsMessengerCreateFlagsEXT::empty(),
                    message_severity: DebugUtilsMessageSeverityFlagsEXT::VERBOSE | DebugUtilsMessageSeverityFlagsEXT::INFO | DebugUtilsMessageSeverityFlagsEXT::WARNING | DebugUtilsMessageSeverityFlagsEXT::ERROR,
                    message_type: DebugUtilsMessageTypeFlagsEXT::VALIDATION | DebugUtilsMessageTypeFlagsEXT::GENERAL | DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                    pfn_user_callback: Some(InstanceShared::debug_callback),
                    ..Default::default()
                },
                None
            );
        }
        // TODO: Sort Physical Devices

        Ok(Self(Arc::new(InstanceShared {
            entry,
            instance,
            physical_devices,
        })))
    }

    pub fn iter_adapters(&self) -> impl Iterator<Item = Adapter> + '_ {
        let InstanceShared { physical_devices, .. } = &*self.0;

        physical_devices.iter().map(|physical_device| Adapter {
            physical_device: *physical_device,
            _instance: Arc::clone(&self.0),
        })
    }

    pub fn new_surface(&self, window: &Window) -> Result<Surface> {
        #[cfg(target_os = "linux")]
        {
            use crate::os::linux::{Window, WindowExt};

            let (surface, imp) = match window.imp() {
                Window::Xcb(_) => {
                    use ::xcb::Xid;

                    use crate::os::linux::xcb::WindowExt;

                    let create_info = vk::XcbSurfaceCreateInfoKHR::builder()
                        .connection(window.connection() as *const _ as *mut _) // Hopefully safe :o :o
                        .window(window.xid().resource_id());

                    let surface = unsafe { self.0.xcb_surface_extension().create_xcb_surface(&create_info, None)? };
                    let imp = SurfaceImp::Xcb { connection: window.connection(), xid: *window.xid() };
                    (surface, imp)
                }
                Window::Wayland(_) => unimplemented!(),
            };

            Ok(Surface {
                surface,
                imp,
                _instance: Arc::clone(&self.0),
            })
        }

        #[cfg(target_os = "windows")]
        {
            use crate::os::windows::WindowExt;

            let win32_extension = khr::Win32Surface::new(&self.0.entry, &self.0.instance);

            let create_info = vk::Win32SurfaceCreateInfoKHR::builder()
                .flags(vk::Win32SurfaceCreateFlagsKHR::empty())
                .hinstance(window.hinstance().0 as *const _)
                .hwnd(window.hwnd().0 as *const _);

            let surface = unsafe { win32_extension.create_win32_surface(&create_info, None)? };
            let extension = khr::Surface::new(&self.0.entry, &self.0.instance);

            Ok(Surface {
                surface,
                extension,
                _instance: Arc::clone(&self.0),
            })
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            unreachable!()
        }
    }

    /// Creates a new device
    ///
    /// # Arguments
    ///
    /// `props` - A structure containing all the necessary information for
    /// creating a device.
    ///
    /// # Panics
    pub fn new_device(&self, props: VulkanDeviceProps) -> Result<Device> {
        let adapter = if let Some(adapter) = props.adapter {
            adapter
        } else {
            // TODO: Intelligently Select a GPU.
            if let Some(surface) = props.surface {
                // TODO: Select a GPU that supports the surface.
                todo!()
            } else {
                Adapter {
                    physical_device: self.0.physical_devices[0],
                    _instance: Arc::clone(&self.0),
                }
            }
        };

        let physical_device = *adapter.raw();

        QueueFamilyAllocator::pick_queue_families(PickQueueFamilyArgs {
            instance: &self.0,
            physical_device,
            queue_families: &unsafe { self.raw().get_physical_device_queue_family_properties(physical_device) },
            requirements: QueueFamilyRequirements {
                surface: props.surface,
                graphics_queues: props.graphics_queues,
                compute_queues: props.compute_queues,
                transfer_queues: props.transfer_queues }
        });

        let create_info = vk::DeviceCreateInfo::builder()
            .flags(vk::DeviceCreateFlags::empty())
            .queue_create_infos(&[])
            .enabled_extension_names(&[]);

        let device = unsafe { self.raw().create_device(*adapter.raw(), &create_info, None)? };

        Ok(Device(Arc::new(DeviceShared {
            device,
            physical_device,
            queue_families: QueueFamilyAllocator {},
            _instance: Arc::clone(&self.0),
        })))
    }

    pub fn new_swapchain(&self, props: SwapchainProps<'_, Device, Surface>) -> Result<DSwapchain> {
        let SwapchainProps {
            device,
            surface,
            images,
            image_extent,
            image_format,
            present_mode,
        } = props;

        let image_format = match image_format {
            Format::R8G8B8A8Srgb => vk::Format::R8G8B8A8_UNORM,
            format => format.into(),
        };

        let image_extent = vk::Extent2D {
            width: image_extent.x,
            height: image_extent.y,
        };

        let present_mode = match present_mode {
            PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
            PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
            PresentMode::Fifo => vk::PresentModeKHR::FIFO,
            PresentMode::FifoRelaxed => vk::PresentModeKHR::FIFO_RELAXED,
        };

        let extension = khr::Swapchain::new(self.raw(), device.raw());
        let create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(*surface.raw())
            .min_image_count(images.get() as _)
            .image_format(image_format)
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_extent(image_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&[])
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(false)
            .old_swapchain(vk::SwapchainKHR::null());

        let swapchain = unsafe { extension.create_swapchain(&create_info, None)? };
        let images = unsafe { extension.get_swapchain_images(swapchain)? };

        let mut image_views = Vec::with_capacity(images.len());
        for image in &images {
            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build();

            let create_info = vk::ImageViewCreateInfo::builder()
                .flags(vk::ImageViewCreateFlags::empty())
                .image(*image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(image_format)
                .subresource_range(subresource_range)
                .build();

            image_views.push(unsafe { device.raw().create_image_view(&create_info, None) }?);
        }

        Ok(DSwapchain(Arc::new(SwapchainShared {
            extension,
            surface,
            swapchain,
            image_format,
            images,
            image_views,
            _device: Arc::clone(&device.0),
            _instance: Arc::clone(&self.0),
        })))
    }

    // fn setup_queue_create_infos(
    //     queue_families: &[vk::QueueFamilyProperties],
    //     queue_family_indices: &QueueFamilyIndices,
    // ) -> Result<Vec<vk::DeviceQueueCreateInfo>> { use vk::DeviceQueueCreateInfo;

    //     const PRIORITIES: [f32; 32] = [1.0; 32];

    //     let QueueFamilyIndices { graphics, compute, transfer } =
    //         Adapter::find_queue_family_indices(queue_families);

    //     let mut create_infos = Vec::with_capacity(3);
    //     for i in [graphics, compute, transfer].into_iter().flatten() {
    //         let create_info = DeviceQueueCreateInfo::builder()
    //             .queue_family_index(i as _)
    //             .queue_priorities(&PRIORITIES[0..1])
    //             .build();

    //         create_infos.push(create_info);
    //     }

    //     Ok(create_infos)
    // }
}

enum SurfaceImp {
    Xcb {
        connection: &'static xcb::Connection,
        xid: xcb::x::Window
    }
}

impl Debug for SurfaceImp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Xcb { connection, xid } => {
                f.debug_struct("SurfaceImp::Xcb").field("xid", xid).finish_non_exhaustive()
            }        
        }
    }
}

#[derive(Debug)]
pub struct Surface {
    surface: vk::SurfaceKHR,
    imp: SurfaceImp,
    _instance: Arc<InstanceShared>,
}

impl_try_from_rhi_all!(Vulkan, Surface);

impl Surface {
    fn raw(&self) -> &vk::SurfaceKHR {
        &self.surface
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        let surface_ext = khr::Surface::new(&self._instance.entry, &self._instance.instance);
        unsafe { surface_ext.destroy_surface(self.surface, None) };
    }
}

#[derive(Debug, Clone)]
pub struct Adapter {
    physical_device: vk::PhysicalDevice,
    _instance: Arc<InstanceShared>,
}

impl_try_from_rhi_all!(Vulkan, Adapter);

impl Adapter {
    fn raw(&self) -> &vk::PhysicalDevice {
        &self.physical_device
    }
}

impl Adapter {
    pub fn id(&self) -> u32 {
        use ::ash::vk::PhysicalDeviceProperties;
        let Self { _instance, .. } = self;

        // SAFETY: TODO
        let properties = unsafe { _instance.raw().get_physical_device_properties(*self.raw()) };
        let PhysicalDeviceProperties { device_id, .. } = properties;

        device_id
    }

    pub fn name(&self) -> String {
        use ::ash::vk::PhysicalDeviceProperties;
        let Self { _instance, .. } = self;

        // SAFETY: TODO
        let properties = unsafe { _instance.raw().get_physical_device_properties(*self.raw()) };
        let PhysicalDeviceProperties { device_name, .. } = properties;

        // SAFETY: TODO
        unsafe { String::from_utf8_unchecked(device_name.map(|b| b as _).to_vec()) }
    }

    pub fn kind(&self) -> AdapterKind {
        use ::ash::vk::{PhysicalDeviceProperties, PhysicalDeviceType};

        let Self { _instance, .. } = self;

        // SAFETY: TODO
        let properties = unsafe { _instance.raw().get_physical_device_properties(*self.raw()) };
        let PhysicalDeviceProperties { device_type, .. } = properties;

        // SAFETY: TODO
        match device_type {
            PhysicalDeviceType::DISCRETE_GPU => AdapterKind::Discrete,
            PhysicalDeviceType::INTEGRATED_GPU | PhysicalDeviceType::CPU => AdapterKind::Integrated,
            PhysicalDeviceType::VIRTUAL_GPU => AdapterKind::Virtual,
            PhysicalDeviceType::OTHER => AdapterKind::Unknown,
            // SAFETY: TODO
            _ => unsafe { unreachable_unchecked() },
        }
    }
}

type VulkanDeviceProps<'a> = DeviceProps<'a, Adapter, Surface>;

impl<'a> TryFrom<DeviceProps<'a>> for VulkanDeviceProps<'a> {
    type Error = BackendError;
    fn try_from(value: DeviceProps<'a>) -> std::result::Result<Self, Self::Error> {
        let DeviceProps {
            adapter,
            surface,
            graphics_queues,
            compute_queues,
            transfer_queues,
        } = value;

        Ok(VulkanDeviceProps {
            adapter: adapter.map(TryInto::try_into).transpose()?,
            surface: surface.map(TryInto::try_into).transpose()?,
            graphics_queues,
            compute_queues,
            transfer_queues,
        })
    }
}

struct DeviceShared {
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    queue_families: QueueFamilyAllocator,
    _instance: Arc<InstanceShared>,
}

impl std::fmt::Debug for DeviceShared {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceShared")
            .field("physical_device", &self.physical_device)
            .field("queue_families", &self.queue_families)
            .field("_instance", &self._instance)
            .finish_non_exhaustive()
    }
}

impl DeviceShared {
    fn raw(&self) -> &ash::Device {
        &self.device
    }

    fn instance(&self) -> &ash::Instance {
        &self._instance.raw()
    }
}

impl Drop for DeviceShared {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
        }
    }
}

pub struct Device(Arc<DeviceShared>);

impl_try_from_rhi_all!(Vulkan, Device);

impl Device {
    fn raw(&self) -> &ash::Device {
        &self.0.raw()
    }

    fn instance(&self) -> &ash::Instance {
        &self.0.instance()
    }

    fn new_pipeline_layout(&self) -> Result<vk::PipelineLayout> {
        let layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&[])
            .push_constant_ranges(&[])
            .build();

        unsafe { self.raw().create_pipeline_layout(&layout_create_info, None) }.map_err(Into::into)
    }
}

impl Device {
    pub fn queue(&self, operations: Operations) -> Option<DQueue> {
        // let queue_family_indices = self.0.queue_family_indices;
        // let family_index = match operations {
        //     Operations::Graphics => queue_family_indices.graphics.unwrap(),
        //     Operations::Compute => queue_family_indices.compute.unwrap(),
        //     Operations::Transfer => queue_family_indices.transfer.unwrap(),
        // };

        // let queue = unsafe { self.raw().get_device_queue(family_index as _, 0) };

        // Some(DQueue(Arc::new(DQueueShared {
        //     queue,
        //     family_index,
        //     operations,
        //     _device: Arc::clone(&self.0),
        // })))

        todo!()
    }

    pub fn new_command_pool(&self, queue: &DQueue) -> Result<DCommandPool> {
        use vk::{CommandPoolCreateFlags, CommandPoolCreateInfo};

        let create_info = CommandPoolCreateInfo::builder()
            .flags(CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue.0.family_index as _);

        let pool = unsafe { self.raw().create_command_pool(&create_info, None)? };

        Ok(DCommandPool(Arc::new(DCommandPoolShared {
            pool,
            _queue: Arc::clone(&queue.0),
        })))
    }

    pub fn new_command_list(&self, command_pool: &mut DCommandPool) -> Result<DCommandList> {
        use vk::{CommandBufferAllocateInfo, CommandBufferLevel};

        let create_info = CommandBufferAllocateInfo::builder()
            .command_pool(*command_pool.0.raw())
            .level(CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let buffer = unsafe { self.raw().allocate_command_buffers(&create_info)? }[0];

        Ok(DCommandList {
            buffer,
            state: State::Initial,
            _command_pool: Arc::clone(&command_pool.0),
        })
    }

    pub fn new_semaphore(&self, value: u64) -> Result<Semaphore> {
        use vk::{SemaphoreCreateInfo, SemaphoreType, SemaphoreTypeCreateInfo};

        let mut timeline_create_info = SemaphoreTypeCreateInfo::builder()
            .initial_value(value)
            .semaphore_type(SemaphoreType::TIMELINE);

        let create_info = SemaphoreCreateInfo::builder().push_next(&mut timeline_create_info);

        let semaphore = unsafe { self.raw().create_semaphore(&create_info, None)? };

        Ok(Semaphore {
            semaphore,
            _device: Arc::clone(&self.0),
        })
    }

    pub fn new_binary_semaphore(&self) -> Result<BinarySemaphore> {
        let create_info = vk::SemaphoreCreateInfo::builder();
        let semaphore = unsafe { self.raw().create_semaphore(&create_info, None)? };

        Ok(BinarySemaphore {
            semaphore,
            _device: Arc::clone(&self.0),
        })
    }

    pub fn new_fence(&self, signaled: bool) -> Result<Fence> {
        use vk::{FenceCreateFlags, FenceCreateInfo};

        let mut flags = FenceCreateFlags::empty();
        if signaled {
            flags |= FenceCreateFlags::SIGNALED
        }

        let create_info = FenceCreateInfo::builder().flags(flags);
        let fence = unsafe { self.raw().create_fence(&create_info, None)? };

        Ok(Fence {
            fence,
            _device: Arc::clone(&self.0),
        })
    }

    pub fn new_render_pass(&self, attachments: &[Attachment]) -> Result<RenderPass> {
        let attachments: Vec<_> = attachments
            .iter()
            .map(|attachment| match attachment {
                Attachment::Color {
                    format,
                    load_op,
                    store_op,
                    layout,
                } => vk::AttachmentDescription::builder()
                    .format(Into::<_>::into(*format))
                    .load_op(Into::<_>::into(*load_op))
                    .store_op(Into::<_>::into(*store_op))
                    .final_layout(match layout {
                        Layout::Undefined => vk::ImageLayout::UNDEFINED,
                        Layout::Preinitialized => vk::ImageLayout::PREINITIALIZED,
                        Layout::General => vk::ImageLayout::GENERAL,
                        Layout::Present => vk::ImageLayout::PRESENT_SRC_KHR,
                    })
                    .samples(vk::SampleCountFlags::TYPE_1),
                Attachment::DepthStencil {
                    format,
                    depth_load_op,
                    depth_store_op,
                    stencil_load_op,
                    stencil_store_op,
                } => vk::AttachmentDescription::builder()
                    .format(Into::<_>::into(*format))
                    .load_op(Into::<_>::into(depth_load_op.unwrap_or_default()))
                    .store_op(Into::<_>::into(depth_store_op.unwrap_or_default()))
                    .stencil_load_op(Into::<_>::into(stencil_load_op.unwrap_or_default()))
                    .store_op(Into::<_>::into(stencil_store_op.unwrap_or_default()))
                    .final_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                    .samples(vk::SampleCountFlags::TYPE_1),
            })
            .map(|e| e.build())
            .collect();

        let references = [vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build()];

        let subpasses = [vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&references)
            .build()];

        let create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(&subpasses);

        let render_pass = unsafe { self.raw().create_render_pass(&create_info, None)? };

        Ok(RenderPass {
            render_pass,
            _device: Arc::clone(&self.0),
        })
    }

    pub fn new_framebuffer<'a, A>(&self, render_pass: &RenderPass, attachments: A, extent: UVec2) -> Result<Framebuffer>
    where
        A: Iterator<Item = &'a DImageView2D>,
    {
        let attachments: Vec<_> = attachments
            .map(|DImageView2D { kind }| match kind {
                DImageViewKind2D::Owned(a) => *a,
                DImageViewKind2D::Swapchain(i, swapchain) => swapchain.image_views[*i],
            })
            .collect();

        // Find the smallest width and height in the `attachments`.
        let create_info = vk::FramebufferCreateInfo::builder()
            .render_pass(*render_pass.raw())
            .width(extent.x)
            .height(extent.y)
            .attachments(&attachments)
            .layers(1);

        let framebuffer = unsafe { self.raw().create_framebuffer(&create_info, None)? };

        Ok(Framebuffer {
            framebuffer,
            _device: Arc::clone(&self.0),
        })
    }

    pub fn new_shader(&self, bytecode: &[u8]) -> Result<Shader> {
        let code: Vec<_> = bytecode
            .iter()
            .cloned()
            .array_chunks::<4>()
            .map(u32::from_ne_bytes)
            .collect();

        let create_info = vk::ShaderModuleCreateInfo::builder().code(&code).build();

        let shader = unsafe { self.raw().create_shader_module(&create_info, None)? };
        Ok(Shader {
            shader,
            _device: Arc::clone(&self.0),
        })
    }

    pub fn new_pipeline(
        &self,
        state: &PipelineState,
        shaders: &[(Shader, ShaderStage)],
        render_pass: &RenderPass,
    ) -> Result<Pipeline> {
        const ENTRYPOINT: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };

        let layout = self.new_pipeline_layout()?;

        let mut stages = vec![];

        for (shader, stage) in shaders {
            let stage = match stage {
                ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
                ShaderStage::Pixel => vk::ShaderStageFlags::FRAGMENT,
                _ => todo!(),
            };

            stages.push(
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(stage)
                    .module(*shader.raw())
                    .name(ENTRYPOINT)
                    .build(),
            );
        }

        let vertex_input = state.vertex_input.as_ref().unwrap();

        let vertex_binding_descriptions = vertex_input
            .bindings
            .iter()
            .map(|element| {
                vk::VertexInputBindingDescription::builder()
                    .binding(element.binding as _)
                    .stride(element.stride as _)
                    .input_rate(match element.rate {
                        VertexInputRate::Vertex => vk::VertexInputRate::VERTEX,
                        VertexInputRate::Instance => vk::VertexInputRate::INSTANCE,
                    })
                    .build()
            })
            .collect::<Vec<_>>();

        let vertex_attribute_descriptions = vertex_input
            .attributes
            .iter()
            .map(|element| {
                vk::VertexInputAttributeDescription::builder()
                    .location(element.location as _)
                    .binding(element.binding as _)
                    .format(element.format.into())
                    .offset(element.offset as _)
                    .build()
            })
            .collect::<Vec<_>>();

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&vertex_binding_descriptions)
            .vertex_attribute_descriptions(&vertex_attribute_descriptions);

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewports = [vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(1200.0)
            .height(800.0)
            .min_depth(0.0)
            .max_depth(1.0)
            .build()];

        let scissors = [vk::Rect2D::default()];

        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false)
            .depth_bias_constant_factor(0.0)
            .depth_bias_clamp(0.0)
            .depth_bias_slope_factor(0.0)
            .line_width(1.0);

        let multisample_state =
            vk::PipelineMultisampleStateCreateInfo::builder().rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let attachments = [vk::PipelineColorBlendAttachmentState::builder()
            .blend_enable(false)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ZERO)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .build()];

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&attachments)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

        let create_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            // .tessellation_state(tessellation_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            // .depth_stencil_state(depth_stencil_state)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .render_pass(*render_pass.raw())
            .subpass(0)
            .build();

        let pipeline = unsafe {
            self.raw()
                .create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
                .unwrap()[0]
        };

        Ok(Pipeline {
            pipeline,
            layout,
            _device: Arc::clone(&self.0),
        })
    }

    pub fn new_descriptor_pool(&self) -> Result<DescriptorPool> {
        let sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
        }];

        let create_info = vk::DescriptorPoolCreateInfo::builder().max_sets(16).pool_sizes(&sizes);

        let descriptor_pool = unsafe { self.raw().create_descriptor_pool(&create_info, None) }?;
        Ok(DescriptorPool {
            descriptor_pool,
            _device: Arc::clone(&self.0),
        })
    }

    pub fn new_buffer(&self, props: &DBufferProps) -> Result<DBuffer> {
        use vk::{BufferCreateInfo, BufferUsageFlags, MemoryAllocateInfo, MemoryPropertyFlags, SharingMode};

        let usage = match props.usage {
            Usage::Uniform => BufferUsageFlags::UNIFORM_BUFFER,
            Usage::Storage => BufferUsageFlags::STORAGE_BUFFER,
            Usage::Vertex => BufferUsageFlags::VERTEX_BUFFER,
            Usage::Index => BufferUsageFlags::INDEX_BUFFER,
        };

        let buffer_create_info = BufferCreateInfo::builder()
            .size(props.size.get() as _)
            .usage(usage | BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(SharingMode::EXCLUSIVE)
            .queue_family_indices(&[])
            .build();

        let buffer = unsafe { self.raw().create_buffer(&buffer_create_info, None)? };

        let memory_requirements = unsafe { self.raw().get_buffer_memory_requirements(buffer) };

        let memory_properties = unsafe {
            self.instance()
                .get_physical_device_memory_properties(self.0.physical_device)
        };

        let mut memory_type_index = 0;
        for i in 0..memory_properties.memory_type_count {
            let required_memory_property_flags: MemoryPropertyFlags =
                MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT;

            let MemoryType { property_flags, .. } = &memory_properties.memory_types[i as usize];
            if property_flags.contains(required_memory_property_flags) {
                memory_type_index = i;
                break;
            }
        }

        let memory_info = MemoryAllocateInfo::builder()
            .allocation_size(memory_requirements.size)
            .memory_type_index(memory_type_index)
            .build();

        let memory = unsafe { self.raw().allocate_memory(&memory_info, None)? };

        unsafe { self.raw().bind_buffer_memory(buffer, memory, 0)? };

        Ok(DBuffer {
            buffer,
            size: props.size,
            memory: Some(memory),
            _device: Arc::clone(&self.0),
        })
    }

    pub fn wait_idle(&self) -> Result<()> {
        unsafe { self.raw().device_wait_idle() }.map_err(Into::into)
    }
}

struct SwapchainShared {
    extension: khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    image_format: vk::Format,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    surface: Surface,
    _device: Arc<DeviceShared>,
    _instance: Arc<InstanceShared>,
}

impl std::fmt::Debug for SwapchainShared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SwapchainShared")
            .field("swapchain", &self.swapchain)
            .field("image_format", &self.image_format)
            .field("images", &self.images)
            .field("image_views", &self.image_views)
            .field("surface", &self.surface)
            .field("_device", &self._device)
            .field("_instance", &self._instance)
            .finish_non_exhaustive()
    }
}

impl Drop for SwapchainShared {
    fn drop(&mut self) {
        for image_view in &self.image_views {
            unsafe { self._device.raw().destroy_image_view(*image_view, None) };
        }

        unsafe { self.extension.destroy_swapchain(self.swapchain, None) };
    }
}

pub struct DSwapchain(Arc<SwapchainShared>);

impl DSwapchain {
    pub unsafe fn next_image_i_unchecked(
        &mut self,
        semaphore: &mut BinarySemaphore,
        fence: Option<&mut Fence>,
        timeout: Duration,
    ) -> Result<Option<usize>> {
        let SwapchainShared { extension, .. } = &*self.0;

        // let device_mask = self.0._device.physical_device_index + 1;

        // let fence = if let Some(fence) = fence {
        //     *fence.raw()
        // } else {
        //     vk::Fence::null()
        // };

        // let acquire_next_image_info = vk::AcquireNextImageInfoKHR::builder()
        //     .swapchain(*self.raw())
        //     .timeout(timeout.as_nanos() as _)
        //     .semaphore(*semaphore.raw())
        //     .fence(fence)
        //     .device_mask(device_mask as _);

        // let result = unsafe { extension.acquire_next_image2(&acquire_next_image_info)
        // }; if matches!(result, Err(vk::Result::TIMEOUT)) {
        //     return Ok(None);
        // }

        // Ok(Some(result?.0 as _))
        todo!()
    }

    pub unsafe fn image_unchecked(&self, i: usize) -> DImageView2D {
        let _ = self.0.image_views[i]; // Assert an image exists for this index.

        DImageView2D {
            kind: DImageViewKind2D::Swapchain(i, Arc::clone(&self.0)),
        }
    }

    pub fn present<'a>(
        &self,
        image_view: &DImageView2D,
        wait_semaphores: impl IntoIterator<Item = &'a mut BinarySemaphore>,
    ) -> Result<()> {
        let SwapchainShared { extension, _device, .. } = &*self.0;

        let wait_semaphores: Vec<_> = wait_semaphores
            .into_iter()
            .map(|wait_semaphore| *wait_semaphore.raw())
            .collect();

        let swapchains = [*self.raw()];

        let image_indices = if let DImageViewKind2D::Swapchain(image_index, _) = image_view.kind {
            [image_index as _]
        } else {
            unreachable!()
        };

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        // let graphics_index = _device.queue_family_indices.graphics.unwrap();
        // let graphics_queue = unsafe { _device.raw().get_device_queue(graphics_index
        // as _, 0) };

        // let _ = unsafe { extension.queue_present(graphics_queue, &present_info) }?;
        Ok(())
    }

    fn raw(&self) -> &vk::SwapchainKHR {
        &self.0.swapchain
    }
}

impl_try_from_rhi_all!(Vulkan, DSwapchain);

struct DQueueShared {
    queue: vk::Queue,
    family_index: usize,
    operations: Operations,
    _device: Arc<DeviceShared>,
}

impl DQueueShared {
    fn raw(&self) -> &vk::Queue {
        &self.queue
    }
}

pub struct DQueue(Arc<DQueueShared>);

impl_try_from_rhi_all!(Vulkan, DQueue);

impl DQueue {
    fn raw(&self) -> &vk::Queue {
        &self.0.queue
    }

    fn extract_command_buffers(submit_infos: &[VulkanSubmitInfo]) -> Vec<Vec<vk::CommandBuffer>> {
        submit_infos
            .iter()
            .map(|submit_info| {
                submit_info
                    .command_lists
                    .iter()
                    .map(|command_list| *command_list.raw())
                    .collect()
            })
            .collect()
    }

    fn extract_wait_semaphores(
        submit_infos: &[VulkanSubmitInfo],
    ) -> (Vec<Vec<vk::Semaphore>>, Vec<Vec<vk::PipelineStageFlags>>, Vec<Vec<u64>>) {
        let mut semaphores = Vec::with_capacity(4);
        let mut stages = Vec::with_capacity(4);
        let mut values = Vec::with_capacity(4);

        for SubmitInfo { wait_semaphores, .. } in submit_infos {
            let mut inner_semaphores = Vec::with_capacity(4);
            let mut inner_stages: Vec<vk::PipelineStageFlags> = Vec::with_capacity(4);
            let mut inner_values = Vec::with_capacity(4);

            for (wait_semaphore, stage) in wait_semaphores {
                let (semaphore, value) = match wait_semaphore {
                    SemaphoreSubmitInfo::Default(semaphore, value) => (*semaphore.raw(), *value),
                    SemaphoreSubmitInfo::Binary(semaphore) => (*semaphore.raw(), 0),
                };

                inner_semaphores.push(semaphore);
                inner_stages.push((*stage).into());
                inner_values.push(value);
            }

            semaphores.push(inner_semaphores);
            stages.push(inner_stages);
            values.push(inner_values);
        }

        // println!("{semaphores:?}");
        (semaphores, stages, values)
    }

    fn extract_signal_semaphores(submit_infos: &[VulkanSubmitInfo]) -> (Vec<Vec<vk::Semaphore>>, Vec<Vec<u64>>) {
        let mut semaphores = Vec::with_capacity(4);
        let mut values = Vec::with_capacity(4);

        for SubmitInfo { signal_semaphores, .. } in submit_infos {
            let mut inner_semaphores = Vec::with_capacity(4);
            let mut inner_values = Vec::with_capacity(4);

            for signal_semaphore in signal_semaphores {
                let (semaphore, value) = match signal_semaphore {
                    SemaphoreSubmitInfo::Default(semaphore, value) => (*semaphore.raw(), *value),
                    SemaphoreSubmitInfo::Binary(semaphore) => (*semaphore.raw(), 0),
                };

                inner_semaphores.push(semaphore);
                inner_values.push(value);
            }

            semaphores.push(inner_semaphores);
            values.push(inner_values);
        }

        (semaphores, values)
    }
}

pub type VulkanSubmitInfo<'a> =
    SubmitInfo<'a, vulkan::DCommandList, vulkan::Semaphore, vulkan::PipelineStage, vulkan::BinarySemaphore>;

impl DQueue {
    pub fn operations(&self) -> Operations {
        self.0.operations
    }

    /// # Safety
    pub unsafe fn submit_unchecked(
        &mut self,
        submit_infos: &[VulkanSubmitInfo],
        fence: Option<&mut Fence>,
    ) -> Result<()> {
        use vk::{PipelineStageFlags, SubmitInfo, TimelineSemaphoreSubmitInfo};
        let submit_infos_len = submit_infos.len();

        let (wait_semaphores, wait_stages, wait_values) = Self::extract_wait_semaphores(submit_infos);

        let command_buffers = Self::extract_command_buffers(submit_infos);
        let (signal_semaphores, signal_values) = Self::extract_signal_semaphores(submit_infos);

        let mut semaphore_submit_infos = Vec::with_capacity(submit_infos.len());
        for i in 0..submit_infos_len {
            // println!(
            //     "Timeline Semaphore {:?}: {:?}",
            //     self.raw(),
            //     wait_semaphores[i]
            // );

            let semaphore_submit_info = TimelineSemaphoreSubmitInfo::builder()
                .wait_semaphore_values(&wait_values[i])
                .signal_semaphore_values(&signal_values[i])
                .build();

            semaphore_submit_infos.push(semaphore_submit_info);
        }

        let mut submit_infos = Vec::with_capacity(submit_infos_len);
        for i in 0..submit_infos_len {
            let submit_info = SubmitInfo::builder()
                .wait_semaphores(&wait_semaphores[i])
                .wait_dst_stage_mask(&wait_stages[i])
                .command_buffers(&command_buffers[i])
                .signal_semaphores(&signal_semaphores[i])
                .push_next(&mut semaphore_submit_infos[i])
                .build();

            submit_infos.push(submit_info);
        }

        // println!("{submit_infos:?}");

        let fence = fence.map(|fence| *fence.raw()).unwrap_or_default();

        let device = &self.0._device.raw();
        unsafe { device.queue_submit(*self.raw(), &submit_infos, fence) }.map_err(Into::into)
    }

    pub fn wait_idle(&mut self) -> Result<()> {
        let device = self.0._device.raw();
        unsafe { device.queue_wait_idle(*self.0.raw()) }.map_err(Into::into)
    }
}

pub type VulkanSemaphoreSubmitInfo<'a> = SemaphoreSubmitInfo<'a, vulkan::Semaphore, vulkan::BinarySemaphore>;

impl<'a> TryFrom<SemaphoreSubmitInfo<'a>> for VulkanSemaphoreSubmitInfo<'a> {
    type Error = BackendError;

    fn try_from(value: SemaphoreSubmitInfo<'a>) -> std::result::Result<Self, Self::Error> {
        use SemaphoreSubmitInfo::*;

        match value {
            Default(semaphore, value) => Ok(Default(semaphore.try_into()?, value)),
            Binary(semaphore) => Ok(Binary(semaphore.try_into()?)),
        }
    }
}

impl<'a> TryFrom<rhi::SubmitInfo<'a>> for VulkanSubmitInfo<'a> {
    type Error = BackendError;

    fn try_from(value: rhi::SubmitInfo<'a>) -> std::result::Result<Self, Self::Error> {
        let command_lists = value
            .command_lists
            .into_iter()
            .map(TryInto::try_into)
            .collect::<std::result::Result<_, _>>()?;

        let wait_semaphores: Vec<_> = value
            .wait_semaphores
            .into_iter()
            .map(|(semaphore, stage)| semaphore.try_into().map(|s| (s, stage)))
            .collect::<std::result::Result<_, _>>()?;

        let signal_semaphores: Vec<_> = value
            .signal_semaphores
            .into_iter()
            .map(TryInto::try_into)
            .collect::<std::result::Result<_, _>>()?;

        Ok(SubmitInfo {
            command_lists,
            wait_semaphores,
            signal_semaphores,
        })
    }
}

impl From<rhi::PipelineStage> for vk::PipelineStageFlags {
    fn from(value: rhi::PipelineStage) -> Self {
        match value {
            PipelineStage::Transfer => vk::PipelineStageFlags::TRANSFER,
            PipelineStage::ColorAttachmentOutput => vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        }
    }
}

struct DCommandPoolShared {
    pool: vk::CommandPool,
    _queue: Arc<DQueueShared>,
}

impl DCommandPoolShared {
    fn raw(&self) -> &vk::CommandPool {
        &self.pool
    }
}

impl Drop for DCommandPoolShared {
    fn drop(&mut self) {
        let device = self._queue._device.raw();
        unsafe { device.destroy_command_pool(self.pool, None) };
    }
}

pub struct DCommandPool(Arc<DCommandPoolShared>);

impl DCommandPool {
    fn raw(&self) -> &vk::CommandPool {
        &self.0.pool
    }
}

impl_try_from_rhi_all!(Vulkan, DCommandPool);

impl DCommandPool {
    pub fn operations(&self) -> Operations {
        self.0._queue.operations
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Initial,
    Recording,
    Executable,
}

pub struct DCommandList {
    buffer: vk::CommandBuffer,
    state: State,
    _command_pool: Arc<DCommandPoolShared>,
}

impl_try_from_rhi_all!(Vulkan, DCommandList);

impl DCommandList {
    /// Returns the operations supported by this command list
    pub fn operations(&self) -> Operations {
        self._command_pool._queue.operations
    }

    /// Returns the current state of this command list
    pub fn state(&self) -> State {
        self.state
    }

    pub unsafe fn reset_unchecked(&mut self, command_pool: &mut DCommandPool) -> Result<()> {
        assert_eq!(self._command_pool.raw(), command_pool.raw());
        let device = self.device();
        device.reset_command_buffer(*self.raw(), vk::CommandBufferResetFlags::empty())?;
        self.state = State::Initial;
        Ok(())
    }

    /// Begins recording for this command list
    ///
    /// # Panics
    ///
    /// Panics if the supplied pool is not the same pool that was used when
    /// creating this command list.
    ///
    /// # Safety
    ///
    /// - The command list must be in the initial state.
    ///
    /// - Only a single command list in a pool can be recording at any given
    ///   time. This means that until `end_unchecked` is called, the pool
    ///   backing this command list must not begin recording for any of its
    ///   other command lists.
    pub unsafe fn begin_unchecked(&mut self, command_pool: &DCommandPool) -> Result<()> {
        use vk::{CommandBufferBeginInfo, CommandBufferInheritanceInfo, CommandBufferUsageFlags};

        assert_eq!(self._command_pool.raw(), command_pool.raw(),);

        let inheritance_info = CommandBufferInheritanceInfo::builder();

        let begin_info = CommandBufferBeginInfo::builder()
            .flags(CommandBufferUsageFlags::empty())
            .inheritance_info(&inheritance_info);

        self.device().begin_command_buffer(*self.raw(), &begin_info)?;

        self.state = State::Recording;
        Ok(())
    }

    /// # Safety
    ///
    /// - The command list must be in the recording state.
    ///
    /// - The command list must not have begun a render pass without ending it.
    pub unsafe fn begin_render_pass_unchecked(&mut self, render_pass: &RenderPass, framebuffer: &mut Framebuffer) {
        let device = self.device();

        let clear_value = [vk::ClearValue {
            color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 0.0] },
        }];

        let begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(*render_pass.raw())
            .framebuffer(*framebuffer.raw())
            .render_area(vk::Rect2D {
                extent: vk::Extent2D { width: 640, height: 480 },
                ..Default::default()
            });
        // .render_area(vk::Rect2D {
        //     extent: vk::Extent2D {
        //         width: 640,
        //         height: 480,
        //     },
        //     ..Default::default()
        // });
        // .clear_values(&clear_value);

        unsafe { device.cmd_begin_render_pass(*self.raw(), &begin_info, vk::SubpassContents::INLINE) };
    }

    pub unsafe fn bind_pipeline_unchecked(&mut self, pipeline: &Pipeline) {
        let device = self.device();
        unsafe { device.cmd_bind_pipeline(*self.raw(), vk::PipelineBindPoint::GRAPHICS, *pipeline.raw()) };
    }

    pub unsafe fn bind_vertex_buffers_unchecked<'a, I>(&mut self, buffers: I)
    where
        I: IntoIterator<Item = &'a DBuffer>,
    {
        let buffers: Vec<_> = buffers.into_iter().map(|buffer| *buffer.raw()).collect();
        let offsets = vec![0; buffers.len()];

        unsafe {
            self.device()
                .cmd_bind_vertex_buffers(*self.raw(), 0, &buffers, &offsets)
        };
    }

    pub unsafe fn set_viewport_unchecked(&mut self, viewport: &ViewportState) {
        let ViewportState { position, extent, depth, scissor } = viewport;

        let viewport = vk::Viewport::builder()
            .x(position.x)
            .y(position.y)
            .width(extent.x)
            .height(extent.y)
            .min_depth(*depth.start())
            .max_depth(*depth.end())
            .build();

        unsafe { self.device().cmd_set_viewport(*self.raw(), 0, &[viewport]) };

        if let Some(ScissorState { offset, extent }) = scissor {
            let scissor = vk::Rect2D {
                offset: vk::Offset2D {
                    x: offset.x as _,
                    y: offset.y as _,
                },
                extent: vk::Extent2D {
                    width: extent.x,
                    height: extent.y,
                },
            };

            unsafe { self.device().cmd_set_scissor(*self.raw(), 0, &[scissor]) };
        }
    }

    pub unsafe fn draw_unchecked(&mut self, nvertices: usize, ninstances: usize) {
        let device = self.device();
        unsafe { device.cmd_draw(*self.raw(), nvertices as _, ninstances as _, 0, 0) };
    }

    /// # Safety
    ///
    /// - The command list must be in the recording state.
    ///
    /// - The command list must have a render pass that is not ended.
    pub unsafe fn end_render_pass_unchecked(&mut self) {
        let device = self.device();
        unsafe { device.cmd_end_render_pass(*self.raw()) };
        self.state = State::Executable;
    }

    /// End recording for this command list
    ///
    /// # Safety
    ///
    /// - The command list must be in the recording state.
    pub unsafe fn end_unchecked(&mut self) -> Result<()> {
        self.device().end_command_buffer(*self.raw())?;
        Ok(())
    }

    pub unsafe fn copy_buffer_unchecked(
        &mut self,
        _command_pool: &mut DCommandPool,
        src: &DBuffer,
        dst: &DBuffer,
        size: usize,
    ) {
        let regions = [vk::BufferCopy::builder()
            .src_offset(0)
            .dst_offset(0)
            .size(size as _)
            .build()];

        unsafe {
            self.device()
                .cmd_copy_buffer(*self.raw(), *src.raw(), *dst.raw(), &regions)
        }
    }

    pub unsafe fn copy_image_unchecked(&mut self, command_pool: &mut DCommandPool, src: &DImage2D, dst: &DImage2D) {}

    fn raw(&self) -> &vk::CommandBuffer {
        &self.buffer
    }

    fn device(&self) -> &ash::Device {
        self._command_pool._queue._device.raw()
    }
}

impl Drop for DCommandList {
    fn drop(&mut self) {
        let device = self._command_pool._queue._device.raw();
        unsafe { device.free_command_buffers(*self._command_pool.raw(), &[self.buffer]) };
    }
}

pub struct Semaphore {
    semaphore: vk::Semaphore,
    _device: Arc<DeviceShared>,
}

impl_try_from_rhi_all!(Vulkan, Semaphore);

impl SemaphoreApi for Semaphore {}

impl Semaphore {
    pub fn wait(&mut self, value: u64, timeout: Duration) -> Result<bool> {
        let device = self._device.raw();

        let semaphores = [*self.raw()];
        let values = [value];

        let wait_info = vk::SemaphoreWaitInfo::builder().semaphores(&semaphores).values(&values);

        match unsafe { device.wait_semaphores(&wait_info, timeout.as_nanos() as _) } {
            Ok(_) => Ok(true),
            Err(e) if e == vk::Result::TIMEOUT => Ok(false),
            Err(e) => Err(Error::from(e)),
        }
    }

    pub fn signal(&mut self, value: u64) -> Result<()> {
        let device = self._device.raw();

        let signal_info = vk::SemaphoreSignalInfo::builder().semaphore(*self.raw()).value(value);

        unsafe { device.signal_semaphore(&signal_info) }.map_err(Into::into)
    }

    pub fn reset(&mut self, value: u64) -> Result<()> {
        let device = self._device.raw();

        todo!()
    }

    /// Executes `f` when the value of this semaphore changes.
    pub fn on_signal(&mut self, f: impl Fn(u64) + 'static) {
        todo!()
    }

    /// Executes `f` when the value of this semaphore reaches `value`.
    pub fn on_value(&mut self, value: u64, f: impl FnOnce() + 'static) {}

    fn raw(&self) -> &vk::Semaphore {
        &self.semaphore
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe { self._device.raw().destroy_semaphore(*self.raw(), None) };
    }
}

pub struct BinarySemaphore {
    semaphore: vk::Semaphore,
    _device: Arc<DeviceShared>,
}

impl BinarySemaphore {
    pub fn raw(&self) -> &vk::Semaphore {
        &self.semaphore
    }
}

impl_try_from_rhi_all!(Vulkan, BinarySemaphore);

impl Drop for BinarySemaphore {
    fn drop(&mut self) {
        unsafe { self._device.raw().destroy_semaphore(*self.raw(), None) };
    }
}

pub struct Fence {
    fence: vk::Fence,
    _device: Arc<DeviceShared>,
}

impl FenceApi for Fence {
    fn wait(&self, timeout: Duration) -> Result<bool> {
        let device = self._device.raw();

        let timeout = timeout.as_nanos() as _;
        match unsafe { device.wait_for_fences(&[*self.raw()], false, timeout) } {
            Ok(_) => Ok(true),
            Err(e) if e == vk::Result::TIMEOUT => Ok(false),
            Err(e) => Err(Error::from(e)),
        }
    }

    fn signaled(&self) -> Result<bool> {
        let device = self._device.raw();
        unsafe { device.get_fence_status(*self.raw()) }.map_err(Into::into)
    }

    fn reset(&mut self) -> Result<()> {
        let device = self._device.raw();
        unsafe { device.reset_fences(&[*self.raw()]) }.map_err(Into::into)
    }

    fn leak(mut self) {
        let device = self._device.raw();

        // If the fence has already been signaled we can just destroy it.
        if !unsafe { device.get_fence_status(*self.raw()) }.unwrap_or_default() {
            self.fence = vk::Fence::null();
        }
    }
}

impl Fence {
    fn raw(&self) -> &vk::Fence {
        &self.fence
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        let device = self._device.raw();

        // We don't want to destroy the fence if it has been leaked.
        if *self.raw() != vk::Fence::null() {
            // TODO:
            // We should store the fence together with the device to make sure it is
            // properly dropped and vulkan doesn't yell at us.
            //
            // If the fence has been leaked we could possibly recycle
            // that fence using a channel.
            let _ = unsafe { device.wait_for_fences(&[*self.raw()], false, u64::MAX) };
            unsafe { device.destroy_fence(*self.raw(), None) };
        }
    }
}

impl_try_from_rhi_all!(Vulkan, Fence);

pub struct DImage2D {
    image: vk::Image,
    _device: Arc<DeviceShared>,
}

impl Drop for DImage2D {
    fn drop(&mut self) {
        unsafe { self._device.raw().destroy_image(self.image, None) };
    }
}

impl_try_from_rhi_all!(Vulkan, DImage2D);

#[derive(Debug, Clone)]
pub struct DImageView2D {
    kind: DImageViewKind2D,
}

#[derive(Debug, Clone)]
enum DImageViewKind2D {
    Swapchain(usize, Arc<SwapchainShared>),
    Owned(vk::ImageView),
}

impl_try_from_rhi_all!(Vulkan, DImageView2D);

impl From<AttachmentLoadOp> for vk::AttachmentLoadOp {
    fn from(value: AttachmentLoadOp) -> Self {
        match value {
            AttachmentLoadOp::Load => Self::LOAD,
            AttachmentLoadOp::Clear => Self::CLEAR,
            AttachmentLoadOp::DontCare => Self::DONT_CARE,
        }
    }
}

impl From<AttachmentStoreOp> for vk::AttachmentStoreOp {
    fn from(value: AttachmentStoreOp) -> Self {
        match value {
            AttachmentStoreOp::Store => Self::STORE,
            AttachmentStoreOp::DontCare => Self::DONT_CARE,
        }
    }
}

pub struct RenderPassProps<'a> {
    pub attachments: &'a [Attachment],
}

pub struct RenderPass {
    render_pass: vk::RenderPass,
    _device: Arc<DeviceShared>,
}

impl_try_from_rhi_all!(Vulkan, RenderPass);

impl RenderPass {
    pub fn attachment(&self, i: usize) -> Option<Attachment> {
        todo!()
    }

    // pub fn subpass(&self, i: usize) -> Option<Subpass> {
    //     todo!()
    // }

    fn raw(&self) -> &vk::RenderPass {
        &self.render_pass
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        let device = self._device.raw();
        unsafe { device.destroy_render_pass(self.render_pass, None) }
    }
}

pub struct Framebuffer {
    framebuffer: vk::Framebuffer,
    _device: Arc<DeviceShared>,
}

impl_try_from_rhi_all!(Vulkan, Framebuffer);

impl Framebuffer {
    fn raw(&self) -> &vk::Framebuffer {
        &self.framebuffer
    }
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        let device = self._device.raw();
        unsafe { device.destroy_framebuffer(self.framebuffer, None) };
    }
}

pub struct Shader {
    shader: vk::ShaderModule,
    _device: Arc<DeviceShared>,
}

impl Shader {
    pub fn raw(&self) -> &vk::ShaderModule {
        &self.shader
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        // SAFETY
        unsafe { self._device.raw().destroy_shader_module(self.shader, None) };
    }
}

// TODO: Maybe should be specified when creating the shaders?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex,
    TessellationControl,
    TessellationEvaluation,
    Geometry,
    Pixel,
    Compute,
}

impl From<ShaderStage> for vk::ShaderStageFlags {
    fn from(value: ShaderStage) -> Self {
        match value {
            ShaderStage::Vertex => Self::VERTEX,
            ShaderStage::TessellationControl => Self::TESSELLATION_CONTROL,
            ShaderStage::TessellationEvaluation => Self::TESSELLATION_EVALUATION,
            ShaderStage::Geometry => Self::GEOMETRY,
            ShaderStage::Pixel => Self::FRAGMENT,
            ShaderStage::Compute => Self::COMPUTE,
        }
    }
}

pub struct PipelineShaderStage<'a, 'b> {
    pub stage: ShaderStage,
    pub shader: &'a Shader,
    pub entrypoint: &'b str,
}

pub struct PipelineLayout {}

pub struct PipelineProps<'a, 'b> {
    pub stages: &'b [PipelineShaderStage<'a, 'b>],
    pub render_pass: &'b RenderPass,
    pub layout: PipelineLayout,
}

impl<'a, 'b> AsRef<Self> for PipelineProps<'a, 'b> {
    fn as_ref(&self) -> &Self {
        self
    }
}

pub struct Pipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    _device: Arc<DeviceShared>,
}

impl Pipeline {
    fn raw(&self) -> &vk::Pipeline {
        &self.pipeline
    }
}

impl<'a> Drop for Pipeline {
    fn drop(&mut self) {
        let device = self._device.raw();
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
        };
    }
}

impl_try_from_rhi_all!(Vulkan, Pipeline);

pub struct DescriptorPool {
    descriptor_pool: vk::DescriptorPool,
    _device: Arc<DeviceShared>,
}

impl_try_from_rhi_all!(Vulkan, DescriptorPool);

pub struct DescriptorSet {
    _device: Arc<DeviceShared>,
}

pub struct DBuffer {
    buffer: vk::Buffer,
    size: NonZeroUsize,
    memory: Option<vk::DeviceMemory>,
    _device: Arc<DeviceShared>,
}

impl DBuffer {
    pub unsafe fn map_unchecked(&self) -> Result<&mut [u8]> {
        const FLAGS: vk::MemoryMapFlags = vk::MemoryMapFlags::empty();

        let memory = unsafe { self.memory.unwrap_unchecked() };
        let size: vk::DeviceSize = self.size.get() as _;
        let data = unsafe { self._device.raw().map_memory(memory, 0, size, FLAGS)? };

        Ok(std::slice::from_raw_parts_mut(data as *mut _, self.size.get()))
    }

    pub unsafe fn unmap_unchecked(&self) {
        let memory = unsafe { self.memory.unwrap_unchecked() };
        self._device.raw().unmap_memory(memory);
    }

    pub fn raw(&self) -> &vk::Buffer {
        &self.buffer
    }
}

impl_try_from_rhi_all!(Vulkan, DBuffer);

impl Drop for DBuffer {
    fn drop(&mut self) {
        let device = self._device.raw();
        if let Some(memory) = self.memory {
            unsafe { device.free_memory(memory, None) }
        }

        unsafe { device.destroy_buffer(*self.raw(), None) };
    }
}

impl From<Format> for vk::Format {
    fn from(value: Format) -> Self {
        match value {
            Format::R8G8B8A8Unorm => Self::R8G8B8A8_UNORM,
            Format::R8G8B8A8Srgb => Self::R8G8B8A8_SRGB,

            Format::R32G32B32A32Float => Self::R32G32B32A32_SFLOAT,
            _ => Self::UNDEFINED,
        }
        // match value {
        //     Format::Unknown => Self::UNDEFINED,
        //     Format::R8 => todo!(),
        //     Format::R8Unorm => Self::R8_UNORM,
        //     Format::R8Snorm => Self::R8_SNORM,
        //     Format::R8Uint => Self::R8_UINT,
        //     Format::R8Sint => Self::R8_SINT,
        //     Format::R16 => todo!(),
        //     Format::R16Unorm => Self::R16_UNORM,
        //     Format::R16Snorm => Self::R16_SNORM,
        //     Format::R16Uint => Self::R16_UINT,
        //     Format::R16Sint => Self::R16_SINT,
        //     Format::R16Float => Self::R16_SFLOAT,
        //     Format::D16Unorm => Self::D16_UNORM,
        //     Format::R32 => todo!(),
        //     Format::R32Uint => Self::R32_UINT,
        //     Format::R32Sint => Self::R32_SINT,
        //     Format::R32Float => Self::R32_SFLOAT,
        //     Format::D32Float => Self::D32_SFLOAT,
        //     Format::R8G8 => todo!(),
        //     Format::R8G8Unorm => Self::R8G8_UNORM,
        //     Format::R8G8Snorm => Self::R8G8_SNORM,
        //     Format::R8G8Uint => Self::R8G8_UINT,
        //     Format::R8G8Sint => Self::R8G8_SINT,
        //     Format::R16G16 => todo!(),
        //     Format::R16G16Unorm => Self::R16G16_UNORM,
        //     Format::R16G16Snorm => Self::R16G16_SNORM,
        //     Format::R16G16Uint => Self::R16G16_UINT,
        //     Format::R16G16Sint => Self::R16G16_SINT,
        //     Format::R16G16Float => Self::R16G16_SFLOAT,
        //     Format::D24UnormS8Uint => Self::D24_UNORM_S8_UINT,
        //     Format::R32G32 => todo!(),
        //     Format::R32G32Uint => Self::R32G32_UINT,
        //     Format::R32G32Sint => Self::R32G32_SINT,
        //     Format::R32G32Float => Self::R32G32_SFLOAT,
        //     Format::R11G11B10Float => todo!(),
        //     Format::R32G32B32 => todo!(),
        //     Format::R32G32B32Uint => Self::R32G32B32_UINT,
        //     Format::R32G32B32Sint => Self::R32G32B32_SINT,
        //     Format::R32G32B32Float => Self::R32G32B32_SFLOAT,
        //     Format::R8G8B8A8 => todo!(),
        //     Format::R8G8B8A8Unorm => Self::R8G8B8A8_UNORM,
        //     Format::R8G8B8A8Snorm => Self::R8G8B8A8_SNORM,
        //     Format::R8G8B8A8Uint => Self::R8G8B8A8_UINT,
        //     Format::R8G8B8A8Sint => Self::R8G8B8A8_SINT,
        //     Format::R8G8B8A8Srgb => Self::R8G8B8A8_SRGB,
        //     Format::R10G10B10A2 => todo!(),
        //     Format::R10G10B10A2Unorm => todo!(),
        //     Format::R10G10B10A2Uint => todo!(),
        //     Format::R16G16B16A16 => todo!(),
        //     Format::R16G16B16A16Unorm => Self::R16G16B16A16_UNORM,
        //     Format::R16G16B16A16Snorm => Self::R16G16B16A16_SNORM,
        //     Format::R16G16B16A16Uint => Self::R16G16B16A16_UINT,
        //     Format::R16G16B16A16Sint => Self::R16G16B16A16_SINT,
        //     Format::R16G16B16A16Float => Self::R16G16B16A16_SFLOAT,
        //     Format::R32G32B32A32 => todo!(),
        //     Format::R32G32B32A32Uint => Self::R32G32B32A32_UINT,
        //     Format::R32G32B32A32Sint => Self::R32G32B32A32_SINT,
        //     Format::R32G32B32A32Float => Self::R32G32B32A32_SFLOAT,
        // }
    }
}

#[derive(Debug, Clone)]
struct QueueFamilyRequirements<'a> {
    surface: Option<&'a Surface>,
    graphics_queues: Option<Range<usize>>,
    compute_queues: (Option<Range<usize>>, bool),
    transfer_queues: (Option<Range<usize>>, bool),
}

#[derive(Debug, Default)]
struct QueueFamily {
    index: usize,
    count: usize,
}

#[derive(Debug, Default)]
struct QueueFamilySet {
    /// Present usually is the graphics queue or one of the other two queues,
    /// but it is not guaranteed. If there is no surface passed into the
    /// requirements present will simply be empty and an assertion will be
    /// thrown when trying to create the swapchain.
    /// The last field in the option, indicates whether the queue family is
    /// dedicated. If no present queue was found, then the option will
    /// contain None. If no surface has been supplied, present will contain
    /// None.
    present: Option<(QueueFamily, bool)>,
    /// Dedicated Graphics + Compute + Transfer
    graphics: Option<QueueFamily>,
    /// Dedicated Compute + Transfer
    /// The last field in the option, indicates whether the queue family is
    /// dedicated.
    compute: Option<(QueueFamily, bool)>,
    /// The last field in the option, indicates whether the queue family is
    /// dedicated. The option will contain None, if the requirements
    /// indicate a dedicated transfer queue is required, but none was found.
    /// Queue Families can share queues, meaning if the physical device contains
    /// 5 graphics queues and only 2 are required for graphics ops. then the
    /// rest of them can go to the transfer queue. Resource Sharing barries
    /// are automatically no-opped if it can be determined automatically that
    /// the resources belong to the same queue.
    transfer: Option<(QueueFamily, bool)>,
}

struct PickQueueFamilyArgs<'a> {
    instance: &'a Arc<InstanceShared>,
    physical_device: vk::PhysicalDevice,
    queue_families: &'a [vk::QueueFamilyProperties],
    requirements: QueueFamilyRequirements<'a>
}

#[derive(Debug)]
struct QueueFamilyAllocator {}

impl QueueFamilyAllocator {
    pub fn new(physical_device: vk::PhysicalDevice) -> Self {
        todo!()
    }

    /// Picks the best queue families from the ones supplied in `families`
    ///
    /// The algorithm first looks for dedicated queue families for the graphics,
    /// compute and transfer and attempts to use the graphics queue as a present
    /// queue. Afterwards depending on dedicated compute and transfer are
    /// required, either fills out these with remaining available queues from
    /// the graphics queue or compute queue.
    /// If no range is specified, maximum 4 queues are enabled for the given
    /// family.
    pub fn pick_queue_families(args: PickQueueFamilyArgs) -> Option<QueueFamilySet> {
        use vk::{QueueFamilyProperties, QueueFlags};
        let PickQueueFamilyArgs {instance, physical_device, queue_families, requirements} = args;

        let mut set = QueueFamilySet::default();
        for (i, QueueFamilyProperties { queue_flags, queue_count, .. }) in queue_families.iter().enumerate() {
            if let Some(surface) = &requirements.surface {
                #[cfg(target_os = "linux")]
                unsafe {
                    match surface.imp {
                        SurfaceImp::Xcb { mut connection, xid } => {
                            use ::xcb::{Xid};

                            instance.xcb_surface_extension().get_physical_device_xcb_presentation_support(
                                physical_device,
                                i as _,
                                std::mem::transmute(&mut connection),
                                xid.resource_id()
                            );
                        }
                    }
                }

            let Range { start, end } = requirements.graphics_queues.clone().unwrap_or(1..5);
            if queue_flags.contains(QueueFlags::GRAPHICS | QueueFlags::COMPUTE) {
                    unsafe {
                        instance.surface_extension().get_physical_device_surface_support(physical_device, i as _, *surface.raw()).unwrap()
                    };
                }

                if set.graphics.is_none() && start >= *queue_count as _ {
                    set.graphics = Some(QueueFamily {
                        index: i,
                        count: (*queue_count as usize).min(end - 1),
                    });

                    continue;
                } else if let Some(graphics) = set.graphics.as_mut() && start >= *queue_count as _ {
                    if graphics.count < *queue_count as _ {
                        *graphics = QueueFamily {
                            index: i,
                            count: (*queue_count as usize).min(end - 1)
                        } 
                    }

                    continue;
                }
            }
        }

        None
    }
}
