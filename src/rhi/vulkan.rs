use std::ffi::*;
use std::mem::ManuallyDrop;
use std::ops::{Range, RangeInclusive};
use std::sync::*;
use std::time::Duration;

use ::ash::extensions::*;
use ::ash::{vk, Entry};
use ::glam::UVec2;
#[cfg(target_os = "windows")]
use ::windows::Win32::Foundation::*;
use ::windows::Win32::System::Threading::*;
use ::windows::Win32::System::WindowsProgramming::*;
use ash::vk::Viewport;

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

const VALIDATION_LAYER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };

struct InstanceShared {
    entry: Entry,
    instance: ash::Instance,
    physical_devices: Vec<vk::PhysicalDevice>,
}

impl InstanceShared {
    fn raw(&self) -> &ash::Instance {
        &self.instance
    }
}

impl std::fmt::Debug for InstanceShared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InstanceShared")
            .field("physical_devices", &self.physical_devices)
            .finish_non_exhaustive()
    }
}

pub struct Instance(Arc<InstanceShared>);

impl Instance {
    const ENGINE_NAME: &'static CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"iglo\0") };

    pub fn new(debug: bool) -> Result<Self> {
        let entry = Entry::linked();

        let application_info = vk::ApplicationInfo::builder()
            .application_version(0)
            .engine_name(Self::ENGINE_NAME)
            .engine_version(0)
            .api_version(vk::API_VERSION_1_3);

        let enabled_layer_names = debug
            .then(|| vec![VALIDATION_LAYER_NAME.as_ptr()])
            .unwrap_or_default();

        let enabled_extension_names = [
            khr::Surface::name().as_ptr(),
            khr::Win32Surface::name().as_ptr(),
        ];

        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&application_info)
            .enabled_layer_names(&enabled_layer_names)
            .enabled_extension_names(&enabled_extension_names);

        let instance = unsafe { entry.create_instance(&create_info, None)? };
        let physical_devices = Self::find_physical_devices(&instance)?;

        Ok(Self(Arc::new(InstanceShared {
            entry,
            instance,
            physical_devices,
        })))
    }

    pub fn new_surface(&self, window: &Window) -> Result<Surface> {
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

    pub fn enumerate_adapters(&self) -> impl Iterator<Item = Adapter> + '_ {
        let physical_devices = &self.0.physical_devices;

        let to_adapter = |(physical_device_index, _)| Adapter {
            physical_device_index,
            _instance: Arc::clone(&self.0),
        };

        physical_devices.iter().enumerate().map(to_adapter)
    }

    /// Creates a new device
    ///
    /// # Arguments
    ///
    /// `props` - A structure containing all the necessary information for
    /// creating a device.
    ///
    /// # Panics
    ///
    /// Panics if `props.adapter` doesn't support `props.surface`.
    // TODO: Rewrite
    // TODO: Implementations are valid if they support the extensions before they
    // got included into the 1.1, 1.2 or 1.3 spec.
    pub fn new_device(
        &self,
        surface: Option<&Surface>,
        adapter: Option<Adapter>,
        props: DeviceProps,
    ) -> Result<Device> {
        let InstanceShared { instance, .. } = &*self.0;
        let physical_devices = &self.0.physical_devices;

        let surface = surface.unwrap();
        let adapter = adapter.unwrap();

        assert!(adapter.is_surface_supported(surface)?);

        let physical_device_index = adapter.physical_device_index;
        let physical_device = physical_devices[physical_device_index];

        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let queue_family_indices = Adapter::find_queue_family_indices(&queue_families);
        let queue_create_infos =
            Self::setup_queue_create_infos(&queue_families, &queue_family_indices)?;

        let enabled_layer_names =
            Vec::from(Adapter::REQUIRED_LAYER_NAMES.map(|name| name.as_ptr()));

        let enabled_extension_names = {
            let mut names = Vec::from(Adapter::REQUIRED_EXTENSION_NAMES.map(|name| name.as_ptr()));
            names.push(khr::Swapchain::name().as_ptr());
            names
        };

        let mut features = vk::PhysicalDeviceFeatures2::builder();
        unsafe { instance.get_physical_device_features2(physical_device, &mut features) };

        let mut timeline_features =
            vk::PhysicalDeviceTimelineSemaphoreFeatures::builder().timeline_semaphore(true);

        let mut features = features.push_next(&mut timeline_features);

        let create_info = vk::DeviceCreateInfo::builder()
            .push_next(&mut features)
            .queue_create_infos(&queue_create_infos)
            .enabled_layer_names(&enabled_layer_names)
            .enabled_extension_names(&enabled_extension_names);

        let device = unsafe { instance.create_device(physical_device, &create_info, None)? };

        Ok(Device(Arc::new(DeviceShared {
            device,
            physical_device_index,
            physical_device,
            queue_family_indices,
            _instance: Arc::clone(&self.0),
        })))
    }

    pub fn new_swapchain(&self, device: &Device, surface: Surface) -> Result<DSwapchain> {
        use vk::{
            ColorSpaceKHR, CompositeAlphaFlagsKHR, Format, ImageUsageFlags, PresentModeKHR,
            SharingMode, SurfaceTransformFlagsKHR, SwapchainCreateFlagsKHR, SwapchainCreateInfoKHR,
            SwapchainKHR,
        };

        let surface_capabilities = unsafe {
            surface.extension.get_physical_device_surface_capabilities(
                device.0.physical_device,
                *surface.raw(),
            )?
        };

        let surface_formats = unsafe {
            surface
                .extension
                .get_physical_device_surface_formats(device.0.physical_device, *surface.raw())?
        };

        let format = Format::R8G8B8A8_UNORM;
        let extent = surface_capabilities.current_extent;
        println!("{extent:?}");

        let present_mode = PresentModeKHR::FIFO;

        let extension = khr::Swapchain::new(self.0.raw(), device.0.raw());
        let create_info = SwapchainCreateInfoKHR::builder()
            .flags(SwapchainCreateFlagsKHR::empty())
            .surface(*surface.raw())
            .min_image_count(1)
            .image_format(format)
            .image_color_space(ColorSpaceKHR::SRGB_NONLINEAR)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(SharingMode::EXCLUSIVE)
            .queue_family_indices(&[])
            .pre_transform(SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(false)
            .old_swapchain(SwapchainKHR::null());

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
                .format(format)
                .subresource_range(subresource_range)
                .build();

            image_views.push(unsafe { device.raw().create_image_view(&create_info, None) }?);
        }

        Ok(DSwapchain(Arc::new(SwapchainShared {
            extension,
            surface,
            swapchain,
            format,
            images,
            image_views,
            _device: Arc::clone(&device.0),
            _instance: Arc::clone(&self.0),
        })))
    }

    fn find_required_layers(
        props: &[vk::LayerProperties],
    ) -> Option<impl Iterator<Item = &vk::LayerProperties>> {
        let found_layers = props.iter().filter(|props| {
            let layer_name = unsafe { CStr::from_ptr(props.layer_name.as_ptr()) };
            Adapter::REQUIRED_LAYER_NAMES.contains(&layer_name)
        });

        let found_layer_names = found_layers
            .clone()
            .map(|props| unsafe { CStr::from_ptr(props.layer_name.as_ptr()) });

        let has_required_layers = Adapter::REQUIRED_LAYER_NAMES
            .iter()
            .all(|&required| found_layer_names.clone().any(|found| found == required));

        has_required_layers.then_some(found_layers)
    }

    fn find_required_extensions(
        props: &[vk::ExtensionProperties],
    ) -> Option<impl Iterator<Item = &vk::ExtensionProperties>> {
        let found_extensions = props.iter().filter(|props| {
            let extension_name = unsafe { CStr::from_ptr(props.extension_name.as_ptr()) };
            Adapter::REQUIRED_EXTENSION_NAMES.contains(&extension_name)
        });

        let found_extension_names = found_extensions
            .clone()
            .map(|props| unsafe { CStr::from_ptr(props.extension_name.as_ptr()) });

        let has_required_extensions = Adapter::REQUIRED_EXTENSION_NAMES
            .iter()
            .all(|&required| found_extension_names.clone().any(|found| found == required));

        has_required_extensions.then_some(found_extensions)
    }

    fn find_physical_devices(instance: &ash::Instance) -> Result<Vec<vk::PhysicalDevice>> {
        let physical_device_groups = unsafe {
            let mut buf =
                vec![Default::default(); instance.enumerate_physical_device_groups_len()?];
            instance.enumerate_physical_device_groups(buf.as_mut())?;

            buf
        };

        println!(
            "{:?}",
            unsafe { instance.enumerate_physical_devices() }.unwrap()
        );

        let physical_device_group = physical_device_groups[0];

        let physical_devices = {
            let len = physical_device_group.physical_device_count as _;
            Vec::from(&physical_device_group.physical_devices[0..len])
        };

        println!("{physical_devices:?}");

        let is_supported = |physical_device: vk::PhysicalDevice| {
            use vk::{api_version_major, api_version_minor, PhysicalDeviceProperties};

            let properties = unsafe { instance.get_physical_device_properties(physical_device) };
            let PhysicalDeviceProperties { api_version, .. } = properties;
            if api_version_major(api_version) != 1 || api_version_minor(api_version) != 3 {
                return Ok::<_, Error>(None);
            }

            let (layers, extensions) = unsafe {
                let layers = instance.enumerate_device_layer_properties(physical_device);
                let extensions = instance.enumerate_device_extension_properties(physical_device);
                (layers?, extensions?)
            };

            let has_required_layers = Self::find_required_layers(&layers).is_some();
            let has_required_extensions = Self::find_required_extensions(&extensions).is_some();

            if !has_required_layers || !has_required_extensions {
                return Ok(None);
            }

            Ok(Some(physical_device))
        };

        physical_devices
            .iter()
            .filter_map(|physical_device| is_supported(*physical_device).transpose())
            .collect()
    }

    fn setup_queue_create_infos(
        queue_families: &[vk::QueueFamilyProperties],
        queue_family_indices: &QueueFamilyIndices,
    ) -> Result<Vec<vk::DeviceQueueCreateInfo>> {
        use vk::DeviceQueueCreateInfo;

        const PRIORITIES: [f32; 32] = [1.0; 32];

        let queue_family_indices = Adapter::find_queue_family_indices(&queue_families);
        let graphics = queue_family_indices.graphics.unwrap_or_else(|| todo!());
        let compute = queue_family_indices.compute.unwrap_or_else(|| todo!());
        let transfer = queue_family_indices.transfer.unwrap_or_else(|| todo!());

        let mut create_infos = Vec::with_capacity(3);
        for i in [graphics, compute, transfer] {
            let create_info = DeviceQueueCreateInfo::builder()
                .queue_family_index(i as _)
                .queue_priorities(&PRIORITIES[0..1])
                .build();

            create_infos.push(create_info);
        }

        Ok(create_infos)
    }
}

impl_try_from_rhi_all!(Vulkan, Device);

pub struct Surface {
    surface: vk::SurfaceKHR,
    extension: khr::Surface,
    _instance: Arc<InstanceShared>,
}

impl std::fmt::Debug for Surface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Surface")
            .field("_instance", &self._instance)
            .finish_non_exhaustive()
    }
}

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

impl_try_from_rhi_all!(Vulkan, Surface);

#[derive(Debug, Clone)]
pub struct Adapter {
    physical_device_index: usize,
    _instance: Arc<InstanceShared>,
}

impl_try_from_rhi_all!(Vulkan, Adapter);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QueueFamilyIndices {
    graphics: Option<usize>,
    compute: Option<usize>,
    transfer: Option<usize>,
}

impl Adapter {
    const REQUIRED_LAYER_NAMES: [&'static CStr; 0] = [];

    #[cfg(target_os = "windows")]
    const REQUIRED_EXTENSION_NAMES: [&'static CStr; 2] = [
        khr::ExternalFenceWin32::name(),
        khr::ExternalSemaphoreWin32::name(),
    ];

    #[cfg(target_os = "linux")]
    const REQUIRED_EXTENSION_NAMES: [&'static CStr; 2] = [
        khr::ExternalFenceFd::name(),
        khr::ExternalSemaphoreFd::name(),
    ];

    /// Returns whether this adapter supports presenting to the passed surface
    pub fn is_surface_supported(&self, surface: &Surface) -> Result<bool> {
        let InstanceShared { instance, .. } = &*self._instance;
        let physical_devices = &self._instance.physical_devices;

        let physical_device = physical_devices[self.physical_device_index];

        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        for (i, _) in queue_family_properties.iter().enumerate() {
            let presentable: bool = unsafe {
                surface.extension.get_physical_device_surface_support(
                    physical_device,
                    i as _,
                    *surface.raw(),
                )?
            };

            if presentable {
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn is_supported(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Result<bool> {
        use vk::{api_version_major, api_version_minor, PhysicalDeviceProperties};

        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let PhysicalDeviceProperties { api_version, .. } = properties;
        if api_version_major(api_version) != 1 || api_version_minor(api_version) != 3 {
            return Ok(false);
        }

        let (layer_props, extension_props) = unsafe {
            let layer_props = instance.enumerate_device_layer_properties(physical_device)?;
            let extension_props =
                instance.enumerate_device_extension_properties(physical_device)?;
            (layer_props, extension_props)
        };

        let has_required_layers = Self::has_required_layers(&layer_props);
        let has_required_extensions = Self::has_required_extensions(&extension_props);
        if !has_required_layers || !has_required_extensions {
            return Ok(false);
        }

        Ok(true)
    }

    fn has_layer(name: &CStr, props: &[vk::LayerProperties]) -> bool {
        props.iter().any(|vk::LayerProperties {layer_name, ..}| unsafe {CStr::from_ptr(layer_name.as_ptr())} == name)
    }

    fn has_extension(name: &CStr, props: &[vk::ExtensionProperties]) -> bool {
        props
            .iter()
            .any(|vk::ExtensionProperties {extension_name, ..}| unsafe { CStr::from_ptr(extension_name.as_ptr()) } == name)
    }

    fn has_required_layers(props: &[vk::LayerProperties]) -> bool {
        let mut layer_names = props
            .iter()
            .map(|props| unsafe { CStr::from_ptr(props.layer_name.as_ptr()) });

        for required_layer_names in Self::REQUIRED_LAYER_NAMES {
            if !layer_names.any(|layer_name| layer_name == required_layer_names) {
                return false;
            }
        }

        true
    }

    fn has_required_extensions(props: &[vk::ExtensionProperties]) -> bool {
        let mut extension_names = props
            .iter()
            .map(|props| unsafe { CStr::from_ptr(props.extension_name.as_ptr()) });

        for required_extension_names in Self::REQUIRED_EXTENSION_NAMES {
            if !extension_names.any(|extension_names| extension_names == required_extension_names) {
                return false;
            }
        }

        true
    }

    fn has_required_features() -> bool {
        true
    }

    // TODO: Find the family with the most amount of queues.
    fn find_queue_family_indices(props: &[vk::QueueFamilyProperties]) -> QueueFamilyIndices {
        use vk::{QueueFamilyProperties, QueueFlags};

        let mut graphics = None;
        let mut compute = None;
        let mut transfer = None;
        for (i, QueueFamilyProperties { queue_flags, .. }) in props.iter().enumerate() {
            if queue_flags.contains(QueueFlags::GRAPHICS | QueueFlags::COMPUTE) {
                graphics = Some(i);
                continue;
            }

            if queue_flags.contains(QueueFlags::COMPUTE) {
                compute = Some(i);
                continue;
            }

            if queue_flags.contains(QueueFlags::TRANSFER) {
                transfer = Some(i);
                continue;
            }
        }

        QueueFamilyIndices {
            graphics,
            compute,
            transfer,
        }
    }
}

struct DeviceShared {
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    physical_device_index: usize,
    queue_family_indices: QueueFamilyIndices,
    _instance: Arc<InstanceShared>,
}

impl std::fmt::Debug for DeviceShared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceShared")
            .field("physical_device", &self.physical_device)
            .field("physical_device_index", &self.physical_device_index)
            .field("queue_family_indies", &self.queue_family_indices)
            .field("_instance", &self._instance)
            .finish_non_exhaustive()
    }
}

impl DeviceShared {
    fn raw(&self) -> &ash::Device {
        &self.device
    }
}

impl Drop for DeviceShared {
    fn drop(&mut self) {
        unsafe { self.device.destroy_device(None) };
    }
}

pub struct Device(Arc<DeviceShared>);

impl Device {
    pub fn queue(&self, operations: Operations) -> Option<DQueue> {
        let queue_family_indices = self.0.queue_family_indices;
        let family_index = match operations {
            Operations::Graphics => queue_family_indices.graphics.unwrap(),
            Operations::Compute => queue_family_indices.compute.unwrap(),
            Operations::Transfer => queue_family_indices.transfer.unwrap(),
        };

        let queue = unsafe { self.0.raw().get_device_queue(family_index as _, 0) };

        Some(DQueue(Arc::new(DQueueShared {
            queue,
            family_index,
            operations,
            _device: Arc::clone(&self.0),
        })))
    }

    pub fn new_command_pool(&self, queue: &DQueue) -> Result<DCommandPool> {
        use vk::{CommandPoolCreateFlags, CommandPoolCreateInfo};

        let create_info = CommandPoolCreateInfo::builder()
            .flags(CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue.0.family_index as _);

        let pool = unsafe { self.0.raw().create_command_pool(&create_info, None)? };

        Ok(DCommandPool(Arc::new(DCommandPoolShared {
            pool,
            _queue: Arc::clone(&queue.0),
        })))
    }

    pub fn new_command_list(&self, pool: &mut DCommandPool) -> Result<DCommandList> {
        use vk::{CommandBufferAllocateInfo, CommandBufferLevel};

        let create_info = CommandBufferAllocateInfo::builder()
            .command_pool(*pool.0.raw())
            .level(CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let buffer = unsafe { self.0.raw().allocate_command_buffers(&create_info)? }[0];

        Ok(DCommandList {
            buffer,
            state: State::Initial,
            _pool: Arc::clone(&pool.0),
        })
    }

    pub fn new_semaphore(&self, value: u64) -> Result<Semaphore> {
        use vk::{SemaphoreCreateInfo, SemaphoreType, SemaphoreTypeCreateInfo};

        let mut timeline_create_info = SemaphoreTypeCreateInfo::builder()
            .initial_value(value)
            .semaphore_type(SemaphoreType::TIMELINE);

        let create_info = SemaphoreCreateInfo::builder().push_next(&mut timeline_create_info);

        let semaphore = unsafe { self.0.raw().create_semaphore(&create_info, None)? };

        Ok(Semaphore {
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
        let fence = unsafe { self.0.raw().create_fence(&create_info, None)? };

        Ok(Fence {
            fence,
            _device: Arc::clone(&self.0),
        })
    }

    pub fn new_image_2d(&self) -> Result<DImage2D> {
        todo!()
    }

    pub fn new_image_3d(&self) -> Result<()> {
        todo!()
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

        let render_pass = unsafe { self.0.raw().create_render_pass(&create_info, None)? };

        Ok(RenderPass {
            render_pass,
            _device: Arc::clone(&self.0),
        })
    }

    pub fn new_framebuffer<'a, A>(
        &self,
        render_pass: &RenderPass,
        attachments: A,
        extent: UVec2,
    ) -> Result<Framebuffer>
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

        let framebuffer = unsafe { self.0.raw().create_framebuffer(&create_info, None)? };

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

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&[])
            .vertex_attribute_descriptions(&[]);

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

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

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

    pub fn wait_idle(&self) -> Result<()> {
        unsafe { self.0.raw().device_wait_idle() }.map_err(Into::into)
    }

    fn raw(&self) -> &ash::Device {
        &self.0.device
    }
}

impl Device {
    fn new_pipeline_layout(&self) -> Result<vk::PipelineLayout> {
        let layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&[])
            .push_constant_ranges(&[])
            .build();

        unsafe { self.raw().create_pipeline_layout(&layout_create_info, None) }.map_err(Into::into)
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        let _ = unsafe { self.0.raw().device_wait_idle() };
    }
}

struct SwapchainShared {
    extension: khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    format: vk::Format,
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
            .field("format", &self.format)
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
    pub unsafe fn image_unchecked(
        &mut self,
        fence: &mut Fence,
        timeout: Duration,
    ) -> Result<Option<DImageView2D>> {
        let SwapchainShared { extension, .. } = &*self.0;

        let device_mask = self.0._device.physical_device_index + 1;

        let acquire_next_image_info = vk::AcquireNextImageInfoKHR::builder()
            .swapchain(*self.raw())
            .timeout(timeout.as_nanos() as _)
            .fence(*fence.raw())
            .device_mask(device_mask as _);

        let result = unsafe { extension.acquire_next_image2(&acquire_next_image_info) };
        if matches!(result, Err(vk::Result::TIMEOUT)) {
            return Ok(None);
        }

        let (i, ..) = result?;

        Ok(Some(DImageView2D {
            kind: DImageViewKind2D::Swapchain(i as _, Arc::clone(&self.0)),
        }))
    }

    pub fn present(&self, image_view: &DImageView2D) -> Result<()> {
        let SwapchainShared {
            extension, _device, ..
        } = &*self.0;

        let swapchain = [*self.raw()];

        let DImageViewKind2D::Swapchain(i, _) = image_view.kind else {
            unreachable!();
        };

        let i = [i as _];

        let present_info = vk::PresentInfoKHR::builder()
            .swapchains(&swapchain)
            .image_indices(&i);

        let graphics_index = _device.queue_family_indices.graphics.unwrap();
        let graphics_queue = unsafe { _device.raw().get_device_queue(graphics_index as _, 0) };

        let _ = unsafe { extension.queue_present(graphics_queue, &present_info) }?;
        Ok(())
    }

    fn raw(&self) -> &vk::SwapchainKHR {
        &self.0.swapchain
    }
}

impl_try_from_rhi_all!(Vulkan, DSwapchain);

pub struct SubmitInfo<'a> {
    command_lists: Vec<&'a DCommandList>,
    wait_semaphores: Vec<(&'a Semaphore, u64)>,
    signal_semaphores: Vec<(&'a Semaphore, u64)>,
}

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

impl DQueue {
    pub fn operations(&self) -> Operations {
        self.0.operations
    }

    /// # Safety
    pub unsafe fn submit_unchecked(
        &mut self,
        infos: &[SubmitInfo],
        fence: Option<&mut Fence>,
    ) -> Result<()> {
        let device = &self.0._device.raw();

        let wait_semaphore_values: Vec<u64> = infos
            .iter()
            .flat_map(|info| info.wait_semaphores.iter().map(|(_, value)| *value))
            .collect();

        let signal_semaphore_values: Vec<u64> = infos
            .iter()
            .flat_map(|info| info.signal_semaphores.iter().map(|(_, value)| *value))
            .collect();

        let mut wait_semaphore_offset = 0;
        let mut signal_semaphore_offset = 0;
        let mut timeline_semaphore_submits: Vec<_> = infos
            .iter()
            .map(|info| {
                let wait_values = &wait_semaphore_values
                    [wait_semaphore_offset..wait_semaphore_offset + info.wait_semaphores.len()];

                let signal_values = &signal_semaphore_values[signal_semaphore_offset
                    ..signal_semaphore_offset + info.signal_semaphores.len()];

                wait_semaphore_offset += info.wait_semaphores.len();
                signal_semaphore_offset += info.signal_semaphores.len();

                vk::TimelineSemaphoreSubmitInfo::builder()
                    .wait_semaphore_values(wait_values)
                    .signal_semaphore_values(signal_values)
                    .build()
            })
            .collect();

        let command_buffers: Vec<Vec<_>> = infos
            .iter()
            .map(|info| {
                info.command_lists
                    .iter()
                    .map(|command_list| command_list.buffer)
                    .collect()
            })
            .collect();

        let wait_semaphores: Vec<Vec<_>> = infos
            .iter()
            .map(|info| {
                info.wait_semaphores
                    .iter()
                    .map(|(semaphore, _)| *semaphore.raw())
                    .collect()
            })
            .collect();

        let signal_semaphores: Vec<Vec<_>> = infos
            .iter()
            .map(|info| {
                info.signal_semaphores
                    .iter()
                    .map(|(semaphore, _)| *semaphore.raw())
                    .collect()
            })
            .collect();

        let submits: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(i, _)| {
                vk::SubmitInfo::builder()
                    .push_next(&mut timeline_semaphore_submits[i])
                    .command_buffers(&command_buffers[i])
                    .wait_semaphores(&wait_semaphores[i])
                    .signal_semaphores(&signal_semaphores[i])
                    .build()
            })
            .collect();

        let fence = fence.map(|fence| *fence.raw()).unwrap_or_default();
        unsafe { device.queue_submit(*self.raw(), &submits, fence) }.map_err(Into::into)
    }

    pub fn wait_idle(&mut self) -> Result<()> {
        let device = self.0._device.raw();
        unsafe { device.queue_wait_idle(*self.0.raw()) }.map_err(Into::into)
    }

    fn raw(&self) -> &vk::Queue {
        &self.0.queue
    }
}

impl_try_from_rhi_all!(Vulkan, DQueue);

impl<'a> TryFrom<rhi::SubmitInfo<'a>> for SubmitInfo<'a> {
    type Error = BackendError;

    fn try_from(value: rhi::SubmitInfo<'a>) -> std::result::Result<Self, Self::Error> {
        let command_lists: Vec<_> = value
            .command_lists
            .into_iter()
            .map(TryInto::try_into)
            .collect::<std::result::Result<_, _>>()?;

        let wait_semaphores: Vec<_> = value
            .wait_semaphores
            .into_iter()
            .map(|(semaphore, value)| semaphore.try_into().map(|s| (s, value)))
            .collect::<std::result::Result<_, _>>()?;

        let signal_semaphores: Vec<_> = value
            .signal_semaphores
            .into_iter()
            .map(|(semaphore, value)| semaphore.try_into().map(|s| (s, value)))
            .collect::<std::result::Result<_, _>>()?;

        Ok(SubmitInfo {
            command_lists,
            wait_semaphores,
            signal_semaphores,
        })
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
        unsafe { device.destroy_command_pool(*self.raw(), None) };
    }
}

pub struct DCommandPool(Arc<DCommandPoolShared>);

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
    _pool: Arc<DCommandPoolShared>,
}

impl_try_from_rhi_all!(Vulkan, DCommandList);

impl DCommandList {
    /// Returns the operations supported by this command list
    pub fn operations(&self) -> Operations {
        self._pool._queue.operations
    }

    /// Returns the current state of this command list
    pub fn state(&self) -> State {
        self.state
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
    pub unsafe fn begin_unchecked(&mut self, pool: &DCommandPool) -> Result<()> {
        use vk::{CommandBufferBeginInfo, CommandBufferInheritanceInfo, CommandBufferUsageFlags};

        assert_eq!(
            self._pool.raw(),
            pool.0.raw(),
            "the pool passed to this function is not the same as the one used to create this command list"
        );

        let inheritance_info = CommandBufferInheritanceInfo::builder();

        let begin_info = CommandBufferBeginInfo::builder()
            .flags(CommandBufferUsageFlags::empty())
            .inheritance_info(&inheritance_info);

        self.device()
            .begin_command_buffer(*self.raw(), &begin_info)?;

        self.state = State::Recording;
        Ok(())
    }

    /// # Safety
    ///
    /// - The command list must be in the recording state.
    ///
    /// - The command list must not have begun a render pass without ending it.
    pub unsafe fn begin_render_pass_unchecked(
        &mut self,
        render_pass: &RenderPass,
        framebuffer: &mut Framebuffer,
    ) {
        let device = self.device();

        let clear_value = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 0.0],
            },
        }];

        let begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(*render_pass.raw())
            .framebuffer(*framebuffer.raw())
            .render_area(vk::Rect2D {
                extent: vk::Extent2D {
                    width: 640,
                    height: 480,
                },
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

        unsafe {
            device.cmd_begin_render_pass(*self.raw(), &begin_info, vk::SubpassContents::INLINE)
        };
    }

    /// # Safety
    pub unsafe fn bind_pipeline_unchecked(&mut self, pipeline: &Pipeline) {
        let device = self.device();
        unsafe {
            device.cmd_bind_pipeline(
                *self.raw(),
                vk::PipelineBindPoint::GRAPHICS,
                *pipeline.raw(),
            )
        };
    }

    pub unsafe fn set_viewport_unchecked(&mut self, viewport: &ViewportState) {
        let ViewportState {
            position,
            extent,
            depth,
            scissor,
        } = viewport;

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
        self.device()
            .end_command_buffer(*self.raw())
            .map_err(Into::into)
    }

    fn raw(&self) -> &vk::CommandBuffer {
        &self.buffer
    }

    fn device(&self) -> &ash::Device {
        self._pool._queue._device.raw()
    }
}

impl Drop for DCommandList {
    fn drop(&mut self) {
        let device = self._pool._queue._device.raw();
        unsafe { device.free_command_buffers(*self._pool.raw(), &[self.buffer]) };
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

        let wait_info = vk::SemaphoreWaitInfo::builder()
            .semaphores(&semaphores)
            .values(&values);

        match unsafe { device.wait_semaphores(&wait_info, timeout.as_nanos() as _) } {
            Ok(_) => Ok(true),
            Err(e) if e == vk::Result::TIMEOUT => Ok(false),
            Err(e) => Err(Error::from(e)),
        }
    }

    pub fn signal(&mut self, value: u64) -> Result<()> {
        let device = self._device.raw();

        let signal_info = vk::SemaphoreSignalInfo::builder()
            .semaphore(*self.raw())
            .value(value);

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

impl From<Format> for vk::Format {
    fn from(value: Format) -> Self {
        match value {
            Format::R8G8B8A8Unorm => Self::R8G8B8A8_UNORM,
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
