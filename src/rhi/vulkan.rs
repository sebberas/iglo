use std::ffi::*;
use std::ops::{Range, RangeInclusive};
use std::sync::*;
use std::time::Duration;

use ash::extensions::*;
use ash::{vk, Entry};
use glam::UVec2;
#[cfg(target_os = "windows")]
use windows::Win32::Foundation::*;
use windows::Win32::System::Threading::*;
use windows::Win32::System::WindowsProgramming::*;

use crate::rhi::macros::*;
use crate::rhi::*;

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
            .hinstance(unsafe { &*window }.hinstance().0 as *const _)
            .hwnd(unsafe { &*window }.hwnd().0 as *const _);

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
    pub fn new_device(&self, props: DeviceProps) -> Result<Device> {
        let InstanceShared { instance, .. } = &*self.0;
        let physical_devices = &self.0.physical_devices;

        let adapter = props.adapter.as_ref().unwrap_or_else(|| todo!());
        let surface = props.surface.as_ref().unwrap_or_else(|| todo!());
        assert!(adapter.is_surface_supported(surface)?);

        let physical_device = physical_devices[adapter.physical_device_index];

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

        let create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_layer_names(&enabled_layer_names)
            .enabled_extension_names(&enabled_extension_names);

        let device = unsafe { instance.create_device(physical_device, &create_info, None)? };

        Ok(Device(Arc::new(DeviceShared {
            device,
            physical_device,
            queue_family_indices,
            _instance: Arc::clone(&self.0),
        })))
    }

    pub fn new_swapchain(&self, device: &Device, surface: Surface) -> Result<Swapchain> {
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

        let _surface_formats = unsafe {
            surface
                .extension
                .get_physical_device_surface_formats(device.0.physical_device, *surface.raw())?
        };

        let image_format = Format::R8G8B8A8_UNORM;
        let image_extent = surface_capabilities.current_extent;

        let present_mode = PresentModeKHR::FIFO;

        let extension = khr::Swapchain::new(self.0.raw(), device.0.raw());
        let create_info = SwapchainCreateInfoKHR::builder()
            .flags(SwapchainCreateFlagsKHR::empty())
            .surface(*surface.raw())
            .min_image_count(1)
            .image_format(image_format)
            .image_color_space(ColorSpaceKHR::SRGB_NONLINEAR)
            .image_extent(image_extent)
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

        Ok(Swapchain(Arc::new(SwapchainShared {
            extension,
            surface,
            swapchain,
            images,
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
            let len = instance.enumerate_physical_device_groups_len()?;

            let mut buf = vec![Default::default(); len];
            instance.enumerate_physical_device_groups(buf.as_mut())?;

            buf
        };

        let physical_device_group = physical_device_groups[0];

        let physical_devices = {
            let len = physical_device_group.physical_device_count as _;
            &physical_device_group.physical_devices[0..len]
        };

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

pub struct Surface {
    surface: vk::SurfaceKHR,
    extension: khr::Surface,
    _instance: Arc<InstanceShared>,
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

// impl_try_from_rhi_all!(Vulkan, Surface);
// impl_into_rhi!(Vulkan, Surface);

#[derive(Clone)]
pub struct Adapter {
    physical_device_index: usize,
    _instance: Arc<InstanceShared>,
}

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

#[derive(Default)]
pub struct DeviceProps<'a> {
    pub surface: Option<&'a Surface>,
    pub adapter: Option<Adapter>,
    pub graphics_queues: Option<Range<usize>>,
    pub compute_queues: Option<Range<usize>>,
    pub transfer_queues: Option<Range<usize>>,
}

impl<'a> AsRef<DeviceProps<'a>> for DeviceProps<'a> {
    fn as_ref(&self) -> &DeviceProps<'a> {
        self
    }
}

struct DeviceShared {
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    queue_family_indices: QueueFamilyIndices,
    _instance: Arc<InstanceShared>,
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

    pub fn new_render_pass(&self, props: &RenderPassProps) -> Result<RenderPass> {
        let mut attachments = Vec::with_capacity(props.attachments.len());
        let mut attachment_refs = Vec::with_capacity(props.attachments.len());
        for (i, attachment) in props.attachments.iter().enumerate() {
            let attachment = vk::AttachmentDescription::builder()
                .format(attachment.format.into())
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(attachment.load_op.into())
                .store_op(attachment.store_op.into())
                .stencil_load_op(attachment.stencil_load_op.into())
                .stencil_store_op(attachment.stencil_store_op.into())
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

            attachments.push(attachment.build());

            let attachment_ref = vk::AttachmentReference::builder()
                .attachment(i as _)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

            attachment_refs.push(attachment_ref.build());
        }

        let subpasses = [vk::SubpassDescription::builder()
            .color_attachments(&attachment_refs[0..1])
            .build()];

        let create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&[]);

        let render_pass = unsafe { self.0.raw().create_render_pass(&create_info, None)? };
        Ok(RenderPass {
            render_pass,
            _device: Arc::clone(&self.0),
        })
    }

    pub fn new_framebuffer(
        &self,
        image_view: &DImageView,
        render_pass: &RenderPass,
    ) -> Result<Framebuffer> {
        let attachment = [*image_view.raw()];

        let create_info = vk::FramebufferCreateInfo::builder()
            .render_pass(*render_pass.raw())
            .attachments(&attachment)
            .width(640)
            .height(480)
            .layers(1);

        let framebuffer = unsafe { self.0.raw().create_framebuffer(&create_info, None)? };
        Ok(Framebuffer {
            framebuffer,
            _device: Arc::clone(&self.0),
        })
    }

    pub fn new_shader(&self, src: &[u8]) -> Result<Shader> {
        use vk::{ShaderModuleCreateFlags, ShaderModuleCreateInfo};

        let mut create_info = ShaderModuleCreateInfo::builder();
        create_info.code_size = src.len();
        create_info.p_code = src.as_ptr() as *const _;

        let shader = unsafe { self.0.raw().create_shader_module(&create_info, None)? };

        Ok(Shader {
            shader,
            _device: Arc::clone(&self.0),
        })
    }

    /// Creates a new pipeline.
    ///
    /// # Panics
    ///
    /// Panics the same variant of [ShaderStage] is specified twice in
    /// [PipelineShaderStage::stage](PipelineShaderStage).
    pub fn new_pipeline<'a, 'b, P>(&self, props: &P) -> Result<Pipeline<'a>>
    where
        P: AsRef<PipelineProps<'a, 'b>>,
        'a: 'b,
    {
        use vk::{
            GraphicsPipelineCreateInfo, PipelineCache, PipelineShaderStageCreateFlags,
            PipelineShaderStageCreateInfo,
        };

        let props = props.as_ref();

        let entrypoints: Vec<_> = props
            .stages
            .iter()
            .map(|PipelineShaderStage { entrypoint, .. }| {
                CString::new(entrypoint.to_string()).unwrap()
            })
            .collect();

        let mut stages = Vec::with_capacity(6);
        for (i, stage) in props.stages.iter().enumerate() {
            let PipelineShaderStage { stage, shader, .. } = stage;

            let create_info = PipelineShaderStageCreateInfo::builder()
                .flags(PipelineShaderStageCreateFlags::empty())
                .stage((*stage).into())
                .module(*shader.raw())
                .name(&entrypoints[i])
                .build();

            stages.push(create_info);
        }

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(&[])
            .vertex_binding_descriptions(&[]);

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let scissor = [vk::Rect2D::builder()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(vk::Extent2D {
                width: 640,
                height: 480,
            })
            .build()];

        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&[vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: 640.0,
                height: 480.0,
                min_depth: 0.0,
                max_depth: 1.0,
            }])
            .scissors(&scissor);

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE);

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .min_sample_shading(1.0)
            .sample_mask(&[])
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false);

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder();

        let attachment = [vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE)
            .alpha_blend_op(vk::BlendOp::ADD)
            .build()];

        let color_blend_state =
            vk::PipelineColorBlendStateCreateInfo::builder().attachments(&attachment);

        let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

        let pipeline_layout = self.setup_vk_pipeline_layout(&props.layout)?;

        let create_info = GraphicsPipelineCreateInfo::builder()
            .stages(&stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            // .depth_stencil_state(&depth_stencil_state)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(*props.render_pass.raw())
            .build();

        let pipeline = unsafe {
            self.0
                .raw()
                .create_graphics_pipelines(PipelineCache::null(), &[create_info], None)
        }
        .unwrap()[0];

        Ok(Pipeline {
            pipeline,
            pipeline_layout,
            _device: Arc::clone(&self.0),
            _marker: PhantomData,
        })
    }

    fn setup_vk_pipeline_layout(&self, layout: &PipelineLayout) -> Result<vk::PipelineLayout> {
        use vk::{PipelineLayoutCreateFlags, PipelineLayoutCreateInfo};

        let create_info = PipelineLayoutCreateInfo::builder()
            .flags(PipelineLayoutCreateFlags::empty())
            .set_layouts(&[])
            .push_constant_ranges(&[]);

        unsafe {
            self.0
                .raw()
                .create_pipeline_layout(&create_info, None)
                .map_err(Into::into)
        }
    }

    pub fn new_fence(&self, signaled: bool) -> Result<Fence> {
        let flags = signaled
            .then_some(vk::FenceCreateFlags::SIGNALED)
            .unwrap_or_default();

        let create_info = vk::FenceCreateInfo::builder().flags(flags);

        let fence = unsafe { self.0.raw().create_fence(&create_info, None)? };
        Ok(Fence {
            fence,
            _device: Arc::clone(&self.0),
        })
    }

    pub fn new_semaphore(&self) -> Result<Semaphore> {
        let create_info = vk::SemaphoreCreateInfo::builder();
        let semaphore = unsafe { self.0.raw().create_semaphore(&create_info, None)? };
        Ok(Semaphore {
            semaphore,
            _device: Arc::clone(&self.0),
        })
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
    images: Vec<vk::Image>,
    surface: Surface,
    _device: Arc<DeviceShared>,
    _instance: Arc<InstanceShared>,
}

pub struct Swapchain(Arc<SwapchainShared>);

impl Swapchain {
    pub fn image(
        &mut self,
        semaphore: Option<&mut Semaphore>,
        fence: Option<&mut Fence>,
    ) -> Result<Option<DImageView>> {
        let SwapchainShared { extension, .. } = &*self.0;
        //  let image_view =

        let timeout = 16 * 1000 * 1000;
        let semaphore = semaphore.map(|e| *e.raw()).unwrap_or(vk::Semaphore::null());
        let fence = fence.map(|e| *e.raw()).unwrap_or(vk::Fence::null());

        let i = unsafe { extension.acquire_next_image(*self.raw(), timeout, semaphore, fence)? }.0;
        let image = self.0.images[i as usize];

        let create_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .components(vk::ComponentMapping::default())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
                ..Default::default()
            });

        let image_view = unsafe { self.0._device.raw().create_image_view(&create_info, None)? };
        Ok(Some(DImageView {
            image_view,
            kind: DImageViewKind::Swapchain(i as _, Arc::clone(&self.0)),
            _device: Arc::clone(&self.0._device),
        }))
    }

    pub fn present(&self, queue: &mut DQueue, image_view: &DImageView) -> Result<()> {
        let SwapchainShared { extension, .. } = &*self.0;

        let swapchain = [*self.raw()];
        let i = match image_view.kind {
            DImageViewKind::Swapchain(i, _) => [i as _],
        };

        let present_info = vk::PresentInfoKHR::builder()
            .swapchains(&swapchain)
            .image_indices(&i);

        let _ = unsafe { extension.queue_present(*queue.0.raw(), &present_info) }?;
        Ok(())
    }

    fn raw(&self) -> &vk::SwapchainKHR {
        &self.0.swapchain
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe { self.0.extension.destroy_swapchain(*self.raw(), None) };
    }
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
        list: &DCommandList,
        wait: Option<&Semaphore>,
        signal: &Semaphore,
        fence: Option<&mut Fence>,
    ) -> Result<()> {
        let device = self.0._device.raw();

        let (wait, dst_stage_mask) = if let Some(wait) = wait {
            (
                vec![*wait.raw()],
                vec![vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
            )
        } else {
            (Vec::default(), Vec::default())
        };

        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(&wait)
            .wait_dst_stage_mask(&dst_stage_mask)
            .command_buffers(&[*list.raw()])
            .signal_semaphores(&[*signal.raw()])
            .build();

        let fence = fence.map(|e| *e.raw()).unwrap_or(vk::Fence::null());
        unsafe { device.queue_submit(*self.0.raw(), &[submit_info], fence) }.map_err(Into::into)
    }

    pub fn wait_idle(&mut self) -> Result<()> {
        let device = self.0._device.raw();
        unsafe { device.queue_wait_idle(*self.0.raw()) }.map_err(Into::into)
    }
}

impl_try_from_rhi_all!(Vulkan, DQueue);

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
            })
            .clear_values(&clear_value);

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

    pub unsafe fn set_viewport_unchecked(
        &mut self,
        position: UVec2,
        size: UVec2,
        depth: RangeInclusive<f32>,
    ) {
        let device = self.device();

        let viewport = vk::Viewport::builder()
            .x(position.x as _)
            .y(position.y as _)
            .width(size.x as _)
            .height(size.y as _)
            .min_depth(*depth.start())
            .max_depth(*depth.end())
            .build();

        unsafe { device.cmd_set_viewport(*self.raw(), 0, &[viewport]) };
    }

    pub unsafe fn set_scissor_unchecked(&mut self, offset: UVec2, extent: UVec2) {
        let device = self.device();

        let scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D {
                x: offset.x as _,
                y: offset.y as _,
            })
            .extent(vk::Extent2D {
                width: extent.x as _,
                height: extent.y as _,
            })
            .build();

        unsafe { device.cmd_set_scissor(*self.raw(), 0, &[scissor]) };
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

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum AttachmentLoadOp {
    Load,
    Clear,
    #[default]
    DontCare,
}

impl From<AttachmentLoadOp> for vk::AttachmentLoadOp {
    fn from(value: AttachmentLoadOp) -> Self {
        match value {
            AttachmentLoadOp::Load => Self::LOAD,
            AttachmentLoadOp::Clear => Self::CLEAR,
            AttachmentLoadOp::DontCare => Self::DONT_CARE,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum AttachmentStoreOp {
    Store,
    #[default]
    DontCare,
}

impl From<AttachmentStoreOp> for vk::AttachmentStoreOp {
    fn from(value: AttachmentStoreOp) -> Self {
        match value {
            AttachmentStoreOp::Store => Self::STORE,
            AttachmentStoreOp::DontCare => Self::DONT_CARE,
        }
    }
}

pub struct Attachment {
    pub format: Format,
    pub load_op: AttachmentLoadOp,
    pub store_op: AttachmentStoreOp,
    pub stencil_load_op: AttachmentLoadOp,
    pub stencil_store_op: AttachmentStoreOp,
}

pub struct RenderPassProps<'a> {
    pub attachments: &'a [Attachment],
}

pub struct RenderPass {
    render_pass: vk::RenderPass,
    _device: Arc<DeviceShared>,
}

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
        unsafe {
            self._device
                .raw()
                .destroy_render_pass(self.render_pass, None);
        }
    }
}

pub struct Framebuffer {
    framebuffer: vk::Framebuffer,
    _device: Arc<DeviceShared>,
}

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
    fn raw(&self) -> &vk::ShaderModule {
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

pub struct Pipeline<'a> {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    _device: Arc<DeviceShared>,
    _marker: PhantomData<&'a ()>,
}

impl<'a> Pipeline<'a> {
    fn raw(&self) -> &vk::Pipeline {
        &self.pipeline
    }
}

impl<'a> Drop for Pipeline<'a> {
    fn drop(&mut self) {
        let device = self._device.raw();
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
        };
    }
}

pub struct Fence {
    fence: vk::Fence,
    _device: Arc<DeviceShared>,
}

impl Fence {
    pub fn wait(&mut self) -> Result<()> {
        let device = self._device.raw();
        unsafe { device.wait_for_fences(&[*self.raw()], true, 16 * 1000 * 1000) }
            .map_err(Into::into)
    }

    pub fn reset(&mut self) -> Result<()> {
        let device = self._device.raw();
        unsafe { device.reset_fences(&[*self.raw()]) }.map_err(Into::into)
    }

    pub fn set_callback(&mut self, f: impl Fn() + 'static) {
        let instance = self._device._instance.raw();
        let device = self._device.raw();

        #[cfg(target_os = "windows")]
        {
            let extension = khr::ExternalFenceWin32::new(instance, device);
            let get_info = vk::FenceGetWin32HandleInfoKHR::builder()
                .fence(*self.raw())
                .handle_type(vk::ExternalFenceHandleTypeFlags::OPAQUE_WIN32);

            let handle = unsafe { extension.get_fence_win32_handle(&get_info) }.unwrap();
            let handle = HANDLE(handle as _);

            let mut out = HANDLE::default();
            unsafe {
                // HACK
                if RegisterWaitForSingleObject(
                    &mut out,
                    handle,
                    Some(Self::callback_proxy),
                    None,
                    INFINITE,
                    WT_EXECUTEINPERSISTENTTHREAD | WT_EXECUTEONLYONCE,
                ) == BOOL(0)
                {
                    println!("{:?}", unsafe { GetLastError() });
                }
            }
        }

        #[cfg(target_os = "linux")]
        {
            todo!()
        }
    }

    #[cfg(target_os = "windows")]
    unsafe extern "system" fn callback_proxy(ptr: *mut c_void, _: BOOLEAN) {
        println!("fired");
    }

    fn raw(&self) -> &vk::Fence {
        &self.fence
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe { self._device.raw().destroy_fence(*self.raw(), None) };
    }
}

pub struct Semaphore {
    semaphore: vk::Semaphore,
    _device: Arc<DeviceShared>,
}

impl Semaphore {
    fn raw(&self) -> &vk::Semaphore {
        &self.semaphore
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe { self._device.raw().destroy_semaphore(*self.raw(), None) };
    }
}

pub struct DImage {
    image: Option<vk::Image>,
    _device: Arc<DeviceShared>,
}

impl Drop for DImage {
    fn drop(&mut self) {
        if let Some(image) = self.image.take() {
            unsafe { self._device.raw().destroy_image(image, None) };
        }
    }
}

enum DImageViewKind {
    // Owned(Arc<ImageShared>),
    Swapchain(usize, Arc<SwapchainShared>),
}

pub struct DImageView {
    image_view: vk::ImageView,
    kind: DImageViewKind,
    _device: Arc<DeviceShared>,
}

impl DImageView {
    fn raw(&self) -> &vk::ImageView {
        &self.image_view
    }
}

impl Drop for DImageView {
    fn drop(&mut self) {
        unsafe { self._device.raw().destroy_image_view(*self.raw(), None) };
    }
}

impl From<Format> for vk::Format {
    fn from(value: Format) -> Self {
        Self::UNDEFINED
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
