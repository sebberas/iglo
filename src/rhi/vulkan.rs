use std::ffi::*;
use std::ops::{Bound, Range, RangeBounds};
use std::slice::SliceIndex;
use std::sync::*;

use ash::extensions::*;
use ash::vk::Handle;
use ash::{vk, Entry};

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
            _ => Error::Unknown,
        }
    }
}

const VALIDATION_LAYER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };

struct InstanceShared {
    entry: Entry,
    instance: ash::Instance,
    physical_devices: Vec<vk::PhysicalDevice>,
}

pub struct Instance(Arc<InstanceShared>);

impl Instance {
    const ENGINE_NAME: &'static CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"iglo\0") };

    const REQUIRED_DEVICE_LAYER_NAMES: [&'static CStr; 0] = [];
    const REQUIRED_DEVICE_EXTENSION_NAMES: [&'static CStr; 0] = [];

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

    pub unsafe fn new_surface(&self, window: *const Window) -> Result<Surface> {
        use crate::os::windows::WindowExt;

        let win32_extension = khr::Win32Surface::new(&self.0.entry, &self.0.instance);

        let create_info = vk::Win32SurfaceCreateInfoKHR::builder()
            .flags(vk::Win32SurfaceCreateFlagsKHR::empty())
            .hinstance(unsafe { &*window }.hinstance().0 as *const _)
            .hwnd(unsafe { &*window }.hwnd().0 as *const _);

        let surface = win32_extension.create_win32_surface(&create_info, None)?;
        let extension = khr::Surface::new(&self.0.entry, &self.0.instance);

        Ok(Surface {
            surface,
            extension,
            _instance: Arc::clone(&self.0),
        })
    }

    pub fn enumerate_adapters(&self) -> impl Iterator<Item = Adapter> + '_ {
        let physical_devices = &self.0.physical_devices;

        physical_devices
            .iter()
            .enumerate()
            .map(|(physical_device_index, _)| Adapter {
                physical_device_index,
                _instance: Arc::clone(&self.0),
            })
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

        let adapter = props.adapter.unwrap_or_else(|| todo!());
        let surface = props.surface.unwrap_or_else(|| todo!());
        assert!(adapter.is_surface_supported(surface)?);

        let physical_device = physical_devices[adapter.physical_device_index];

        // TODO: Abstract into a few functions.
        let queue_create_infos = {
            use vk::{DeviceQueueCreateInfo, QueueFamilyProperties};

            let queue_family_props =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

            let QueueFamilyIndices {
                graphics_family_index,
                compute_family_index,
                transfer_family_index,
            } = Adapter::find_queue_family_indices(&queue_family_props);

            let mut queue_create_infos = Vec::with_capacity(3);
            let mut graphics_queue_priorities = Vec::default();

            {
                let queue_family_index = graphics_family_index.unwrap_or_else(|| todo!());
                let queue_count = queue_family_props[queue_family_index].queue_count as usize;

                let (min_queue_count, max_queue_count) = {
                    let range = props.graphics_queues.unwrap_or(1..queue_count + 1);
                    (range.start, range.end)
                };

                if min_queue_count > queue_count {
                    return Err(Error::NotSupported);
                }

                graphics_queue_priorities = vec![1.0; queue_count.min(max_queue_count)];

                queue_create_infos.push(
                    DeviceQueueCreateInfo::builder()
                        .queue_family_index(queue_family_index as _)
                        .queue_priorities(&queue_priorities),
                );
            }

            queue_create_infos
        };

        let mut enabled_extension_names =
            Vec::from(Adapter::REQUIRED_EXTENSION_NAMES.map(|name| name.as_ptr()));

        enabled_extension_names.push(khr::Swapchain::name().as_ptr());

        let create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&[])
            .enabled_extension_names(&enabled_extension_names);

        let device = unsafe { instance.create_device(physical_device, &create_info, None)? };

        Ok(Device(Arc::new(DeviceShared {
            device,
            _instance: Arc::clone(&self.0),
        })))
    }

    fn find_required_layers(
        props: &[vk::LayerProperties],
    ) -> Option<impl Iterator<Item = &vk::LayerProperties>> {
        let found_layers = props.iter().filter(|props| {
            let layer_name = unsafe { CStr::from_ptr(props.layer_name.as_ptr()) };
            Self::REQUIRED_DEVICE_LAYER_NAMES.contains(&layer_name)
        });

        let found_layer_names = found_layers
            .clone()
            .map(|props| unsafe { CStr::from_ptr(props.layer_name.as_ptr()) });

        let has_required_layers = Self::REQUIRED_DEVICE_LAYER_NAMES
            .iter()
            .all(|&required| found_layer_names.clone().any(|found| found == required));

        has_required_layers.then_some(found_layers)
    }

    fn find_required_extensions(
        props: &[vk::ExtensionProperties],
    ) -> Option<impl Iterator<Item = &vk::ExtensionProperties>> {
        let found_extensions = props.iter().filter(|props| {
            let extension_name = unsafe { CStr::from_ptr(props.extension_name.as_ptr()) };
            Self::REQUIRED_DEVICE_EXTENSION_NAMES.contains(&extension_name)
        });

        let found_extension_names = found_extensions
            .clone()
            .map(|props| unsafe { CStr::from_ptr(props.extension_name.as_ptr()) });

        let has_required_extensions = Self::REQUIRED_DEVICE_EXTENSION_NAMES
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

impl_try_from_rhi_all!(Vulkan, Surface);
impl_into_rhi!(Vulkan, Surface);

#[derive(Clone)]
pub struct Adapter {
    physical_device_index: usize,
    _instance: Arc<InstanceShared>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QueueFamilyIndices {
    graphics_family_index: Option<usize>,
    compute_family_index: Option<usize>,
    transfer_family_index: Option<usize>,
}

impl Adapter {
    const REQUIRED_LAYER_NAMES: [&'static CStr; 0] = [];
    const REQUIRED_EXTENSION_NAMES: [&'static CStr; 0] = [];

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

        let mut graphics_family_index = None;
        let mut compute_family_index = None;
        let mut transfer_family_index = None;
        for (family_index, QueueFamilyProperties { queue_flags, .. }) in props.iter().enumerate() {
            if queue_flags.contains(QueueFlags::GRAPHICS | QueueFlags::COMPUTE) {
                graphics_family_index = Some(family_index);
                continue;
            }

            if queue_flags.contains(QueueFlags::COMPUTE) {
                compute_family_index = Some(family_index);
                continue;
            }

            if queue_flags.contains(QueueFlags::TRANSFER) {
                transfer_family_index = Some(family_index);
                continue;
            }
        }

        QueueFamilyIndices {
            graphics_family_index,
            compute_family_index,
            transfer_family_index,
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
    _instance: Arc<InstanceShared>,
}

pub struct Device(Arc<DeviceShared>);
