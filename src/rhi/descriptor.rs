use crate::rhi::backend::*;
use crate::rhi::*;

impl Device {
    pub fn new_descriptor_pool(
        &self,
        props: &DescriptorPoolProps,
    ) -> Result<DescriptorPool, Error> {
        match self {
            Self::Vulkan(device) => device.new_descriptor_pool().map(Into::into),
        }
    }

    pub fn new_ddescriptor_set(&self, pool: &mut DescriptorPool) -> Result<DDescriptorSet, Error> {
        match self {
            Self::Vulkan(device) => {
                let pool = pool.try_into().unwrap();
                device.new_descriptor_set(pool).map(Into::into)
            }
        }
    }
}

pub struct DescriptorPoolProps {}

pub enum DescriptorPool {
    Vulkan(vulkan::DescriptorPool),
}

impl DescriptorPool {}

pub struct DescriptorSetLayoutProps {}

pub enum DDescriptorSetLayout {
    Vulkan(vulkan::DDescriptorSetLayout),
}

pub enum DDescriptorSet {
    Vulkan(vulkan::DDescriptorSet),
}

impl DDescriptorSet {
    pub unsafe fn insert_unchecked(&mut self, binding: usize, buffer: &DBuffer) {
        match self {
            Self::Vulkan(descriptor_set) => {
                let buffer = buffer.try_into().unwrap();
                descriptor_set.insert_unchecked(binding, buffer);
            }
        }
    }
}

pub struct DescriptorSetLayout<T>(DDescriptorSetLayout, PhantomData<T>);

pub struct DescriptorSet<T>(DDescriptorSet, PhantomData<T>);
