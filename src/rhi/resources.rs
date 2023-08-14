use std::num::*;

use crate::rhi::*;

pub enum Usage {
    Uniform,
    Storage,
    Vertex,
    Index,
}

pub struct DBufferProps {
    pub size: NonZeroUsize,
    pub usage: Usage,
}

pub enum DBuffer {
    Vulkan(vulkan::DBuffer),
}

impl DBuffer {
    pub unsafe fn map_unchecked(&self) -> Result<&mut [u8], Error> {
        match self {
            Self::Vulkan(dbuffer) => dbuffer.map_unchecked(),
        }
    }

    pub unsafe fn unmap_unchecked(&self) {
        match self {
            Self::Vulkan(dbuffer) => dbuffer.unmap_unchecked(),
        }
    }
}

pub enum DImage2D {}

pub enum DImage3D {}
