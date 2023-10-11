use std::num::NonZeroUsize;
use std::ops::RangeInclusive;

use ::glam::*;

use crate::rhi::*;
#[derive(Debug, Clone, Copy)]
pub enum VertexInputRate {
    Vertex,
    Instance,
}

#[derive(Debug, Clone, Copy)]
pub struct VertexInputBinding {
    pub binding: usize,
    pub stride: usize,
    pub rate: VertexInputRate,
}

#[derive(Debug, Clone, Copy)]
pub struct VertexInputAttribute {
    pub location: usize,
    pub binding: usize,
    pub format: Format,
    pub offset: usize,
}

#[derive(Debug, Clone)]
pub struct VertexInputState {
    pub bindings: Vec<VertexInputBinding>,
    pub attributes: Vec<VertexInputAttribute>,
}

#[derive(Debug, Clone, Copy)]
pub struct ScissorState {
    pub offset: UVec2,
    pub extent: UVec2,
}

#[derive(Debug, Clone)]
pub struct ViewportState {
    pub position: Vec2,
    pub extent: Vec2,
    pub depth: RangeInclusive<f32>,
    pub scissor: Option<ScissorState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Samples(NonZeroUsize);

impl Samples {
    pub const ONE: Samples = Samples(unsafe { NonZeroUsize::new_unchecked(1) });
    pub const TWO: Samples = Samples(unsafe { NonZeroUsize::new_unchecked(2) });
    pub const FOUR: Samples = Samples(unsafe { NonZeroUsize::new_unchecked(4) });
    pub const SIXTEEN: Samples = Samples(unsafe { NonZeroUsize::new_unchecked(1) });
    pub const THIRTYTWO: Samples = Samples(unsafe { NonZeroUsize::new_unchecked(2) });
    pub const SIXTYFOUR: Samples = Samples(unsafe { NonZeroUsize::new_unchecked(4) });

    pub fn new(count: usize) -> Option<Samples> {
        match count {
            1 | 2 | 4 | 16 | 32 | 64 => Some(unsafe { Self::new_unchecked(count) }),
            _ => None,
        }
    }

    pub unsafe fn new_unchecked(count: usize) -> Samples {
        Self(unsafe { NonZeroUsize::new_unchecked(count) })
    }
}

impl Default for Samples {
    fn default() -> Self {
        Self::ONE
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MultisampleState {
    pub samples: Samples,
}

#[derive(Debug, Clone)]
pub struct PipelineState {
    pub vertex_input: Option<VertexInputState>,
    pub viewport: Option<ViewportState>,
    pub multisample: MultisampleState,
}

pub struct PipelineShaders<'a, Shader = rhi::Shader> {
    pub vertex: Option<&'a Shader>,
    pub tessellation_control: Option<&'a Shader>,
    pub tessellation_evalution: Option<&'a Shader>,
    pub geometry: Option<&'a Shader>,
    pub pixel: Option<&'a Shader>,
    pub compute: Option<&'a Shader>,
}

impl<'a, Shader> Default for PipelineShaders<'a, Shader> {
    fn default() -> Self {
        Self {
            vertex: None,
            tessellation_control: None,
            tessellation_evalution: None,
            geometry: None,
            pixel: None,
            compute: None,
        }
    }
}

pub struct PipelineProps<
    'a,
    Shader = rhi::Shader,
    DDescriptorSetLayout = rhi::DDescriptorSetLayout,
    RenderPass = rhi::RenderPass,
> {
    pub shaders: PipelineShaders<'a, Shader>,
    pub sets: Vec<&'a DDescriptorSetLayout>,
    pub state: &'a PipelineState,
    pub render_pass: &'a RenderPass,
}

pub enum Pipeline {
    Vulkan(vulkan::Pipeline),
}
