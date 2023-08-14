pub use self::ops::{OperationsError, OperationsType};
use crate::rhi::backend::*;
use crate::rhi::*;

pub enum DQueue {
    Vulkan(vulkan::DQueue),
}

impl DQueue {
    /// Returns the operations supported by this queue.
    pub fn operations(&self) -> Operations {
        match &self {
            Self::Vulkan(q) => q.operations(),
        }
    }

    /// Submits commands to the GPU.
    ///
    /// # Safety
    ///
    /// It is undefined behaviour to drop any of the the command lists before
    /// the device is finished executing them.
    ///
    /// # Panics
    ///
    /// This function panics if one of the `infos` doesn't have any command
    /// lists attached with it.
    ///
    /// This function panics if any of the command lists belong to another
    /// queue.
    // TODO Measure overhead of type conversions. Maybe use iterators with generics
    // instead.
    pub unsafe fn submit_unchecked<'a>(
        &mut self,
        submit_infos: impl IntoIterator<Item = SubmitInfo<'a>>,
        fence: Option<&mut Fence>,
    ) -> Result<(), Error> {
        match self {
            Self::Vulkan(queue) => {
                let submit_infos: Vec<_> = submit_infos
                    .into_iter()
                    .map(TryInto::try_into)
                    .collect::<Result<_, _>>()
                    .unwrap();

                let fence = fence.map(|fence| fence.try_into().unwrap());
                queue.submit_unchecked(submit_infos.as_slice(), fence)
            }
        }
    }

    /// Blocks on the calling-thread until the queue has completed all pending
    /// work.
    pub fn wait_idle(&mut self) -> Result<(), Error> {
        match self {
            Self::Vulkan(queue) => queue.wait_idle(),
        }
    }
}

impl AsRef<DQueue> for DQueue {
    fn as_ref(&self) -> &DQueue {
        self
    }
}

impl AsMut<DQueue> for DQueue {
    fn as_mut(&mut self) -> &mut DQueue {
        self
    }
}

pub struct Queue<O: OperationsType = ops::Graphics>(DQueue, PhantomData<O>);

impl<O: OperationsType> Queue<O> {
    /// Creates a new queue where the supported operations are encoded in the
    /// type.
    pub fn new(dqueue: DQueue) -> Result<Self, OperationsError> {
        let operations = dqueue.operations();
        if operations == O::OPERATIONS {
            Ok(unsafe { Self::new_unchecked(dqueue) })
        } else {
            Err(OperationsError {
                expected: O::OPERATIONS,
                found: operations,
            })
        }
    }

    /// Creates a new queue that is guaranteed to support `O` operations.
    ///
    /// For the safe variant see [`Self::new`].
    ///
    /// # Safety
    /// It is undefined behaviour if `dqueue` doesn't support `O`
    /// operations.
    pub unsafe fn new_unchecked(dqueue: DQueue) -> Self {
        Self(dqueue, PhantomData)
    }
}

impl<O: OperationsType> AsRef<DQueue> for Queue<O> {
    fn as_ref(&self) -> &DQueue {
        &self.0
    }
}

impl<O: OperationsType> AsMut<DQueue> for Queue<O> {
    fn as_mut(&mut self) -> &mut DQueue {
        &mut self.0
    }
}

pub type GraphicsQueue = Queue<ops::Graphics>;
pub type ComputeQueue = Queue<ops::Compute>;
pub type TransferQueue = Queue<ops::Transfer>;

pub enum DCommandPool {
    Vulkan(vulkan::DCommandPool),
}

impl DCommandPool {
    pub fn operations(&self) -> Operations {
        match self {
            Self::Vulkan(command_pool) => command_pool.operations(),
        }
    }
}

impl AsMut<DCommandPool> for DCommandPool {
    fn as_mut(&mut self) -> &mut DCommandPool {
        self
    }
}

pub struct CommandPool<O: OperationsType = ops::Graphics>(DCommandPool, PhantomData<O>);

impl<O: OperationsType> CommandPool<O> {
    pub fn new(command_pool: DCommandPool) -> Option<Self> {
        match command_pool.operations() {
            operations if operations == O::OPERATIONS => {
                Some(unsafe { CommandPool::new_unchecked(command_pool) })
            }
            _ => None,
        }
    }

    pub unsafe fn new_unchecked(command_pool: DCommandPool) -> Self {
        Self(command_pool, PhantomData)
    }
}

impl<O: OperationsType> AsMut<DCommandPool> for CommandPool<O> {
    fn as_mut(&mut self) -> &mut DCommandPool {
        &mut self.0
    }
}

pub type GraphicsCommandPool = CommandPool<ops::Graphics>;
pub type ComputeCommandPool = CommandPool<ops::Compute>;
pub type TransferCommandPool = CommandPool<ops::Transfer>;

pub enum DCommandList {
    Vulkan(vulkan::DCommandList),
}

impl DCommandList {
    pub fn operations(&self) -> Operations {
        match self {
            Self::Vulkan(command_list) => command_list.operations(),
        }
    }
}

impl DCommandList {
    /// Resets this command list
    ///
    /// # Panics
    ///
    /// Panics if `command_pool` is not the same command pool as the one that
    /// was passed in when creating this command list initially.
    ///
    /// # Safety
    ///
    /// This command list must not be in the initial or executable state.
    pub unsafe fn reset_unchecked(&mut self, command_pool: &mut DCommandPool) -> Result<(), Error> {
        match self {
            Self::Vulkan(command_list) => {
                let command_pool: &mut _ = command_pool.try_into().unwrap();
                command_list.reset_unchecked(command_pool)
            }
        }
    }

    /// Begins recording on this command list.
    ///
    /// This sets the internal state of the command list to recording.
    ///
    /// # Safety
    ///
    /// This command list must be in the initial state.
    ///
    /// Only a single command list in a pool can be recording at any given
    /// time. This means that until `end_unchecked` is called, `command_pool`
    /// must not begin recording for any of its
    /// other command lists.
    ///
    /// # Panics
    ///
    /// Panics if `command_pool` is not the command pool as the one that was
    /// passed in when creating this command list initially.
    pub unsafe fn begin_unchecked(&mut self, command_pool: &mut DCommandPool) -> Result<(), Error> {
        match self {
            Self::Vulkan(command_list) => {
                let command_pool: &mut _ = command_pool.try_into().unwrap();
                command_list.begin_unchecked(command_pool)
            }
        }
    }

    /// Ends recording for this command list.
    ///
    /// After this has been called it is again safe to use the command pool for
    /// recording another command list.
    ///
    ///  # Safety
    ///
    /// It is undefined behaviour if this command list is not in the recording
    /// state.
    pub unsafe fn end_unchecked(&mut self) -> Result<(), Error> {
        match self {
            Self::Vulkan(command_list) => command_list.end_unchecked(),
        }
    }

    /// Begins a render pass on this command list.
    ///
    /// # Safety
    ///
    /// It is undefined behaviour if any of the following are broken:
    /// - This command list must support graphics operations and be in the
    ///   recording state.
    /// - No other render passes are currently recording meaning you can't call
    ///   `begin_render_pass_unchecked` two times in a row without calling
    ///   `end_render_pass_unchecked` in between.
    pub unsafe fn begin_render_pass_unchecked(
        &mut self,
        render_pass: &RenderPass,
        framebuffer: &mut Framebuffer,
    ) {
        match self {
            Self::Vulkan(command_list) => {
                let render_pass = render_pass.try_into().unwrap();
                let framebuffer = framebuffer.try_into().unwrap();
                command_list.begin_render_pass_unchecked(render_pass, framebuffer);
            }
        }
    }

    /// Ends a render pass on this command list.
    ///
    /// After calling this method it is again safe to call
    /// `begin_render_pass_unchecked`.
    ///
    /// # Safety
    ///
    /// It is undefined behaviour if any of the following are broken:
    /// - This command list must support graphics operations and be in the
    ///   recording state.
    /// - This command list is not currently recording a render pass.
    pub unsafe fn end_render_pass_unchecked(&mut self) {
        match self {
            Self::Vulkan(command_list) => {
                command_list.end_render_pass_unchecked();
            }
        }
    }

    /// Binds a pipeline to this command list.
    ///
    ///
    /// # Safety
    ///
    /// If `pipeline` is a graphics pipeline, this command list must be
    /// currently recording a render pass
    ///
    /// If `pipeline` is a compute pipeline, this command list must be currently
    /// recording a compute pass.
    pub unsafe fn bind_pipeline_unchecked(&mut self, pipeline: &Pipeline) {
        match self {
            Self::Vulkan(command_list) => {
                let pipeline = pipeline.try_into().unwrap();
                command_list.bind_pipeline_unchecked(pipeline)
            }
        }
    }

    /// Binds vertex buffers to this command list
    ///
    /// # Safety
    pub unsafe fn bind_vertex_buffers_unchecked<'a, I>(&mut self, buffers: I)
    where
        I: IntoIterator<Item = &'a DBuffer>,
    {
        match self {
            Self::Vulkan(command_list) => {
                let buffers = buffers.into_iter().map(|buffer| buffer.try_into().unwrap());
                command_list.bind_vertex_buffers_unchecked(buffers);
            }
        }
    }

    /// Sets the viewport state
    ///
    /// # Panics
    ///
    /// - Panics if the pipeline doesn't support updating the viewport state
    ///   dynamically.
    ///
    /// # Safety
    ///
    /// A pipeline must have been bound
    pub unsafe fn set_viewport_unchecked(&mut self, viewport: &ViewportState) {
        match self {
            Self::Vulkan(command_list) => command_list.set_viewport_unchecked(viewport),
        }
    }

    /// Draw
    ///
    /// # Safety
    ///
    /// This command list must be recording a render pass.
    pub unsafe fn draw_unchecked(&mut self, vertices: usize, instances: usize) {
        match self {
            Self::Vulkan(command_list) => command_list.draw_unchecked(vertices, instances),
        }
    }

    /// Copy one buffer to another
    pub unsafe fn copy_buffer_unchecked(
        &mut self,
        command_pool: &mut DCommandPool,
        src: &DBuffer,
        dst: &DBuffer,
        size: usize,
    ) {
        match self {
            Self::Vulkan(command_list) => {
                let command_pool = command_pool.try_into().unwrap();
                let src = src.try_into().unwrap();
                let dst = dst.try_into().unwrap();
                command_list.copy_buffer_unchecked(command_pool, src, dst, size)
            }
        }
    }

    /// Copy one image to another
    pub unsafe fn copy_image_unchecked(
        &mut self,
        command_pool: &mut DCommandPool,
        src: &DImage2D,
        dst: &DImage2D,
    ) {
        match self {
            Self::Vulkan(command_list) => {
                let command_pool = command_pool.try_into().unwrap();
                let src = src.try_into().unwrap();
                let dst = dst.try_into().unwrap();
                command_list.copy_image_unchecked(command_pool, src, dst)
            }
        }
    }
}

pub struct CommandList<O: OperationsType = ops::Graphics>(DCommandList, PhantomData<O>);

impl<O: OperationsType> CommandList<O> {
    pub fn new(command_list: DCommandList) -> Option<Self> {
        match command_list.operations() {
            operations if operations == O::OPERATIONS => {
                Some(unsafe { CommandList::new_unchecked(command_list) })
            }
            _ => None,
        }
    }

    pub unsafe fn new_unchecked(command_list: DCommandList) -> Self {
        Self(command_list, PhantomData)
    }
}

impl<O: OperationsType> AsMut<DCommandList> for CommandList<O> {
    fn as_mut(&mut self) -> &mut DCommandList {
        &mut self.0
    }
}

pub type GraphicsCommandList = CommandList<ops::Graphics>;
pub type ComputeCommandList = CommandList<ops::Compute>;
pub type TransferCommandList = CommandList<ops::Transfer>;
