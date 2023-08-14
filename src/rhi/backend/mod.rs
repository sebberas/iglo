// pub mod dx12;

use std::fmt::*;

pub mod vulkan;

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    Mismatch,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    Vulkan,
    #[cfg(target_os = "windows")]
    DX12,
    #[cfg(target_os = "macos")]
    Metal,
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    OpenGl460,
    #[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
    OpenGl330,
    #[cfg(any(target_os = "windows"))]
    DX11,
}

impl Display for Backend {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        use Backend::*;

        match &self {
            #[cfg(target_os = "windows")]
            DX12 => f.write_str("DirectX 12"),
            #[cfg(any(target_os = "windows", target_os = "linux"))]
            Vulkan => f.write_str("Vulkan"),
            #[cfg(target_os = "macos")]
            Metal => f.write_str("Metal"),
            #[cfg(any(target_os = "windows", target_os = "linux"))]
            OpenGl460 => f.write_str("OpenGL 4.60"),
            #[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
            OpenGl330 => f.write_str("OpenGL 3.30"),
            #[cfg(any(target_os = "windows"))]
            DX11 => f.write_str("DirectX 11"),
        }
    }
}
