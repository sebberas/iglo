use ::glam::*;

use self::xcb::{Window as XcbWindow, WindowError as XcbWindowError};
use crate::os;

pub mod xcb;

#[derive(Debug, Clone)]
pub enum WindowError {
    Xcb(XcbWindowError),
    Wayland(()),
}

pub enum Window {
    Xcb(XcbWindow),
    Wayland(()),
}

impl os::WindowApi for Window {
    fn new(title: &str) -> Result<Self, WindowError> {
        // TODO: Decide if XCB or Wayland
        XcbWindow::new(title)
            .map(Window::Xcb)
            .map_err(WindowError::Xcb)
    }

    fn client_extent(&self) -> UVec2 {
        todo!()
    }

    fn extent(&self) -> UVec2 {
        todo!()
    }

    fn show(&mut self) {
        match self {
            Self::Xcb(window) => window.show(),
            Self::Wayland(_) => unimplemented!(),
        }
    }

    fn hide(&mut self) {
        todo!()
    }

    fn poll_events(&self) -> os::EventIterator {
        [].into_iter()
    }

    fn wait_events(&self) -> os::EventIterator {
        [].into_iter()
    }

    fn should_close(&self) -> bool {
        todo!()
    }
}

pub trait WindowExt {
    fn imp(&self) -> &Window;
}

impl WindowExt for os::Window {
    fn imp(&self) -> &Window {
        &self.0
    }
}

pub struct SaveFileDialog;

pub struct OpenFileDialog;
