#[cfg(windows)]
pub mod windows;

use glam::UVec2;

#[cfg(windows)]
use self::windows as imp;

enum Cursor {
    Arrow,
}

trait WindowApi: Sized {
    fn new(title: &str) -> Result<Self, imp::WindowError>;

    fn extent(&self) -> UVec2;

    fn show(&mut self);
    fn hide(&mut self);

    fn poll_events(&self) -> ();
    fn wait_events(&self) -> ();

    fn should_close(&self) -> bool;
}

#[derive(Debug)]
pub struct WindowError(imp::WindowError);

pub struct Window(imp::Window);

impl Window {
    #[inline]
    pub fn new(title: &str) -> Result<Self, WindowError> {
        imp::Window::new(title).map(Self).map_err(WindowError)
    }

    #[inline]
    pub fn extent(&self) -> UVec2 {
        self.0.extent()
    }

    #[inline]
    pub fn show(&mut self) {
        self.0.show()
    }

    #[inline]
    pub fn hide(&mut self) {
        self.0.hide()
    }

    #[inline]
    pub fn poll_events(&self) -> impl Iterator<Item = &Event> {
        self.0.poll_events();
        [].into_iter()
    }

    #[inline]
    pub fn wait_events(&self) -> impl Iterator<Item = &Event> {
        self.0.wait_events();
        [].into_iter()
    }

    pub fn should_close(&self) -> bool {
        self.0.should_close()
    }
}

pub enum Event {}

pub struct SaveFileDialog(imp::SaveFileDialog);

pub struct OpenFileDialog(imp::OpenFileDialog);
