use std::sync::*;

use ::xcb::{x, Connection};

use crate::os;

static CONNECTION: LazyLock<(Connection, i32)> = LazyLock::new(|| Connection::connect(None).unwrap());

#[derive(Debug, Clone)]
pub enum WindowError {}

pub struct Window {
    xid: x::Window,
}

impl Window {
    pub fn new(title: &str) -> Result<Self, WindowError> {
        let (connection, screen) = &*CONNECTION;
        let screen = connection.get_setup().roots().nth(*screen as _).unwrap();

        let window = connection.generate_id();

        let cookie = connection.send_request_checked(&x::CreateWindow {
            depth: x::COPY_FROM_PARENT as u8,
            wid: window,
            parent: screen.root(),
            x: 0,
            y: 0,
            width: 128,
            height: 128,
            border_width: 0,
            class: x::WindowClass::InputOutput,
            visual: screen.root_visual(),
            value_list: &[
                x::Cw::BackPixel(screen.black_pixel()),
                x::Cw::EventMask(x::EventMask::all()),
            ],
        });

        connection.check_request(cookie).unwrap();

        let cookie = connection.send_request_checked(&x::ChangeProperty {
            mode: x::PropMode::Replace,
            window,
            property: x::ATOM_WM_NAME,
            r#type: x::ATOM_STRING,
            data: title.as_bytes(),
        });

        connection.check_request(cookie).unwrap();

        Ok(Self { xid: window })
    }

    pub fn show(&mut self) {
        let (connection, _) = &*CONNECTION;
        let Self { xid } = self;

        let cookie = connection.send_request_checked(&x::MapWindow { window: *xid });
        connection.check_request(cookie).unwrap();
    }

    pub fn wait_events(&mut self) {
        let (connection, _) = &*CONNECTION;
    }
}

pub trait WindowExt {
    fn connection(&self) -> &'static Connection;
    fn xid(&self) -> &x::Window;
}

impl WindowExt for os::Window {
    fn connection(&self) -> &'static Connection {
        let (connection, _) = &*CONNECTION;
        connection
    }

    fn xid(&self) -> &x::Window {
        match &self.0 {
            super::Window::Xcb(window) => &window.xid,
            _ => unreachable!(),
        }
    }
}
