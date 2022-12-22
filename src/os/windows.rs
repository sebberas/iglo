use std::os::raw::c_void;

use windows::core::{HSTRING, PCWSTR};
use windows::Win32::Foundation::*;
use windows::Win32::Graphics::Gdi::*;
use windows::Win32::System::LibraryLoader::*;
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::*;

use crate::os;

#[derive(Debug)]
pub struct WindowError;

struct WindowShared {
    hwnd: HWND,
    hinstance: HINSTANCE,
}

pub struct Window(Box<WindowShared>);

impl Window {
    extern "system" fn wndproc_setup(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
        unsafe { DefWindowProcW(hwnd, msg, wp, lp) }
    }
}

static mut IS_WINDOW_REGISTERED: bool = false;

impl os::WindowApi for Window {
    fn new(title: &str) -> Result<Self, WindowError> {
        // SAFETY:
        let hinstance = unsafe { GetModuleHandleW(PCWSTR::null()) }.unwrap();

        // SAFETY:
        const WINDOW_CLASS_NAME: PCWSTR = w!("iglo_window");
        if unsafe { !IS_WINDOW_REGISTERED } {
            let window_class = WNDCLASSEXW {
                cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
                style: WNDCLASS_STYLES(0),
                lpfnWndProc: Some(Self::wndproc_setup),
                cbClsExtra: 0,
                cbWndExtra: 0,
                hInstance: hinstance,
                hIcon: HICON::default(),
                hCursor: HCURSOR::default(),
                hbrBackground: HBRUSH::default(),
                hIconSm: HICON::default(),
                lpszMenuName: PCWSTR::null(),
                lpszClassName: WINDOW_CLASS_NAME,
            };

            // SAFETY:
            unsafe { RegisterClassExW(&window_class) };

            // SAFETY:
            unsafe { IS_WINDOW_REGISTERED = true };
        }

        let mut window_shared = Box::new(WindowShared {
            hwnd: HWND::default(),
            hinstance,
        });

        // SAFETY:
        window_shared.hwnd = unsafe {
            CreateWindowExW(
                WINDOW_EX_STYLE(0),
                WINDOW_CLASS_NAME,
                &HSTRING::from(title),
                WS_OVERLAPPEDWINDOW,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                HWND::default(),
                HMENU::default(),
                hinstance,
                Some(&*window_shared as *const WindowShared as *const _),
            )
        };

        Ok(Self(window_shared))
    }

    fn show(&mut self) {
        // SAFETY:
        unsafe { ShowWindow(self.0.hwnd, SW_SHOW) };
    }

    fn hide(&mut self) {
        // SAFETY:
        unsafe { ShowWindow(self.0.hwnd, SW_HIDE) };
    }

    fn poll_events(&self) -> () {
        let mut msg = MSG::default();
        while unsafe { PeekMessageW(&mut msg, self.0.hwnd, 0, 0, PM_REMOVE) }.as_bool() {
            unsafe {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }

        ()
    }

    fn wait_events(&self) -> () {
        let mut msg = MSG::default();
        while unsafe { GetMessageW(&mut msg, self.0.hwnd, 0, 0) }.as_bool() {
            unsafe {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }

        ()
    }
}

pub trait WindowExt {
    fn hwnd(&self) -> HWND;
    fn hinstance(&self) -> HINSTANCE;
}

impl WindowExt for os::Window {
    fn hwnd(&self) -> HWND {
        let platform = &self.0;
        platform.0.hwnd
    }

    fn hinstance(&self) -> HINSTANCE {
        let platform = &self.0;
        platform.0.hinstance
    }
}
