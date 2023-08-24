use std::os::raw::c_void;

use ::glam::*;
use ::windows::core::{HSTRING, PCWSTR};
use ::windows::Win32::Foundation::*;
use ::windows::Win32::Graphics::Gdi::*;
use ::windows::Win32::System::LibraryLoader::*;
use ::windows::Win32::UI::WindowsAndMessaging::*;
use ::windows::*;
use windows::core::w;

use crate::os;

#[derive(Debug)]
pub struct WindowError;

struct WindowData {
    hwnd: HWND,
    hinstance: HINSTANCE,
    should_close: bool,
}

impl WindowData {
    fn wndproc(&mut self, hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
        match msg {
            WM_CLOSE => {
                self.should_close = true;
                println!("WNDPROC: {:?}", self.should_close);
                return LRESULT(0);
            }
            _ => unsafe { DefWindowProcW(hwnd, msg, wp, lp) },
        }
    }

    extern "system" fn wndproc_setup(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
        match msg {
            WM_NCCREATE => {
                // SAFETY:
                // The windows documentation states that when the msg is WM_NCCREATE, lp is a
                // pointer to a valid CREATESTRUCTW struct.
                let CREATESTRUCTW { lpCreateParams, .. } =
                    unsafe { std::mem::transmute::<_, &CREATESTRUCTW>(lp) };

                // SAFETY:
                // - We know data is non-null because we supply it when calling CreateWindowExW.
                // - The object is destroyed when the window is dropped because the allocation
                //   is managed by a box.
                let data: *mut Self = unsafe { std::mem::transmute(*lpCreateParams) };

                // SAFETY:
                // - `data` is valid for the entire lifetime since it is tied to the life of the
                //   window.
                unsafe { SetWindowLongPtrW(hwnd, GWLP_USERDATA, data as *const _ as isize) };
                unsafe { SetWindowLongPtrW(hwnd, GWLP_WNDPROC, Self::wndproc_proxy as isize) };

                // SAFETY:
                // - We know that `data` is non-null, because we supply a valid pointer when
                //   calling CreateWindowExW during the creation of the window.
                unsafe { &mut (*data) }.wndproc(hwnd, msg, wp, lp)
            }
            // SAFETY:
            _ => unsafe { DefWindowProcW(hwnd, msg, wp, lp) },
        }
    }

    extern "system" fn wndproc_proxy(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
        let data: &mut Self =
            unsafe { std::mem::transmute(GetWindowLongPtrW(hwnd, GWLP_USERDATA)) };

        data.wndproc(hwnd, msg, wp, lp)
    }
}

pub struct Window(Box<WindowData>);

static mut IS_WINDOW_REGISTERED: bool = false;

impl os::WindowApi for Window {
    fn new(title: &str) -> Result<Self, WindowError> {
        // SAFETY:
        let hmodule = unsafe { GetModuleHandleW(PCWSTR::null()) }.unwrap();

        // SAFETY:
        const WINDOW_CLASS_NAME: PCWSTR = w!("iglo_window");
        if unsafe { !IS_WINDOW_REGISTERED } {
            let window_class = WNDCLASSEXW {
                cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
                style: WNDCLASS_STYLES(0),
                lpfnWndProc: Some(WindowData::wndproc_setup),
                cbClsExtra: 0,
                cbWndExtra: 0,
                hInstance: hmodule.into(),
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

        let mut window_data = Box::new(WindowData {
            hwnd: HWND::default(),
            hinstance: hmodule.into(),
            should_close: false,
        });

        // SAFETY:
        window_data.hwnd = unsafe {
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
                HINSTANCE::from(hmodule),
                Some(window_data.as_mut() as *mut _ as *mut c_void),
            )
        };

        Ok(Self(window_data))
    }

    fn inner_extent(&self) -> UVec2 {
        let mut rect = RECT::default();
        unsafe { GetClientRect(self.0.hwnd, &mut rect) };

        let RECT { left, top, right, bottom } = rect;

        uvec2((right - left) as _, (bottom - top) as _)
    }

    fn outer_extent(&self) -> UVec2 {
        let mut rect = RECT::default();
        unsafe { GetWindowRect(self.0.hwnd, &mut rect) };

        let RECT { left, top, right, bottom } = rect;

        uvec2((right - left) as _, (bottom - top) as _)
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
        while unsafe { PeekMessageW(&mut msg, self.0.hwnd, 0, 0, PM_REMOVE) }.0 != 0 {
            unsafe {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
    }

    fn wait_events(&self) -> () {
        unsafe { WaitMessage() };
        self.poll_events()
    }

    fn should_close(&self) -> bool {
        self.0.should_close
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

pub struct SaveFileDialog {}

pub struct OpenFileDialog {}
