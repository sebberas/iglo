#![feature(adt_const_params)]
#![feature(fn_traits)]
#![feature(generic_const_exprs)]
#![feature(pin_macro)]
#![feature(async_closure)]
#![feature(trivial_bounds)]
#![feature(nonzero_min_max)]
#![feature(generic_arg_infer)]
#![feature(try_find)]
#![feature(iterator_try_collect)]
#![feature(type_alias_impl_trait)]
#![feature(iter_array_chunks)]
#![feature(trait_alias)]
#![feature(let_chains)]
#![feature(portable_simd)]

pub mod core;
pub mod os;
// pub mod renderer;
pub mod rhi;

pub mod wasm;
