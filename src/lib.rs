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
#![feature(return_position_impl_trait_in_trait)]
#![feature(lazy_cell)]
#![feature(let_chains)]

pub mod core;
pub mod os;
// pub mod renderer;
pub mod parser;
pub mod rhi;
