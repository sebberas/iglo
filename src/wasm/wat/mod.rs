use ::combine::parser::char::*;
use ::combine::parser::combinator::*;
use ::combine::parser::repeat::*;
use ::combine::parser::sequence::*;
use ::combine::parser::EasyParser;
use ::combine::*;

mod float;

pub mod types {
    use std::simd::*;

    pub enum NumberType {
        I32(i32),
        I64(i64),
        F32(f32),
        F64(f64),
    }

    pub enum VectorType {
        I32x4(i32x4),
        I64x2(i64x4),
        F32x4(f32x4),
        F64x4(f64x2),
    }

    pub enum RefType {
        Function(),
        External(),
    }

    pub enum ValueType {
        Number(NumberType),
        Vector(VectorType),
        Ref(RefType),
    }
}
