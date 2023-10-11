use std::ops::Div;

use ::combine::parser::char::*;
use ::combine::parser::combinator::*;
use ::combine::parser::repeat::*;
use ::combine::parser::sequence::*;
use ::combine::parser::EasyParser;
use ::combine::*;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum FloatValue {
    F32(f32),
    F64(f64),
}

parser! {
    #[derive(Clone, Copy)]
    struct Frac;

    fn frac[Input]()(Input) -> f64
    where [Input: Stream<Token = char>]
    {
        (
            digit(),
            optional(
                (
                    optional(char('_')),
                    frac()
                )
            )
        ).map(|dp| {
            match dp {
                (d, None) => (unsafe { d.to_digit(10).unwrap_unchecked() } as f64) / 10.0,
                (d, Some((_, p))) => (unsafe { d.to_digit(10).unwrap_unchecked() as f64 } + p) / 10.0
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frac() {
        let x = frac().easy_parse("123_456_789").unwrap();
        assert!(x.0 == 0.123456789)

        // hex_digit().easy_parse("").unwrap();
        // digit().easy_parse("a").unwrap();
    }
}

// // Parses a float-point value.
// fn fpv<Input>() -> impl Parser<Input, Output = Float>
// where
//     Input: Stream<Token = char>,
// {
//     // let frac = |_| choice!(
//     //     digit(),
//     // )

//     ignore(any()).map(|_| todo!())
// }
