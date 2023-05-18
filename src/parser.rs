use std::marker::PhantomData;

pub trait Parser: Sized {
    type Input: Copy;
    type Output;

    fn parse(&self, input: Self::Input) -> Result<Self::Output, ()> {
        todo!()
    }

    fn or<P: Parser<Input = Self::Input>>(self, other: P) -> Or<Self, P> {
        Or(self, other)
    }
}

pub struct Or<A, B>(A, B);

impl<I, O, A, B> Parser for Or<A, B>
where
    I: Copy,
    A: Parser<Input = I, Output = O>,
    B: Parser<Input = I, Output = O>,
{
    type Input = I;
    type Output = O;

    fn parse(&self, input: Self::Input) -> Result<Self::Output, ()> {
        match self.0.parse(input) {
            Ok(o) => Ok(o),
            _ => self.1.parse(input),
        }
    }
}

pub struct Literal<I>(I);

impl<I: Copy> Parser for Literal<I> {
    type Input = I;
    type Output = ();

    fn parse(&self, input: I) -> Result<Self::Output, ()> {
        todo!()
    }
}

pub fn literal<I>(input: I) -> Literal<I> {
    Literal(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal() {
        const ALPHABET: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        assert!(literal("ABC").parse(ALPHABET).is_ok());
        assert!(literal("XYZ").parse(ALPHABET).is_err());

        const BYTES: &[u8] = &[0, 255, 5, 12, 139, 71];
        assert!(literal([0, 255, 5].as_slice()).parse(BYTES).is_ok());
        assert!(literal([12, 139, 71].as_slice()).parse(BYTES).is_err());
    }
}
