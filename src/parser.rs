pub trait Parser<'a> {
    type Input;
    type Output;

    fn parse(&self, input: I) -> Result<Self::Output, ()> {
        todo!()
    }
}

pub struct Literal<I>(I);

impl<I, O> Parser for Literal<I> {
    type Input = I;
    type Output = O;

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

        const BYTES: &[u8] = [0, 255, 5, 12, 139, 71];
        assert!(literal([0, 255, 5]).parse(ALPHABET).is_ok());
        assert!(literal([12, 139, 71]).parse(ALPHABET).is_err());
    }
}
