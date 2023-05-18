use std::marker::PhantomData;

use crate::parser::{literal, Parser};

pub struct SpirvModule {}

pub struct SpirvParser<'a>(PhantomData<&'a ()>);

impl<'a> Parser for SpirvParser<'a> {
    type Input = &'a [u8];
    type Output = &'a SpirvModule;

    fn parse(&self, input: Self::Input) -> Result<Self::Output, ()> {
        const LE_MAGIC_NUMBER: &[u8] = &0x07230203u32.to_le_bytes();
        const BE_MAGIC_NUMBER: &[u8] = &0x07230203u32.to_be_bytes();

        let p = literal(LE_MAGIC_NUMBER).or(literal(BE_MAGIC_NUMBER));

        todo!()
    }
}
