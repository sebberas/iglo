// use std::marker::*;

// use crate::parser::*;

// pub struct SpirvModule {}

// pub struct SpirvParser<'a>(PhantomData<&'a ()>);

// impl<'a> Parser<u8, SpirvModule> for SpirvParser<'a> {
//     fn parse<C: Container<u8>>(&self, input: C) -> ParseResult<&'a [u8],
// SpirvModule> {         const LE_MAGIC_NUMBER: &[u8] =
// &0x07230203u32.to_le_bytes();         const BE_MAGIC_NUMBER: &[u8] =
// &0x07230203u32.to_be_bytes();

//         let p = literal(LE_MAGIC_NUMBER).or(literal(BE_MAGIC_NUMBER));

//         todo!()
//     }
// }
