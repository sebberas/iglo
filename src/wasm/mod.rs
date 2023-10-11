// parser! {
//     fn line_comment[Input]()(Input) -> ()
//     where [Input: Stream<Token = char>]
//     {
//         ignore((
//             string(";;"),
//             repeat_until::<Vec<_>, _, _, _>(any(),
// choice((ignore(newline()), eof()))),
// choice((ignore(newline()), eof())),         ))
//     }
// }

// parser! {
//     fn block_comment[Input]()(Input) -> ()
//     where [Input: Stream<Token = char>]
//     {
//         ignore((
//             string("(;"),
//             many::<Vec<_>, _, _>(
//                 choice!(
//                     ignore(none_of([';', '('])),
//                     ignore(attempt((char('('),
// not_followed_by(char(';'))))),
// ignore(attempt((char(';'), not_followed_by(char(')'))))),
// block_comment()                 )
//             ),
//             string(";)")
//         ))
//     }
// }

// parser! {
//     fn comment[Input]()(Input) -> ()
//     where [Input: Stream<Token = char>]
//     {
//         choice!(
//             line_comment(),
//             block_comment()
//         )
//     }
// }

// parser! {
//     fn whitespace[Input]()(Input) -> ()
//     where [Input: Stream<Token = char>]
//     {
//         ignore(
//             many::<Vec<()>, _, _>(
//                 choice!(
//                     ignore(char(' ')),
//                     ignore(one_of(['\t', '\n', '\r'])),
//                     ignore(comment())
//                 )
//             )
//         )
//     }
// }

// parser! {
//     fn number[Input]()(Input) -> ()
//     where [Input: Stream<Token = char>]
//     {

//         ignore((
//             digit(),
//             many::<Vec<_>, _, _, >(
//                 (optional(char('_')), digit())
//             )
//         ))
//     }
// }

// parser! {
//     fn hex_number[Input]()(Input) -> ()
//     where [Input: Stream<Token = char>]
//     {

//         ignore((
//             hex_digit(),
//             many::<Vec<_>, _, _, >(
//                 (optional(char('_')), hex_digit())
//             )
//         ))
//     }
// }

// parser! {
//     fn unsigned_n[Input]()(Input) -> ()
//     where [Input: Stream<Token = char>]
//     {
//         choice!(
//             number(),
//             ignore((string("0x"), hex_number()))
//         )
//     }
// }

// parser! {
//     fn signed_n[Input]()(Input) -> ()
//     where [Input: Stream<Token = char>]
//     {
//         ignore((
//             one_of(['+', '-']),
//             choice!(
//                 number(),
//                 ignore((string("0x"), hex_number()))
//             )
//         ))
//     }
// }

// parser! {
//     fn n[Input]()(Input) -> ()
//     where [Input: Stream<Token = char>]
//     {
//         choice!(
//             number(),
//             ignore((string("0x"), hex_number()))
//
// #[test]
// fn test_comment() {
//     comment()
//         .easy_parse("(; hello world (; skrt skrt ;) (; skrt skrt ;) (;
// skrt skrt ;) ;)")         .unwrap();

//     whitespace()
//         .easy_parse(";; hello world ;)\n    (; din far er skaldet\n\n
// ;)")         .unwrap();
// }

// #[test]
// fn test_number() {
//     number().easy_parse("1_23").unwrap();
// }

// #[test]
// fn test_hex_number() {
//     hex_number().easy_parse("1_2_3FF").unwrap();
// }

// pub struct WasmParser {}

pub mod wat;
