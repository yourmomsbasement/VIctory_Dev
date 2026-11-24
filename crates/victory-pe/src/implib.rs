//! COFF import libraries (`.lib`).
//!
//! An umbrella import library tells us which DLL provides each API that a
//! console title may link against. That is the map the runtime needs to
//! decide which of our implementations answers which import.

use object::Object;
use object::ObjectSymbol;
use object::read::archive::ArchiveFile;
use object::read::coff::{CoffFile, ImportFile, ImportName, ImportType};

use crate::{ImportSymbol, PeError};

/// One `dll!export` pair described by an import library.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct LibImport {
    pub dll: String,
    pub export: ImportSymbol,
    pub kind: LibImportKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LibImportKind {
    Code,
    Data,
    Const,
}

/// Result of reading an import library.
#[derive(Debug, Default)]
pub struct ImportLibrary {
    pub imports: Vec<LibImport>,
    /// Members that are not short-format imports (regular objects, long
    /// import descriptors and their null thunks). Reported so nothing is
    /// silently lost.
    pub other_members: usize,
}

// Values from the COFF import-object header. They are kept as plain integers
// here because the `object` crate exposes them as newtypes whose constants
// cannot be compared with bytes read from an archive.
const SIG1: u16 = 0; // IMAGE_FILE_MACHINE_UNKNOWN
const SIG2: u16 = 0xffff; // IMPORT_OBJECT_HDR_SIG2
#[cfg(test)]
const TYPE_CODE: u16 = 0;
const TYPE_DATA: u16 = 1;
const TYPE_CONST: u16 = 2;
const NAME_ORDINAL: u16 = 0;
const NAME: u16 = 1;
const NAME_NO_PREFIX: u16 = 2;
const NAME_UNDECORATE: u16 = 3;
const NAME_EXPORTAS: u16 = 4;

/// A long import descriptor has the same 20-byte prefix as a short import
/// object but carries no inline string data. The names live in the COFF
/// object that follows the null descriptor and null thunk members.
fn import_header(bytes: &[u8]) -> Option<(u16, u32, u16, u16)> {
    if bytes.len() < 20 {
        return None;
    }
    let sig1 = u16::from_le_bytes(bytes[0..2].try_into().ok()?);
    let sig2 = u16::from_le_bytes(bytes[2..4].try_into().ok()?);
    if sig1 != SIG1 || sig2 != SIG2 {
        return None;
    }
    Some((
        u16::from_le_bytes(bytes[8..10].try_into().ok()?), // machine
        u32::from_le_bytes(bytes[12..16].try_into().ok()?), // size_of_data
        u16::from_le_bytes(bytes[16..18].try_into().ok()?), // ordinal/hint
        u16::from_le_bytes(bytes[18..20].try_into().ok()?), // type/name type
    ))
}

fn import_kind(flags: u16) -> LibImportKind {
    match flags & 0b11 {
        TYPE_DATA => LibImportKind::Data,
        TYPE_CONST => LibImportKind::Const,
        _ => LibImportKind::Code,
    }
}

/// Names defined by the COFF object that belongs to a long descriptor.
fn coff_symbols(bytes: &[u8]) -> Vec<String> {
    let Ok(file) = CoffFile::<_, object::pe::ImageFileHeader>::parse(bytes) else {
        return Vec::new();
    };
    file.symbols()
        .filter_map(|symbol| symbol.name().ok().map(str::to_owned))
        .collect()
}

fn strip_decoration(name: &str, name_type: u16) -> String {
    // `IMPORT_OBJECT_NAME` is passed through as-is; only the no-prefix and
    // undecorate forms ask us to remove a leading decoration.
    if name_type == NAME {
        return name.to_owned();
    }
    let stripped = if matches!(name_type, NAME_NO_PREFIX | NAME_UNDECORATE) {
        name.strip_prefix(['?', '@', '_']).unwrap_or(name)
    } else {
        name
    };
    if name_type == NAME_UNDECORATE {
        stripped.split('@').next().unwrap_or(stripped).to_owned()
    } else {
        stripped.to_owned()
    }
}

impl ImportLibrary {
    pub fn parse(data: &[u8]) -> Result<Self, PeError> {
        let archive = ArchiveFile::parse(data)?;
        let mut lib = ImportLibrary::default();
        let members: Vec<_> = archive.members().collect::<Result<Vec<_>, _>>()?;
        let contents: Vec<Vec<u8>> = members
            .iter()
            .map(|member| member.data(data).map(<[u8]>::to_vec))
            .collect::<Result<_, _>>()?;
        let mut consumed = vec![false; contents.len()];

        for (index, member) in contents.iter().enumerate() {
            if consumed[index] {
                continue;
            }
            if let Some((_, size, ordinal, flags)) = import_header(member)
                && size == 0 {
                    // Long-format descriptor. Its DLL and export names live
                    // in the COFF object(s) that follow the null members.
                    consumed[index] = true;
                    lib.other_members += 1;
                    let kind = import_kind(flags);
                    let name_type = flags >> 2 & 0b111;
                    let mut dll = None;
                    let mut names = Vec::new();
                    for (next, object) in contents.iter().enumerate().skip(index + 1) {
                        if consumed[next] || import_header(object).is_some() {
                            break;
                        }
                        let symbols = coff_symbols(object);
                        if symbols.is_empty() {
                            continue;
                        }
                        consumed[next] = true;
                        lib.other_members += 1;
                        for symbol in symbols {
                            if let Some(d) = symbol.strip_prefix("__IMPORT_DESCRIPTOR_") {
                                dll = Some(d.to_owned());
                            } else if let Some(export) = symbol.strip_prefix("__imp_") {
                                names.push(export.to_owned());
                            }
                        }
                    }
                    let Some(dll) = dll else {
                        continue;
                    };
                    for name in names {
                        let export = if name_type == NAME_ORDINAL {
                            ImportSymbol::Ordinal(ordinal)
                        } else if name_type == NAME_EXPORTAS {
                            ImportSymbol::Name { hint: 0, name }
                        } else {
                            ImportSymbol::Name {
                                hint: 0,
                                name: strip_decoration(&name, name_type),
                            }
                        };
                        lib.imports.push(LibImport {
                            dll: dll.clone(),
                            export,
                            kind,
                        });
                    }
                    continue;
                }
            match ImportFile::parse(&member[..]) {
                Ok(import) => {
                    lib.imports.push(LibImport {
                        dll: String::from_utf8_lossy(import.dll()).into_owned(),
                        export: match import.import() {
                            ImportName::Name(name) => ImportSymbol::Name {
                                hint: 0,
                                name: String::from_utf8_lossy(name).into_owned(),
                            },
                            ImportName::Ordinal(o) => ImportSymbol::Ordinal(o),
                        },
                        kind: match import.import_type() {
                            ImportType::Code => LibImportKind::Code,
                            ImportType::Data => LibImportKind::Data,
                            ImportType::Const => LibImportKind::Const,
                        },
                    });
                }
                Err(_) => lib.other_members += 1,
            }
        }
        lib.imports.sort();
        lib.imports.dedup();
        Ok(lib)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn member(out: &mut Vec<u8>, name: &str, data: &[u8]) {
        let header = format!(
            "{:<16}{:<12}{:<6}{:<6}{:<8}{:<10}`\n",
            name,
            "0",
            "0",
            "0",
            "644",
            data.len()
        );
        assert_eq!(header.len(), 60);
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        if data.len() % 2 == 1 {
            out.push(b'\n');
        }
    }

    fn archive(members: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut out = b"!<arch>\n".to_vec();
        for (name, data) in members {
            member(&mut out, name, data);
        }
        out
    }

    fn short_import(dll: &str, symbol: &str, flags: u16) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&SIG1.to_le_bytes());
        out.extend_from_slice(&SIG2.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // version
        out.extend_from_slice(&0x8664u16.to_le_bytes()); // AMD64
        out.extend_from_slice(&0u32.to_le_bytes()); // timestamp
        // Short import files store the public symbol first, then the DLL.
        let strings = [symbol.as_bytes(), dll.as_bytes()].join(&b'\0');
        out.extend_from_slice(&((strings.len() + 1) as u32).to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes()); // hint
        out.extend_from_slice(&flags.to_le_bytes());
        out.extend_from_slice(&strings);
        out.push(0);
        out
    }

    fn long_descriptor(ordinal: u16, flags: u16) -> Vec<u8> {
        let mut out = vec![0u8; 20];
        out[2..4].copy_from_slice(&SIG2.to_le_bytes());
        out[8..10].copy_from_slice(&0x8664u16.to_le_bytes());
        out[16..18].copy_from_slice(&ordinal.to_le_bytes());
        out[18..20].copy_from_slice(&flags.to_le_bytes());
        out
    }

    /// A minimal COFF object with undefined external symbols.
    fn coff_object(names: &[&str]) -> Vec<u8> {
        let mut strings = Vec::new();
        let mut offsets = Vec::new();
        for name in names {
            offsets.push(strings.len() as u32 + 4);
            strings.extend_from_slice(name.as_bytes());
            strings.push(0);
        }
        let mut out = Vec::new();
        out.extend_from_slice(&0x8664u16.to_le_bytes()); // machine
        out.extend_from_slice(&0u16.to_le_bytes()); // sections
        out.extend_from_slice(&0u32.to_le_bytes()); // timestamp
        out.extend_from_slice(&20u32.to_le_bytes()); // symbol table offset
        out.extend_from_slice(&(names.len() as u32).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // optional header
        out.extend_from_slice(&0u16.to_le_bytes()); // characteristics
        for offset in offsets {
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes()); // value
            out.extend_from_slice(&0i16.to_le_bytes()); // undefined section
            out.extend_from_slice(&0u16.to_le_bytes()); // type
            out.push(2); // external
            out.push(0); // no aux records
        }
        out.extend_from_slice(&(strings.len() as u32 + 4).to_le_bytes());
        out.extend_from_slice(&strings);
        out
    }

    #[test]
    fn short_imports_are_read() {
        let bytes = archive(&[(
            "short.o",
            short_import("xsystem.dll", "XSystemGetDeviceType", TYPE_CODE | NAME << 2),
        )]);
        let lib = ImportLibrary::parse(&bytes).unwrap();
        assert_eq!(lib.imports.len(), 1);
        assert_eq!(lib.imports[0].dll, "xsystem.dll");
        assert_eq!(
            lib.imports[0].export,
            ImportSymbol::Name {
                hint: 0,
                name: "XSystemGetDeviceType".into()
            }
        );
        assert_eq!(lib.other_members, 0);
    }

    #[test]
    fn long_imports_take_their_names_from_the_coff_object() {
        let object = coff_object(&[
            "__IMPORT_DESCRIPTOR_xsystem.dll",
            "__imp_XSystemGetDeviceType",
            "__imp_XSystemSetDeviceType",
        ]);
        let bytes = archive(&[
            ("descriptor.o", long_descriptor(0, TYPE_CODE | NAME << 2)),
            ("null.o", vec![0u8; 20]),
            ("thunk.o", vec![0u8; 8]),
            ("thunks.o", object),
        ]);
        let lib = ImportLibrary::parse(&bytes).unwrap();
        let exports: Vec<_> = lib
            .imports
            .iter()
            .filter(|i| i.dll == "xsystem.dll")
            .map(|i| i.export.to_string())
            .collect();
        assert_eq!(exports, ["XSystemGetDeviceType", "XSystemSetDeviceType"]);
        // Descriptor, null descriptor, null thunk and the symbols object.
        assert_eq!(lib.other_members, 4);
    }

    #[test]
    fn long_import_by_ordinal_uses_the_header_value() {
        let object = coff_object(&[
            "__IMPORT_DESCRIPTOR_xsystem.dll",
            "__imp_XSystemGetDeviceType",
        ]);
        let bytes = archive(&[
            ("descriptor.o", long_descriptor(42, NAME_ORDINAL)),
            ("thunks.o", object),
        ]);
        let lib = ImportLibrary::parse(&bytes).unwrap();
        assert_eq!(lib.imports.len(), 1);
        assert_eq!(lib.imports[0].export, ImportSymbol::Ordinal(42));
    }

    #[test]
    fn unknown_members_are_reported_not_silently_dropped() {
        let bytes = archive(&[("junk.o", vec![1, 2, 3])]);
        let lib = ImportLibrary::parse(&bytes).unwrap();
        assert!(lib.imports.is_empty());
        assert_eq!(lib.other_members, 1);
    }
}
