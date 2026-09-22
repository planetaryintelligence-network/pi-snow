use thiserror::Error;

#[derive(Debug, Error)]
pub enum IoFormatError {
    #[error("{format} file has no [DATA] section")]
    NoDataSection { format: &'static str },
    #[error("{format} file has no `fields` header")]
    NoFields { format: &'static str },
    #[error("{format} line {line}: expected {expected} values, found {found}")]
    WrongArity {
        format: &'static str,
        line: usize,
        expected: usize,
        found: usize,
    },
    #[error("{format} line {line}: cannot parse {what}: {text:?}")]
    BadValue {
        format: &'static str,
        line: usize,
        what: &'static str,
        text: String,
    },
    #[error("{format}: required header key {key:?} is missing")]
    MissingHeader {
        format: &'static str,
        key: &'static str,
    },
    #[error("PRO value appeared before any 0500 timestamp, at line {line}")]
    ValueBeforeTimestamp { line: usize },
    #[error("duplicate {what} {value:?}")]
    Duplicate { what: &'static str, value: String },
}
