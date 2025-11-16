//! High-performance fixed-point decimal types for financial calculations
//!
//! This library provides two decimal types optimized for speed:
//! - `D64`: 64-bit with 8 decimal places (±92 billion range)
//! - `D128`: 128-bit with 18 decimal places (±170 trillion range)

#![no_std]
#![cfg_attr(test, allow(unused_imports))]

#[cfg(test)]
extern crate std;

#[cfg(feature = "alloc")]
extern crate alloc;

mod d128;
mod d64;

pub use d64::D64;
pub use d128::D128;

use thiserror::Error;

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecimalError {
    #[error("overflow: value too large to represent")]
    Overflow,

    #[error("underflow: value too small to represent")]
    Underflow,

    #[error("division by zero")]
    DivisionByZero,

    #[error("invalid string format")]
    InvalidFormat,

    #[error("precision loss would occur")]
    PrecisionLoss,
}

pub type Result<T> = core::result::Result<T, DecimalError>;
