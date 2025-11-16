//! High-performance fixed-point decimal types for financial calculations
//!
//! This library provides two decimal types optimized for speed:
//! - `D64`: 64-bit with 8 decimal places (±92 billion range)
//! - `D128`: 128-bit with 18 decimal places (±170 trillion range)

#![no_std]

mod d128;
mod d64;

pub use d64::D64;
pub use d128::D128;
