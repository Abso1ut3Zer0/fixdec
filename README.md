# fixdec

**High-performance fixed-point decimal arithmetic for financial calculations and cryptocurrency.**

`fixdec` provides two blazingly fast decimal types with **fixed precision**, optimized for performance-critical applications where the precision requirements are known at compile time. If you need configurable precision at runtime, use [`rust_decimal`](https://crates.io/crates/rust_decimal). If you need maximum speed with fixed precision, use `fixdec`.

## When to Use fixdec

✅ **Use fixdec when:**
- You need **maximum performance** in financial calculations
- Your precision requirements are **fixed and known** (8 or 12 decimal places)
- You're building **high-frequency trading** systems
- You're working with **cryptocurrency** (DeFi, DEX, pricing engines)
- You need `no_std` support for embedded or WASM
- You want exact decimal arithmetic without floating-point errors

❌ **Don't use fixdec when:**
- You need **configurable precision** at runtime
- You need more than 12 decimal places
- You need arbitrary precision arithmetic
- Performance is not a critical concern

## Performance

`fixdec` is designed for speed. Here's how it compares to `rust_decimal`:

| Operation | D64 | rust_decimal | Speedup |
|-----------|-----|--------------|---------|
| Addition | 0.62 ns | 7.51 ns | **12x faster** |
| Subtraction | 0.60 ns | 7.53 ns | **12.5x faster** |
| Multiplication | 1.24 ns | 7.57 ns | **6x faster** |
| Division | 3.70 ns | 21.29 ns | **5.7x faster** |
| Square root | 102.87 ns | 750.89 ns | **7.3x faster** |
| Power (powi) | 5.15 ns | 42.67 ns | **8.3x faster** |
| Bincode serialize | 6.0 ns | 62.9 ns | **10.5x faster** |

*Benchmarks run on [your system specs]. See `benches/` for details.*

## Types

### `D64` - Traditional Finance
- **Storage**: 64-bit (8 bytes)
- **Precision**: 8 decimal places (0.00000001)
- **Range**: ±92,233,720,368.54775807 (±92 billion)
- **Use cases**: Traditional financial applications & trading systems

### `D96` - Cryptocurrency
- **Storage**: 128-bit (16 bytes, but only 96 bits used)
- **Precision**: 12 decimal places (0.000000000001)
- **Range**: ±39,614,081,257,132.168796771975 (±39 trillion)
- **Use cases**: Cryptocurrency pricing, DeFi protocols, gas calculations, extreme price ranges

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
fixdec = "0.1"
```

### Basic Usage

```rust
use fixdec::D64;
use core::str::FromStr;

// Create from strings
let price = D64::from_str("1234.56")?;
let quantity = D64::from_i32(100);

// Fast arithmetic
let total = price * quantity;
assert_eq!(total.to_string(), "123456");

// Checked arithmetic
let result = price.checked_mul(quantity).ok_or("overflow")?;

// Financial operations
let fee = total.percent_of(D64::from_str("0.1")?)?; // 0.1% fee
```

### Cryptocurrency Example

```rust
use fixdec::D96;
use core::str::FromStr;

// High precision for crypto
let eth_price = D96::from_str("2500.123456789012")?;
let amount = D96::from_str("0.5")?;
let total_value = eth_price * amount;

// Built-in crypto constants
let gas_price = D96::from_i64(50) * D96::GWEI; // 50 gwei
let tx_value = D96::from_str("0.00000001")?;   // 1 satoshi equivalent
```

## Features

### Core Features
- **Fixed precision, maximum speed**: Compile-time precision means zero runtime overhead
- **Exact decimal math**: No floating-point rounding errors
- **Comprehensive arithmetic**: Checked, saturating, and wrapping variants
- **Financial constants**: Basis points, bond fractions (32nds, 64ths), percentages
- **Crypto constants**: Satoshi, gwei, microGwei for blockchain applications

### Optimizations
- **Reciprocal multiplication**: Uses "magic division" with precomputed constants
- **Fast string parsing**: SWAR (SIMD Within A Register) techniques
- **Branchless validation**: Optimized digit checking
- **Binary serialization**: Raw integer encoding for minimal overhead

### Platform Support
- **`no_std` compatible**: Works in embedded systems and WebAssembly
- **Optional `alloc`**: For `Vec` and `String` support
- **Optional `std`**: For `Error` trait and standard library features
- **Serde support**: Efficient JSON and binary serialization

## Feature Flags

```toml
[dependencies]
fixdec = { version = "0.1", features = ["serde"] }
```

| Feature | Description |
|---------|-------------|
| `default` | No additional features (pure `no_std`) |
| `alloc` | Enable `Vec` and `String` support |
| `std` | Enable standard library and `Error` trait |
| `serde` | Enable Serde serialization (requires `alloc`) |
| `full` | Enable all features |

## API Overview

### Construction

```rust
// From integers
D64::from_i32(42)           // Always succeeds
D64::from_i64(1000000)?     // Checked, may overflow
D64::from_u64(1000000)?     // Checked

// From strings (exact)
D64::from_str("123.45")?                    // Errors if > 8 decimals
D64::from_str_lossy("123.456789123")?      // Rounds to 8 decimals

// From floats (lossy)
D64::from_f64(123.45)?      // May lose precision

// From raw scaled values (advanced)
D64::from_raw(12345000000)  // 123.45 in raw form

// From mantissa and scale (rust_decimal compatibility)
D64::with_scale(12345, 2)?  // 123.45 (mantissa=12345, scale=2)
```

### Arithmetic Operations

```rust
// Standard operators (panic on overflow)
let z = x + y;
let z = x - y;
let z = x * y;
let z = x / y;

// Checked (returns Option)
x.checked_add(y)?
x.checked_sub(y)?
x.checked_mul(y)?
x.checked_div(y)?

// Saturating (clamps to min/max)
x.saturating_add(y)
x.saturating_sub(y)
x.saturating_mul(y)
x.saturating_div(y)

// Wrapping (wraps on overflow)
x.wrapping_add(y)
x.wrapping_sub(y)
x.wrapping_mul(y)
x.wrapping_div(y)

// Fast integer multiplication (quantity * price)
price.mul_i64(quantity)?

// Fused multiply-add (one rounding step)
x.mul_add(y, z)?  // (x * y) + z
```

### Rounding

```rust
let x = D64::from_str("123.456789")?;

x.floor()         // 123.00000000
x.ceil()          // 124.00000000
x.round()         // 123.00000000 (banker's rounding)
x.round_dp(2)     // 123.46000000 (round to 2 decimals)
x.trunc()         // 123.00000000 (truncate)
x.fract()         // 0.45678900 (fractional part)
```

### Financial Operations

```rust
// Basis points (1 bp = 0.0001)
let rate = D64::from_basis_points(50)?;  // 0.005 (50 bps)
let bps = rate.to_basis_points();        // 50

// Percentage calculations
let tax = price.percent_of(D64::from_str("8.5")?)?;      // 8.5% of price
let with_markup = price.add_percent(D64::from_str("10")?)?; // price * 1.10

// Financial constants
D64::BASIS_POINT          // 0.0001 (1 bp)
D64::HALF_BASIS_POINT     // 0.00005
D64::THIRTY_SECOND        // 0.03125 (US Treasury bond tick)
D64::SIXTY_FOURTH         // 0.015625
D64::CENT                 // 0.01 (1 cent)
D64::PERCENT              // 0.01 (1%)
```

### Mathematical Operations

```rust
// Square root
let sqrt = x.sqrt()?;

// Integer powers
let squared = x.powi(2)?;
let cubed = x.powi(3)?;

// Reciprocal
let recip = x.recip()?;  // 1/x

// Absolute value
let abs = x.abs();

// Sign
let sign = x.signum();  // -1, 0, or 1
```

### Serialization

With the `serde` feature enabled:

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct Trade {
    price: D64,
    quantity: D64,
}

// JSON: uses string representation for precision
let json = serde_json::to_string(&trade)?;
// {"price":"1234.56","quantity":"100.00000000"}

// Bincode: uses raw i64 (extremely fast)
let bytes = bincode::serialize(&trade)?;
// Just 16 bytes (8 bytes per D64)
```

### Constants

#### D64 Constants
```rust
D64::ZERO                    // 0
D64::ONE                     // 1.0
D64::TEN                     // 10.0
D64::HUNDRED                 // 100.0
D64::THOUSAND                // 1000.0

// Currency
D64::CENT                    // 0.01
D64::MIL                     // 0.001

// Basis points
D64::BASIS_POINT             // 0.0001
D64::HALF_BASIS_POINT        // 0.00005

// Bond pricing
D64::THIRTY_SECOND           // 1/32
D64::SIXTY_FOURTH            // 1/64

// Legacy equity fractions
D64::EIGHTH                  // 1/8
D64::SIXTEENTH               // 1/16
```

#### D96 Constants (includes all D64 constants plus)
```rust
// Cryptocurrency
D96::SATOSHI                 // 0.00000001 (Bitcoin)
D96::GWEI                    // 0.000000001 (Ethereum gas unit)
D96::MICRO_GWEI              // 0.000000000001 (minimum precision)
D96::KILO_WEI                // 0.000000000001 (1000 wei)
```

## no_std Usage

`fixdec` works in `no_std` environments by default:

```rust
#![no_std]

use fixdec::D64;

// All core operations work without std
let x = D64::from_i32(42);
let y = D64::from_i32(10);
let z = x / y;

// String formatting requires alloc
#[cfg(feature = "alloc")]
extern crate alloc;
```

## Real-World Examples

### Portfolio P&L Calculation
```rust
use fixdec::D64;

struct Position {
    symbol: &'static str,
    quantity: i64,
    entry_price: D64,
    current_price: D64,
}

fn calculate_pnl(position: &Position) -> D64 {
    let entry_value = position.entry_price.mul_i64(position.quantity).unwrap();
    let current_value = position.current_price.mul_i64(position.quantity).unwrap();
    current_value - entry_value
}

let pos = Position {
    symbol: "AAPL",
    quantity: 1000,
    entry_price: D64::from_str("150.25")?,
    current_price: D64::from_str("155.75")?,
};

let pnl = calculate_pnl(&pos);
assert_eq!(pnl.to_string(), "5500");  // $5,500 profit
```

### DeFi Token Swap Calculation
```rust
use fixdec::D96;

fn calculate_swap_output(
    amount_in: D96,
    reserve_in: D96,
    reserve_out: D96,
    fee_bps: i64,  // e.g., 30 for 0.3%
) -> Option<D96> {
    let fee_multiplier = D96::from_basis_points(10000 - fee_bps)?;
    let amount_in_with_fee = amount_in.checked_mul(fee_multiplier)?
        .checked_div(D96::from_i32(10000))?;
    
    let numerator = amount_in_with_fee.checked_mul(reserve_out)?;
    let denominator = reserve_in.checked_add(amount_in_with_fee)?;
    
    numerator.checked_div(denominator)
}

let amount_out = calculate_swap_output(
    D96::from_str("1.0")?,      // 1 ETH in
    D96::from_str("1000")?,     // Reserve: 1000 ETH
    D96::from_str("2000000")?,  // Reserve: 2M USDC
    30,                          // 0.3% fee
)?;
```

### Bond Price Calculation (32nds)
```rust
use fixdec::D64;

// US Treasury bonds are quoted in 32nds
// e.g., "99-16" means 99 + 16/32 = 99.5
fn parse_bond_price(whole: i64, thirty_seconds: i64) -> D64 {
    let whole_part = D64::from_i64(whole);
    let fraction = D64::THIRTY_SECOND.mul_i64(thirty_seconds).unwrap();
    whole_part + fraction
}

let price = parse_bond_price(99, 16);
assert_eq!(price.to_string(), "99.5");
```

## Comparison with rust_decimal

| Feature | fixdec | rust_decimal |
|---------|--------|--------------|
| **Precision** | Fixed (8 or 12 decimals) | Configurable (0-28 decimals) |
| **Performance** | **6-12x faster** | Slower due to flexibility |
| **Use case** | Performance-critical with known precision | General purpose, configurable precision |
| **no_std** | ✅ Full support | ✅ Full support |
| **Serialization** | ✅ Optimized for binary | ✅ General purpose |
| **API similarity** | High (easy migration) | - |

`fixdec` is built for **speed** when you know your precision requirements. `rust_decimal` is built for **flexibility** when you need configurable precision.

## Safety and Correctness

- **Overflow behavior**: All arithmetic operations have `checked`, `saturating`, and `wrapping` variants
- **No unsafe code**: Pure safe Rust (except in tests)
- **Extensive testing**: Property-based tests with `proptest` verify correctness against baseline implementations
- **Banker's rounding**: IEEE 754 round-half-to-even for tie-breaking

## Contributing

Contributions are welcome! Areas of interest:
- Performance optimizations
- Additional financial operations
- Documentation improvements
- Bug reports and fixes

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
