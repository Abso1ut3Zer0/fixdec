use core::fmt;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::str::FromStr;

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

#[cfg(feature = "serde")]
use alloc::string::String;

use crate::DecimalError;

/// 64-bit fixed-point decimal with 8 decimal places of precision.
///
/// Range: ±92,233,720,368.54775807
/// Precision: 0.00000001
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct D64 {
    value: i64,
}

// ============================================================================
// Constants
// ============================================================================

impl D64 {
    /// The scale factor: 10^8
    pub const SCALE: i64 = 100_000_000;

    /// The number of decimal places
    pub const DECIMALS: u8 = 8;

    /// Maximum value: ~92 billion
    pub const MAX: Self = Self { value: i64::MAX };

    /// Minimum value: ~-92 billion
    pub const MIN: Self = Self { value: i64::MIN };

    /// Zero
    pub const ZERO: Self = Self { value: 0 };

    /// One (1.0)
    pub const ONE: Self = Self { value: Self::SCALE };

    /// Ten (10.0)
    pub const TEN: Self = Self {
        value: 10 * Self::SCALE,
    };

    /// One hundred (100.0)
    pub const HUNDRED: Self = Self {
        value: 100 * Self::SCALE,
    };

    /// One thousand (1000.0)
    pub const THOUSAND: Self = Self {
        value: 1000 * Self::SCALE,
    };

    // ===== Currency Fractions =====

    /// One cent (0.01) - USD, EUR, and most currency smallest unit
    pub const CENT: Self = Self {
        value: Self::SCALE / 100,
    };

    /// One mil (0.001) - used in some pricing contexts
    pub const MIL: Self = Self {
        value: Self::SCALE / 1000,
    };

    // ===== Basis Points =====

    /// One basis point (0.0001) - 1 bps, common in interest rates
    pub const BASIS_POINT: Self = Self {
        value: Self::SCALE / 10_000,
    };

    /// One half basis point (0.00005)
    pub const HALF_BASIS_POINT: Self = Self {
        value: Self::SCALE / 20_000,
    };

    /// One quarter basis point (0.000025)
    pub const QUARTER_BASIS_POINT: Self = Self {
        value: Self::SCALE / 40_000,
    };

    // ===== Bond Price Fractions (32nds and 64ths) =====

    /// One thirty-second (1/32 = 0.03125) - US Treasury bond price tick
    pub const THIRTY_SECOND: Self = Self {
        value: Self::SCALE / 32,
    };

    /// One sixty-fourth (1/64 = 0.015625) - US Treasury bond price tick
    pub const SIXTY_FOURTH: Self = Self {
        value: Self::SCALE / 64,
    };

    /// One half of a thirty-second (1/64) - alternative name
    pub const HALF_THIRTY_SECOND: Self = Self::SIXTY_FOURTH;

    /// One quarter of a thirty-second (1/128 = 0.0078125)
    pub const QUARTER_THIRTY_SECOND: Self = Self {
        value: Self::SCALE / 128,
    };

    /// One eighth of a thirty-second (1/256 = 0.00390625)
    pub const EIGHTH_THIRTY_SECOND: Self = Self {
        value: Self::SCALE / 256,
    };

    // ===== Equity Price Fractions =====

    /// One eighth (0.125) - legacy stock pricing increment
    pub const EIGHTH: Self = Self {
        value: Self::SCALE / 8,
    };

    /// One sixteenth (0.0625) - legacy stock pricing increment
    pub const SIXTEENTH: Self = Self {
        value: Self::SCALE / 16,
    };

    /// One quarter (0.25)
    pub const QUARTER: Self = Self {
        value: Self::SCALE / 4,
    };

    /// One half (0.5)
    pub const HALF: Self = Self {
        value: Self::SCALE / 2,
    };

    // ===== Percentage Helpers =====

    /// One percent (0.01) - same as CENT but semantic clarity
    pub const PERCENT: Self = Self::CENT;

    /// Ten basis points (0.001) - same as MIL
    pub const TEN_BPS: Self = Self::MIL;

    /// One hundred basis points (0.01) - same as PERCENT
    pub const HUNDRED_BPS: Self = Self::PERCENT;
}

// ============================================================================
// Constructors and Raw Access
// ============================================================================
impl Default for D64 {
    fn default() -> Self {
        Self::ZERO
    }
}

impl D64 {
    /// Creates a new D64 from a raw scaled value.
    ///
    /// # Safety
    /// The caller must ensure the value is properly scaled by 10^8.
    #[inline(always)]
    pub const fn from_raw(value: i64) -> Self {
        Self { value }
    }

    /// Returns the raw internal value (scaled by 10^8).
    #[inline(always)]
    pub const fn to_raw(self) -> i64 {
        self.value
    }

    /// Creates a D64 from integer and fractional parts at compile time
    /// Example: `new(123, 45_000_000)` → 123.45
    ///
    /// The fractional part should always be positive.
    /// For negative numbers, use a negative integer part:
    /// `new(-123, 45_000_000)` → -123.45
    ///
    /// # Panics
    /// Panics if the value would overflow i64 range.
    pub const fn new(integer: i64, fractional: i64) -> Self {
        let scaled = match integer.checked_mul(Self::SCALE) {
            Some(v) => v,
            None => panic!("overflow in D64::new: integer part too large"),
        };

        let value = if integer >= 0 {
            match scaled.checked_add(fractional) {
                Some(v) => v,
                None => panic!("overflow in D64::new: result too large"),
            }
        } else {
            match scaled.checked_sub(fractional) {
                Some(v) => v,
                None => panic!("overflow in D64::new: result too large"),
            }
        };

        Self { value }
    }

    /// Create from basis points (1 bp = 0.0001)
    /// Example: `from_basis_points(100)` → 0.01 (1%)
    pub const fn from_basis_points(bps: i64) -> Option<Self> {
        // 1 bp = 0.0001 = SCALE / 10_000
        // So bps basis points = (bps * SCALE) / 10_000
        let numerator = match bps.checked_mul(Self::SCALE) {
            Some(v) => v,
            None => return None,
        };

        Some(Self {
            value: numerator / 10_000,
        })
    }

    /// Convert to basis points
    /// Example: `D64::from_str("0.01").unwrap().to_basis_points()` → 100
    pub const fn to_basis_points(self) -> i64 {
        // Reverse: (value * 10_000) / SCALE
        self.value * 10_000 / Self::SCALE
    }
}

// ============================================================================
// Arithmetic Operations - Addition
// ============================================================================

impl D64 {
    /// Checked addition. Returns `None` if overflow occurred.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn checked_add(self, rhs: Self) -> Option<Self> {
        if let Some(result) = self.value.checked_add(rhs.value) {
            Some(Self { value: result })
        } else {
            None
        }
    }

    /// Saturating addition. Clamps on overflow.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn saturating_add(self, rhs: Self) -> Self {
        Self {
            value: self.value.saturating_add(rhs.value),
        }
    }

    /// Wrapping addition. Wraps on overflow.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn wrapping_add(self, rhs: Self) -> Self {
        Self {
            value: self.value.wrapping_add(rhs.value),
        }
    }

    /// Checked addition. Returns an error if overflow occurred.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn try_add(self, rhs: Self) -> crate::Result<Self> {
        match self.checked_add(rhs) {
            Some(result) => Ok(result),
            None => Err(DecimalError::Overflow),
        }
    }
}

// ============================================================================
// Arithmetic Operations - Subtraction
// ============================================================================

impl D64 {
    /// Checked subtraction. Returns `None` if overflow occurred.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn checked_sub(self, rhs: Self) -> Option<Self> {
        if let Some(result) = self.value.checked_sub(rhs.value) {
            Some(Self { value: result })
        } else {
            None
        }
    }

    /// Saturating subtraction. Clamps on overflow.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn saturating_sub(self, rhs: Self) -> Self {
        Self {
            value: self.value.saturating_sub(rhs.value),
        }
    }

    /// Wrapping subtraction. Wraps on overflow.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn wrapping_sub(self, rhs: Self) -> Self {
        Self {
            value: self.value.wrapping_sub(rhs.value),
        }
    }

    /// Checked subtraction. Returns an error if overflow occurred.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn try_sub(self, rhs: Self) -> crate::Result<Self> {
        match self.checked_sub(rhs) {
            Some(result) => Ok(result),
            None => Err(DecimalError::Overflow),
        }
    }
}

// ============================================================================
// Arithmetic Operations - Multiplication
// ============================================================================

impl D64 {
    /// Checked multiplication. Returns `None` if overflow occurred.
    ///
    /// Internally widens to i128 to prevent intermediate overflow.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn checked_mul(self, rhs: Self) -> Option<Self> {
        let a = self.value as i128;
        let b = rhs.value as i128;

        // Multiply then divide by scale
        let result = a * b / Self::SCALE as i128;

        // Check if result fits in i64
        if result > i64::MAX as i128 || result < i64::MIN as i128 {
            None
        } else {
            Some(Self {
                value: result as i64,
            })
        }
    }

    /// Saturating multiplication. Clamps on overflow.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn saturating_mul(self, rhs: Self) -> Self {
        let a = self.value as i128;
        let b = rhs.value as i128;

        let result = a * b / Self::SCALE as i128;

        if result > i64::MAX as i128 {
            Self::MAX
        } else if result < i64::MIN as i128 {
            Self::MIN
        } else {
            Self {
                value: result as i64,
            }
        }
    }

    /// Wrapping multiplication. Wraps on overflow.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn wrapping_mul(self, rhs: Self) -> Self {
        let a = self.value as i128;
        let b = rhs.value as i128;

        let result = a * b / Self::SCALE as i128;

        Self {
            value: result as i64,
        }
    }

    /// Checked multiplication. Returns an error if overflow occurred.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn try_mul(self, rhs: Self) -> crate::Result<Self> {
        match self.checked_mul(rhs) {
            Some(result) => Ok(result),
            None => Err(DecimalError::Overflow),
        }
    }

    /// Multiply by an integer (faster than general multiplication)
    /// Useful for: quantity * price, shares * rate, etc.
    pub const fn mul_i64(self, rhs: i64) -> Option<Self> {
        match self.value.checked_mul(rhs) {
            Some(result) => Some(Self { value: result }),
            None => None,
        }
    }

    pub const fn try_mul_i64(self, rhs: i64) -> crate::Result<Self> {
        match self.mul_i64(rhs) {
            Some(v) => Ok(v),
            None => Err(DecimalError::Overflow),
        }
    }

    /// Computes (self * mul) + add with only one rounding step
    /// More accurate and faster than separate mul + add
    pub const fn mul_add(self, mul: Self, add: Self) -> Option<Self> {
        let a = self.value as i128;
        let b = mul.value as i128;
        let c = add.value as i128;

        // (a * b) / SCALE + c
        let product = a * b / Self::SCALE as i128;
        let result = product + c;

        if result > i64::MAX as i128 || result < i64::MIN as i128 {
            None
        } else {
            Some(Self {
                value: result as i64,
            })
        }
    }
}

// ============================================================================
// Arithmetic Operations - Division
// ============================================================================

impl D64 {
    /// Checked division. Returns `None` if `rhs` is zero or overflow occurred.
    ///
    /// Internally widens to i128 to maintain precision.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn checked_div(self, rhs: Self) -> Option<Self> {
        if rhs.value == 0 {
            return None;
        }

        let a = self.value as i128;
        let b = rhs.value as i128;

        // Multiply by scale then divide
        let result = (a * Self::SCALE as i128) / b;

        // Check if result fits in i64
        if result > i64::MAX as i128 || result < i64::MIN as i128 {
            None
        } else {
            Some(Self {
                value: result as i64,
            })
        }
    }

    /// Saturating division. Clamps on overflow. Returns zero if `rhs` is zero.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn saturating_div(self, rhs: Self) -> Self {
        if rhs.value == 0 {
            return Self::ZERO;
        }

        let a = self.value as i128;
        let b = rhs.value as i128;

        let result = (a * Self::SCALE as i128) / b;

        if result > i64::MAX as i128 {
            Self::MAX
        } else if result < i64::MIN as i128 {
            Self::MIN
        } else {
            Self {
                value: result as i64,
            }
        }
    }

    /// Wrapping division. Wraps on overflow. Returns zero if `rhs` is zero.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn wrapping_div(self, rhs: Self) -> Self {
        if rhs.value == 0 {
            return Self::ZERO;
        }

        let a = self.value as i128;
        let b = rhs.value as i128;

        let result = (a * Self::SCALE as i128) / b;

        Self {
            value: result as i64,
        }
    }

    /// Checked division. Returns an error if `rhs` is zero or overflow occurred.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn try_div(self, rhs: Self) -> crate::Result<Self> {
        if rhs.value == 0 {
            return Err(DecimalError::DivisionByZero);
        }
        match self.checked_div(rhs) {
            Some(result) => Ok(result),
            None => Err(DecimalError::Overflow),
        }
    }
}

// ============================================================================
// Arithmetic Operations - Negation
// ============================================================================

impl D64 {
    /// Checked negation. Returns `None` if the result would overflow.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn checked_neg(self) -> Option<Self> {
        if let Some(result) = self.value.checked_neg() {
            Some(Self { value: result })
        } else {
            None
        }
    }

    /// Saturating negation. Clamps on overflow.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn saturating_neg(self) -> Self {
        Self {
            value: self.value.saturating_neg(),
        }
    }

    /// Wrapping negation. Wraps on overflow.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn wrapping_neg(self) -> Self {
        Self {
            value: self.value.wrapping_neg(),
        }
    }

    /// Checked negation. Returns an error if overflow occurred.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn try_neg(self) -> crate::Result<Self> {
        match self.checked_neg() {
            Some(result) => Ok(result),
            None => Err(DecimalError::Overflow),
        }
    }
}

// ============================================================================
// Arithmetic Operations - Absolute Value
// ============================================================================

impl D64 {
    /// Returns the absolute value of `self`.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn abs(self) -> Self {
        Self {
            value: if self.value < 0 {
                -self.value
            } else {
                self.value
            },
        }
    }

    /// Checked absolute value. Returns `None` if the result would overflow.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn checked_abs(self) -> Option<Self> {
        if self.value == i64::MIN {
            None
        } else {
            Some(self.abs())
        }
    }

    /// Saturating absolute value. Clamps on overflow.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn saturating_abs(self) -> Self {
        if self.value == i64::MIN {
            Self::MAX
        } else {
            self.abs()
        }
    }

    /// Wrapping absolute value. Wraps on overflow.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn wrapping_abs(self) -> Self {
        Self {
            value: self.value.wrapping_abs(),
        }
    }

    /// Checked absolute value. Returns an error if overflow occurred.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn try_abs(self) -> crate::Result<Self> {
        match self.checked_abs() {
            Some(result) => Ok(result),
            None => Err(DecimalError::Overflow),
        }
    }
}

// ============================================================================
// Sign Operations
// ============================================================================

impl D64 {
    /// Returns `true` if `self` is positive.
    #[inline(always)]
    pub const fn is_positive(self) -> bool {
        self.value > 0
    }

    /// Returns `true` if `self` is negative.
    #[inline(always)]
    pub const fn is_negative(self) -> bool {
        self.value < 0
    }

    /// Returns `true` if `self` is zero.
    #[inline(always)]
    pub const fn is_zero(self) -> bool {
        self.value == 0
    }

    /// Returns the sign of `self` as -1, 0, or 1.
    #[inline(always)]
    pub const fn signum(self) -> i32 {
        if self.value > 0 {
            1
        } else if self.value < 0 {
            -1
        } else {
            0
        }
    }
}

// ============================================================================
// Comparison Utilities
// ============================================================================

impl D64 {
    /// Returns the minimum of two values.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn min(self, other: Self) -> Self {
        if self.value < other.value {
            self
        } else {
            other
        }
    }

    /// Returns the maximum of two values.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn max(self, other: Self) -> Self {
        if self.value > other.value {
            self
        } else {
            other
        }
    }

    /// Restricts a value to a certain interval.
    ///
    /// Returns `max` if `self` is greater than `max`, and `min` if `self` is less than `min`.
    /// Otherwise returns `self`.
    ///
    /// # Panics
    ///
    /// Panics if `min > max`.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn clamp(self, min: Self, max: Self) -> Self {
        assert!(
            min.value <= max.value,
            "min must be less than or equal to max"
        );
        if self.value < min.value {
            min
        } else if self.value > max.value {
            max
        } else {
            self
        }
    }
}

// ============================================================================
// Rounding Operations
// ============================================================================

impl D64 {
    /// Returns the largest integer less than or equal to `self`.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn floor(self) -> Self {
        let remainder = self.value % Self::SCALE;
        if remainder >= 0 {
            Self {
                value: self.value - remainder,
            }
        } else {
            Self {
                value: self.value - remainder - Self::SCALE,
            }
        }
    }

    /// Returns the smallest integer greater than or equal to `self`.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn ceil(self) -> Self {
        let remainder = self.value % Self::SCALE;
        if remainder > 0 {
            Self {
                value: self.value - remainder + Self::SCALE,
            }
        } else {
            Self {
                value: self.value - remainder,
            }
        }
    }

    /// Returns the integer part of `self`, truncating any fractional part.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn trunc(self) -> Self {
        Self {
            value: (self.value / Self::SCALE) * Self::SCALE,
        }
    }

    /// Returns the fractional part of `self`.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn fract(self) -> Self {
        Self {
            value: self.value % Self::SCALE,
        }
    }

    /// Rounds to the nearest integer, using banker's rounding (round half to even).
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn round(self) -> Self {
        let quotient = self.value / Self::SCALE;
        let remainder = self.value % Self::SCALE;
        let half = Self::SCALE / 2;

        let rounded_quotient = if remainder > half {
            quotient + 1
        } else if remainder < -half {
            quotient - 1
        } else if remainder == half {
            // Banker's rounding: round to even
            if quotient % 2 == 0 {
                quotient
            } else {
                quotient + 1
            }
        } else if remainder == -half {
            // Banker's rounding: round to even
            if quotient % 2 == 0 {
                quotient
            } else {
                quotient - 1
            }
        } else {
            quotient
        };

        Self {
            value: rounded_quotient * Self::SCALE,
        }
    }

    /// Rounds to the specified number of decimal places.
    ///
    /// # Arguments
    /// * `decimal_places` - Number of decimal places to round to (0-8 for D64)
    ///
    /// # Panics
    /// Panics if `decimal_places > Self::DECIMALS`
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn round_dp(self, decimal_places: u8) -> Self {
        assert!(
            decimal_places <= Self::DECIMALS,
            "decimal_places must be <= DECIMALS"
        );

        if decimal_places == Self::DECIMALS {
            return self;
        }

        // Calculate the rounding scale (e.g., for 2 dp: 10^6)
        let scale_reduction = Self::DECIMALS - decimal_places;
        let rounding_factor = const_pow10(scale_reduction);

        // Round to nearest
        let quotient = self.value / rounding_factor;
        let remainder = self.value % rounding_factor;
        let half = rounding_factor / 2;

        let rounded = if remainder > half {
            quotient + 1
        } else if remainder < -half {
            quotient - 1
        } else if remainder == half {
            // Banker's rounding
            if quotient % 2 == 0 {
                quotient
            } else {
                quotient + 1
            }
        } else if remainder == -half {
            if quotient % 2 == 0 {
                quotient
            } else {
                quotient - 1
            }
        } else {
            quotient
        };

        Self {
            value: rounded * rounding_factor,
        }
    }
}

// ============================================================================
// Mathematical Operations
// ============================================================================

impl D64 {
    /// Returns the reciprocal (multiplicative inverse) of `self`.
    ///
    /// Returns `None` if `self` is zero.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn recip(self) -> Option<Self> {
        if self.value == 0 {
            None
        } else {
            Self::ONE.checked_div(self)
        }
    }

    /// Checked reciprocal. Returns an error if `self` is zero.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn try_recip(self) -> crate::Result<Self> {
        match self.recip() {
            Some(result) => Ok(result),
            None => Err(DecimalError::DivisionByZero),
        }
    }

    /// Raises `self` to an integer power.
    ///
    /// Returns `None` if overflow occurs.
    ///
    /// # Performance
    /// Uses exponentiation by squaring for efficiency.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn powi(self, mut exp: i32) -> Option<Self> {
        if exp == 0 {
            return Some(Self::ONE);
        }

        if exp < 0 {
            // For negative exponents, compute 1 / (self ^ |exp|)
            let pos_result = match self.powi(-exp) {
                Some(r) => r,
                None => return None,
            };
            return Self::ONE.checked_div(pos_result);
        }

        let mut base = self;
        let mut result = Self::ONE;

        // Exponentiation by squaring
        while exp > 0 {
            if exp % 2 == 1 {
                result = match result.checked_mul(base) {
                    Some(r) => r,
                    None => return None,
                };
            }
            if exp > 1 {
                base = match base.checked_mul(base) {
                    Some(b) => b,
                    None => return None,
                };
            }
            exp /= 2;
        }

        Some(result)
    }

    /// Checked integer power. Returns an error if overflow occurred.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn try_powi(self, exp: i32) -> crate::Result<Self> {
        match self.powi(exp) {
            Some(result) => Ok(result),
            None => Err(DecimalError::Overflow),
        }
    }

    /// Returns the square root of `self` using Newton's method.
    ///
    /// Returns `None` if `self` is negative.
    ///
    /// # Performance
    /// Uses a maximum of 20 iterations of Newton's method.
    /// Typically converges in 5-10 iterations for most values.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn sqrt(self) -> Option<Self> {
        if self.value < 0 {
            return None;
        }

        if self.value == 0 {
            return Some(Self::ZERO);
        }

        if self.value == Self::SCALE {
            return Some(Self::ONE);
        }

        // Newton's method: x_{n+1} = (x_n + S/x_n) / 2
        // We work in i128 to avoid overflow

        // Initial guess: use the integer square root of the raw value
        let raw_sqrt_approx = isqrt(self.value as u64) as i128;
        let mut x = raw_sqrt_approx * (Self::SCALE as i128).isqrt();

        const MAX_ITERATIONS: u32 = 20;
        let s = self.value as i128 * Self::SCALE as i128; // Scale up for precision

        let mut i = 0;
        while i < MAX_ITERATIONS {
            let x_next = (x + s / x) / 2;

            // Check for convergence (difference less than 1)
            if (x_next - x).abs() <= 1 {
                // Final result, scaled back down
                let result = x_next;
                if result > i64::MAX as i128 || result < i64::MIN as i128 {
                    return None;
                }
                return Some(Self {
                    value: result as i64,
                });
            }

            x = x_next;
            i += 1;
        }

        // Return best approximation after max iterations
        if x > i64::MAX as i128 || x < i64::MIN as i128 {
            None
        } else {
            Some(Self { value: x as i64 })
        }
    }

    /// Checked square root. Returns an error if `self` is negative.
    #[inline(always)]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn try_sqrt(self) -> crate::Result<Self> {
        match self.sqrt() {
            Some(result) => Ok(result),
            None => Err(DecimalError::InvalidFormat),
        }
    }
}

// ============================================================================
// Integer Conversions
// ============================================================================

impl D64 {
    /// Creates a D64 from an i64 integer.
    #[inline(always)]
    pub const fn from_i64(value: i64) -> Option<Self> {
        // Check if multiplication would overflow
        match value.checked_mul(Self::SCALE) {
            Some(scaled) => Some(Self { value: scaled }),
            None => None,
        }
    }

    /// Creates a D64 from an i32 integer (always succeeds).
    #[inline(always)]
    pub const fn from_i32(value: i32) -> Self {
        Self {
            value: value as i64 * Self::SCALE,
        }
    }

    /// Creates a D64 from a u32 integer (always succeeds).
    #[inline(always)]
    pub const fn from_u32(value: u32) -> Self {
        Self {
            value: value as i64 * Self::SCALE,
        }
    }

    /// Creates a D64 from a u64 integer.
    #[inline(always)]
    pub const fn from_u64(value: u64) -> Option<Self> {
        if value > i64::MAX as u64 / Self::SCALE as u64 {
            None
        } else {
            Some(Self {
                value: value as i64 * Self::SCALE,
            })
        }
    }

    /// Converts to i64, truncating any fractional part.
    #[inline(always)]
    pub const fn to_i64(self) -> i64 {
        self.value / Self::SCALE
    }

    /// Converts to i64, rounding to nearest (banker's rounding on ties).
    #[inline(always)]
    pub const fn to_i64_round(self) -> i64 {
        let quotient = self.value / Self::SCALE;
        let remainder = self.value % Self::SCALE;
        let half = Self::SCALE / 2;

        if remainder > half {
            quotient + 1
        } else if remainder < -half {
            quotient - 1
        } else if remainder == half || remainder == -half {
            // Banker's rounding: round to even
            if quotient % 2 == 0 {
                quotient
            } else {
                quotient + 1
            }
        } else {
            quotient
        }
    }

    /// Creates a D64 from an i64, returning an error on overflow.
    #[inline(always)]
    pub const fn try_from_i64(value: i64) -> crate::Result<Self> {
        match Self::from_i64(value) {
            Some(v) => Ok(v),
            None => Err(DecimalError::Overflow),
        }
    }

    /// Creates a D64 from a u64, returning an error on overflow.
    #[inline(always)]
    pub const fn try_from_u64(value: u64) -> crate::Result<Self> {
        match Self::from_u64(value) {
            Some(v) => Ok(v),
            None => Err(DecimalError::Overflow),
        }
    }
}

// ============================================================================
// Float Conversions
// ============================================================================

impl D64 {
    /// Creates a D64 from an f64.
    ///
    /// Returns `None` if the value is NaN, infinite, or out of range.
    #[inline(always)]
    pub fn from_f64(value: f64) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }

        let scaled = value * Self::SCALE as f64;

        if scaled > i64::MAX as f64 || scaled < i64::MIN as f64 {
            return None;
        }

        Some(Self {
            value: scaled.round() as i64,
        })
    }

    /// Converts to f64.
    ///
    /// Note: May lose precision for very large values.
    #[inline(always)]
    pub fn to_f64(self) -> f64 {
        self.value as f64 / Self::SCALE as f64
    }

    /// Creates a D64 from an f32.
    #[inline(always)]
    pub fn from_f32(value: f32) -> Option<Self> {
        Self::from_f64(value as f64)
    }

    /// Converts to f32.
    ///
    /// Note: May lose precision.
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        self.to_f64() as f32
    }

    /// Creates a D64 from an f64, returning an error if invalid.
    #[inline(always)]
    pub fn try_from_f64(value: f64) -> crate::Result<Self> {
        if value.is_nan() || value.is_infinite() {
            return Err(DecimalError::InvalidFormat);
        }

        let scaled = value * Self::SCALE as f64;

        if scaled > i64::MAX as f64 {
            return Err(DecimalError::Overflow);
        }
        if scaled < i64::MIN as f64 {
            return Err(DecimalError::Underflow);
        }

        Ok(Self {
            value: scaled.round() as i64,
        })
    }
}

// ============================================================================
// Percentage Calculations
// ============================================================================
impl D64 {
    /// Calculate percentage: self * (percent / 100)
    /// Example: 1000.percent_of(5) → 50 (5% of 1000)
    #[inline(always)]
    pub fn percent_of(self, percent: Self) -> Option<Self> {
        self.checked_mul(percent)?.checked_div(Self::HUNDRED)
    }

    /// Add percentage: self * (1 + percent/100)
    /// Example: 1000.add_percent(5) → 1050 (add 5%)
    #[inline(always)]
    pub fn add_percent(self, percent: Self) -> Option<Self> {
        let multiplier = Self::HUNDRED.checked_add(percent)?;
        self.checked_mul(multiplier)?.checked_div(Self::HUNDRED)
    }
}

// ============================================================================
// String Parsing
// ============================================================================

impl D64 {
    /// Parses a decimal string into a D64.
    ///
    /// Supports formats like: "123", "123.45", "-123.45", "0.00000001"
    ///
    /// Zero-allocation implementation using iterators.
    ///
    /// # Errors
    /// Returns `DecimalError::InvalidFormat` if the string is not a valid decimal.
    /// Returns `DecimalError::Overflow` if the value is too large.
    /// Returns `DecimalError::PrecisionLoss` if more than 8 decimal places are provided.
    pub fn from_str_exact(s: &str) -> crate::Result<Self> {
        let s = s.trim();

        if s.is_empty() {
            return Err(DecimalError::InvalidFormat);
        }

        // Check for sign
        let (is_negative, s) = if let Some(rest) = s.strip_prefix('-') {
            (true, rest)
        } else if let Some(rest) = s.strip_prefix('+') {
            (false, rest)
        } else {
            (false, s)
        };

        // After stripping sign, check if empty or has another sign
        if s.is_empty() || s.starts_with('-') || s.starts_with('+') {
            return Err(DecimalError::InvalidFormat);
        }

        // Find decimal point using iterator
        let mut parts = s.split('.');

        let integer_str = parts.next().ok_or(DecimalError::InvalidFormat)?;
        let fractional_str = parts.next();

        // Ensure no more than one decimal point
        if parts.next().is_some() {
            return Err(DecimalError::InvalidFormat);
        }

        // Check for invalid case of just "."
        if integer_str.is_empty() && fractional_str == Some("") {
            return Err(DecimalError::InvalidFormat);
        }

        // Parse integer part
        let integer_part = if integer_str.is_empty() {
            0i64
        } else {
            integer_str
                .parse::<i64>()
                .map_err(|_| DecimalError::InvalidFormat)?
        };

        // Parse fractional part
        let fractional_part = if let Some(frac_str) = fractional_str {
            if frac_str.is_empty() {
                // Just a trailing decimal point like "123."
                0i64
            } else if frac_str.len() > Self::DECIMALS as usize {
                return Err(DecimalError::PrecisionLoss);
            } else {
                // Parse the fractional digits and scale appropriately
                // e.g., "45" with 8 decimals -> 45000000
                let frac_value = frac_str
                    .parse::<i64>()
                    .map_err(|_| DecimalError::InvalidFormat)?;

                // Calculate how many zeros to append
                let digits_provided = frac_str.len() as u8;
                let scale_multiplier = const_pow10(Self::DECIMALS - digits_provided);

                frac_value
                    .checked_mul(scale_multiplier)
                    .ok_or(DecimalError::Overflow)?
            }
        } else {
            0i64
        };

        // Combine integer and fractional parts
        // Need to handle negative numbers carefully
        let abs_int_scaled = integer_part
            .abs()
            .checked_mul(Self::SCALE)
            .ok_or(DecimalError::Overflow)?;

        let abs_value = abs_int_scaled
            .checked_add(fractional_part)
            .ok_or(DecimalError::Overflow)?;

        let value = if is_negative { -abs_value } else { abs_value };

        Ok(Self { value })
    }

    /// Parse assuming a fixed number of decimals (no decimal point in string)
    /// E.g., parse_fixed_point_str("12345", 2) → 123.45
    pub fn from_fixed_point_str(s: &str, decimals: u8) -> crate::Result<Self> {
        let value = s.parse::<i64>().map_err(|_| DecimalError::InvalidFormat)?;

        if decimals > Self::DECIMALS {
            return Err(DecimalError::PrecisionLoss);
        }

        let scale_diff = Self::DECIMALS - decimals;
        let multiplier = const_pow10(scale_diff);

        let scaled = value
            .checked_mul(multiplier)
            .ok_or(DecimalError::Overflow)?;

        Ok(Self { value: scaled })
    }
}

impl FromStr for D64 {
    type Err = DecimalError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str_exact(s)
    }
}

// ============================================================================
// Bytes Operations
// ============================================================================
impl D64 {
    /// The size of this type in bytes.
    pub const BYTES: usize = core::mem::size_of::<i64>();

    /// Parse from byte slice (useful for binary protocols)
    pub fn from_utf8_bytes(bytes: &[u8]) -> crate::Result<Self> {
        let s = core::str::from_utf8(bytes).map_err(|_| DecimalError::InvalidFormat)?;
        Self::from_str_exact(s)
    }

    /// Creates a D64 from its representation as a byte array in big endian.
    #[inline]
    pub const fn from_be_bytes(bytes: [u8; 8]) -> Self {
        Self {
            value: i64::from_be_bytes(bytes),
        }
    }

    /// Creates a D64 from its representation as a byte array in little endian.
    #[inline]
    pub const fn from_le_bytes(bytes: [u8; 8]) -> Self {
        Self {
            value: i64::from_le_bytes(bytes),
        }
    }

    /// Creates a D64 from its representation as a byte array in native endian.
    #[inline]
    pub const fn from_ne_bytes(bytes: [u8; 8]) -> Self {
        Self {
            value: i64::from_ne_bytes(bytes),
        }
    }

    /// Returns the memory representation of this decimal as a byte array in big-endian byte order.
    #[inline]
    pub const fn to_be_bytes(self) -> [u8; 8] {
        self.value.to_be_bytes()
    }

    /// Returns the memory representation of this decimal as a byte array in little-endian byte order.
    #[inline]
    pub const fn to_le_bytes(self) -> [u8; 8] {
        self.value.to_le_bytes()
    }

    /// Returns the memory representation of this decimal as a byte array in native byte order.
    #[inline]
    pub const fn to_ne_bytes(self) -> [u8; 8] {
        self.value.to_ne_bytes()
    }

    /// Writes the decimal as bytes in little-endian order to the given buffer.
    ///
    /// # Panics
    /// Panics if `buf.len() < 8`.
    #[inline]
    pub fn write_le_bytes(&self, buf: &mut [u8]) {
        buf[..8].copy_from_slice(&self.to_le_bytes());
    }

    /// Writes the decimal as bytes in big-endian order to the given buffer.
    ///
    /// # Panics
    /// Panics if `buf.len() < 8`.
    #[inline]
    pub fn write_be_bytes(&self, buf: &mut [u8]) {
        buf[..8].copy_from_slice(&self.to_be_bytes());
    }

    /// Writes the decimal as bytes in native-endian order to the given buffer.
    ///
    /// # Panics
    /// Panics if `buf.len() < 8`.
    #[inline]
    pub fn write_ne_bytes(&self, buf: &mut [u8]) {
        buf[..8].copy_from_slice(&self.to_ne_bytes());
    }

    /// Reads a decimal from bytes in little-endian order from the given buffer.
    ///
    /// # Panics
    /// Panics if `buf.len() < 8`.
    #[inline]
    pub fn read_le_bytes(buf: &[u8]) -> Self {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&buf[..8]);
        Self::from_le_bytes(bytes)
    }

    /// Reads a decimal from bytes in big-endian order from the given buffer.
    ///
    /// # Panics
    /// Panics if `buf.len() < 8`.
    #[inline]
    pub fn read_be_bytes(buf: &[u8]) -> Self {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&buf[..8]);
        Self::from_be_bytes(bytes)
    }

    /// Reads a decimal from bytes in native-endian order from the given buffer.
    ///
    /// # Panics
    /// Panics if `buf.len() < 8`.
    #[inline]
    pub fn read_ne_bytes(buf: &[u8]) -> Self {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&buf[..8]);
        Self::from_ne_bytes(bytes)
    }

    /// Tries to write the decimal as bytes in little-endian order to the given buffer.
    ///
    /// Returns `None` if `buf.len() < 8`.
    #[inline]
    pub fn try_write_le_bytes(&self, buf: &mut [u8]) -> Option<()> {
        if buf.len() < 8 {
            return None;
        }
        buf[..8].copy_from_slice(&self.to_le_bytes());
        Some(())
    }

    /// Tries to write the decimal as bytes in big-endian order to the given buffer.
    ///
    /// Returns `None` if `buf.len() < 8`.
    #[inline]
    pub fn try_write_be_bytes(&self, buf: &mut [u8]) -> Option<()> {
        if buf.len() < 8 {
            return None;
        }
        buf[..8].copy_from_slice(&self.to_be_bytes());
        Some(())
    }

    /// Tries to write the decimal as bytes in native-endian order to the given buffer.
    ///
    /// Returns `None` if `buf.len() < 8`.
    #[inline]
    pub fn try_write_ne_bytes(&self, buf: &mut [u8]) -> Option<()> {
        if buf.len() < 8 {
            return None;
        }
        buf[..8].copy_from_slice(&self.to_ne_bytes());
        Some(())
    }

    /// Tries to read a decimal from bytes in little-endian order from the given buffer.
    ///
    /// Returns `None` if `buf.len() < 8`.
    #[inline]
    pub fn try_read_le_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < 8 {
            return None;
        }
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&buf[..8]);
        Some(Self::from_le_bytes(bytes))
    }

    /// Tries to read a decimal from bytes in big-endian order from the given buffer.
    ///
    /// Returns `None` if `buf.len() < 8`.
    #[inline]
    pub fn try_read_be_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < 8 {
            return None;
        }
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&buf[..8]);
        Some(Self::from_be_bytes(bytes))
    }

    /// Tries to read a decimal from bytes in native-endian order from the given buffer.
    ///
    /// Returns `None` if `buf.len() < 8`.
    #[inline]
    pub fn try_read_ne_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < 8 {
            return None;
        }
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&buf[..8]);
        Some(Self::from_ne_bytes(bytes))
    }
}

// ============================================================================
// Operator Overloading
// ============================================================================

impl Add for D64 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        self.checked_add(rhs).expect("attempt to add with overflow")
    }
}

impl Sub for D64 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        self.checked_sub(rhs)
            .expect("attempt to subtract with overflow")
    }
}

impl Mul for D64 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self.checked_mul(rhs)
            .expect("attempt to multiply with overflow")
    }
}

impl Div for D64 {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        self.checked_div(rhs)
            .expect("attempt to divide by zero or overflow")
    }
}

impl Neg for D64 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        self.checked_neg().expect("attempt to negate with overflow")
    }
}

impl AddAssign for D64 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for D64 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for D64 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl DivAssign for D64 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

// ============================================================================
// Standard Library Trait Implementations
// ============================================================================

impl TryFrom<i64> for D64 {
    type Error = DecimalError;

    #[inline(always)]
    fn try_from(value: i64) -> crate::Result<Self> {
        Self::try_from_i64(value)
    }
}

impl TryFrom<u64> for D64 {
    type Error = DecimalError;

    #[inline(always)]
    fn try_from(value: u64) -> crate::Result<Self> {
        Self::try_from_u64(value)
    }
}

impl TryFrom<f64> for D64 {
    type Error = DecimalError;

    #[inline(always)]
    fn try_from(value: f64) -> crate::Result<Self> {
        Self::try_from_f64(value)
    }
}

impl TryFrom<f32> for D64 {
    type Error = DecimalError;

    #[inline(always)]
    fn try_from(value: f32) -> crate::Result<Self> {
        Self::try_from_f64(value as f64)
    }
}

impl From<i32> for D64 {
    #[inline(always)]
    fn from(value: i32) -> Self {
        Self::from_i32(value)
    }
}

impl From<u32> for D64 {
    #[inline(always)]
    fn from(value: u32) -> Self {
        Self::from_u32(value)
    }
}

impl From<i16> for D64 {
    #[inline(always)]
    fn from(value: i16) -> Self {
        Self::from_i32(value as i32)
    }
}

impl From<u16> for D64 {
    #[inline(always)]
    fn from(value: u16) -> Self {
        Self::from_u32(value as u32)
    }
}

impl From<i8> for D64 {
    #[inline(always)]
    fn from(value: i8) -> Self {
        Self::from_i32(value as i32)
    }
}

impl From<u8> for D64 {
    #[inline(always)]
    fn from(value: u8) -> Self {
        Self::from_u32(value as u32)
    }
}

impl fmt::Display for D64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let is_negative = self.value < 0;
        let abs_value = self.value.unsigned_abs();

        let integer_part = abs_value / Self::SCALE as u64;
        let fractional_part = abs_value % Self::SCALE as u64;

        if is_negative {
            f.write_str("-")?;
        }

        write!(f, "{}", integer_part)?;

        // Handle precision specifier
        if let Some(precision) = f.precision() {
            if precision > 0 && fractional_part > 0 {
                // Only show decimal if non-zero
                f.write_str(".")?;

                let precision = precision.min(Self::DECIMALS as usize);
                let scale_reduction = Self::DECIMALS as usize - precision;
                let divisor = 10u64.pow(scale_reduction as u32);
                let rounded_frac = (fractional_part + divisor / 2) / divisor;

                write!(f, "{:0width$}", rounded_frac, width = precision)?;
            }
        } else {
            // Default: strip trailing zeros
            if fractional_part > 0 {
                f.write_str(".")?;

                // Write fractional part, then strip trailing zeros
                let mut buf = [0u8; 8];
                format_u64_padded(fractional_part, &mut buf);

                // Find last non-zero digit
                let mut end = 8;
                while end > 0 && buf[end - 1] == b'0' {
                    end -= 1;
                }

                // Safety: buf contains only ASCII digits
                let s = unsafe { core::str::from_utf8_unchecked(&buf[..end]) };
                f.write_str(s)?;
            }
        }

        Ok(())
    }
}

impl fmt::Debug for D64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            // {:#?} shows raw internals
            f.debug_struct("D64").field("value", &self.value).finish()
        } else {
            // {:?} shows formatted decimal
            write!(f, "D64({})", self)
        }
    }
}

// ============================================================================
// Iterator Trait Implementations
// ============================================================================

impl Sum for D64 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<'a> Sum<&'a D64> for D64 {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + *x)
    }
}

impl Product for D64 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl<'a> Product<&'a D64> for D64 {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * *x)
    }
}

// ============================================================================
// Serde Support
// ============================================================================

#[cfg(feature = "serde")]
impl Serialize for D64 {
    fn serialize<S>(&self, serializer: S) -> core::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Serialize as string to preserve precision
        let s = alloc::format!("{}", self);
        serializer.serialize_str(&s)
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for D64 {
    fn deserialize<D>(deserializer: D) -> core::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::from_str(&s).map_err(de::Error::custom)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute 10^n at compile time for rounding operations
const fn const_pow10(n: u8) -> i64 {
    match n {
        0 => 1,
        1 => 10,
        2 => 100,
        3 => 1_000,
        4 => 10_000,
        5 => 100_000,
        6 => 1_000_000,
        7 => 10_000_000,
        8 => 100_000_000,
        _ => panic!("pow10 out of range"),
    }
}

/// Integer square root using binary search
const fn isqrt(n: u64) -> u64 {
    if n < 2 {
        return n;
    }

    let mut left = 1u64;
    let mut right = n;

    while left <= right {
        let mid = left + (right - left) / 2;

        // Check if mid * mid == n (avoiding overflow)
        if mid <= n / mid {
            // mid^2 <= n
            let next_mid = mid + 1;
            if next_mid > n / next_mid {
                // (mid+1)^2 > n, so mid is the answer
                return mid;
            }
            left = mid + 1;
        } else {
            // mid^2 > n
            right = mid - 1;
        }
    }

    right
}

/// Format a u64 with leading zeros into a fixed-size buffer
#[inline(always)]
fn format_u64_padded(mut n: u64, buf: &mut [u8; 8]) {
    for i in (0..8).rev() {
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_d64_constants() {
        assert_eq!(D64::ZERO.to_raw(), 0);
        assert_eq!(D64::ONE.to_raw(), 100_000_000);
        assert_eq!(D64::SCALE, 100_000_000);
    }

    #[test]
    fn test_addition() {
        let a = D64::from_raw(100_000_000); // 1.0
        let b = D64::from_raw(200_000_000); // 2.0

        assert_eq!(a.checked_add(b), Some(D64::from_raw(300_000_000))); // 3.0
        assert_eq!(a.saturating_add(b), D64::from_raw(300_000_000));
        assert_eq!(a.wrapping_add(b), D64::from_raw(300_000_000));
    }

    #[test]
    fn test_addition_overflow() {
        let a = D64::MAX;
        let b = D64::ONE;

        assert_eq!(a.checked_add(b), None);
        assert_eq!(a.saturating_add(b), D64::MAX);
    }

    #[test]
    fn test_subtraction() {
        let a = D64::from_raw(300_000_000); // 3.0
        let b = D64::from_raw(100_000_000); // 1.0

        assert_eq!(a.checked_sub(b), Some(D64::from_raw(200_000_000))); // 2.0
    }

    #[test]
    fn test_multiplication() {
        let a = D64::from_raw(200_000_000); // 2.0
        let b = D64::from_raw(300_000_000); // 3.0

        assert_eq!(a.checked_mul(b), Some(D64::from_raw(600_000_000))); // 6.0
    }

    #[test]
    fn test_division() {
        let a = D64::from_raw(600_000_000); // 6.0
        let b = D64::from_raw(200_000_000); // 2.0

        assert_eq!(a.checked_div(b), Some(D64::from_raw(300_000_000))); // 3.0
    }

    #[test]
    fn test_division_by_zero() {
        let a = D64::ONE;
        let b = D64::ZERO;

        assert_eq!(a.checked_div(b), None);
        assert_eq!(a.saturating_div(b), D64::ZERO);
    }

    #[test]
    fn test_negation() {
        let a = D64::from_raw(100_000_000); // 1.0

        assert_eq!(a.checked_neg(), Some(D64::from_raw(-100_000_000))); // -1.0
    }

    #[test]
    fn test_abs() {
        let a = D64::from_raw(-100_000_000); // -1.0

        assert_eq!(a.abs(), D64::from_raw(100_000_000)); // 1.0
    }

    #[test]
    fn test_sign_checks() {
        assert!(D64::ONE.is_positive());
        assert!(!D64::ONE.is_negative());
        assert!(!D64::ONE.is_zero());

        assert!(D64::ZERO.is_zero());
        assert!(!D64::ZERO.is_positive());
        assert!(!D64::ZERO.is_negative());

        let neg = D64::from_raw(-100_000_000);
        assert!(neg.is_negative());
        assert!(!neg.is_positive());
    }

    #[test]
    fn test_signum() {
        assert_eq!(D64::ONE.signum(), 1);
        assert_eq!(D64::ZERO.signum(), 0);
        assert_eq!(D64::from_raw(-100_000_000).signum(), -1);
    }
}

#[cfg(test)]
mod operator_tests {
    use super::*;

    #[test]
    fn test_add_operator() {
        let a = D64::from_raw(100_000_000); // 1.0
        let b = D64::from_raw(200_000_000); // 2.0
        let c = a + b;
        assert_eq!(c.to_raw(), 300_000_000); // 3.0
    }

    #[test]
    #[should_panic(expected = "attempt to add with overflow")]
    fn test_add_operator_panic() {
        let _ = D64::MAX + D64::ONE;
    }

    #[test]
    fn test_sub_operator() {
        let a = D64::from_raw(300_000_000); // 3.0
        let b = D64::from_raw(100_000_000); // 1.0
        let c = a - b;
        assert_eq!(c.to_raw(), 200_000_000); // 2.0
    }

    #[test]
    fn test_mul_operator() {
        let a = D64::from_raw(200_000_000); // 2.0
        let b = D64::from_raw(300_000_000); // 3.0
        let c = a * b;
        assert_eq!(c.to_raw(), 600_000_000); // 6.0
    }

    #[test]
    fn test_div_operator() {
        let a = D64::from_raw(600_000_000); // 6.0
        let b = D64::from_raw(200_000_000); // 2.0
        let c = a / b;
        assert_eq!(c.to_raw(), 300_000_000); // 3.0
    }

    #[test]
    #[should_panic(expected = "attempt to divide by zero")]
    fn test_div_by_zero_panic() {
        let _ = D64::ONE / D64::ZERO;
    }

    #[test]
    fn test_neg_operator() {
        let a = D64::from_raw(100_000_000); // 1.0
        let b = -a;
        assert_eq!(b.to_raw(), -100_000_000); // -1.0
    }

    #[test]
    fn test_add_assign() {
        let mut a = D64::from_raw(100_000_000); // 1.0
        a += D64::from_raw(200_000_000); // 2.0
        assert_eq!(a.to_raw(), 300_000_000); // 3.0
    }

    #[test]
    fn test_sub_assign() {
        let mut a = D64::from_raw(300_000_000); // 3.0
        a -= D64::from_raw(100_000_000); // 1.0
        assert_eq!(a.to_raw(), 200_000_000); // 2.0
    }

    #[test]
    fn test_mul_assign() {
        let mut a = D64::from_raw(200_000_000); // 2.0
        a *= D64::from_raw(300_000_000); // 3.0
        assert_eq!(a.to_raw(), 600_000_000); // 6.0
    }

    #[test]
    fn test_div_assign() {
        let mut a = D64::from_raw(600_000_000); // 6.0
        a /= D64::from_raw(200_000_000); // 2.0
        assert_eq!(a.to_raw(), 300_000_000); // 3.0
    }

    #[test]
    fn test_operator_chaining() {
        let a = D64::from_raw(100_000_000); // 1.0
        let b = D64::from_raw(200_000_000); // 2.0
        let c = D64::from_raw(300_000_000); // 3.0

        let result = a + b * c; // 1.0 + (2.0 * 3.0) = 7.0
        assert_eq!(result.to_raw(), 700_000_000);
    }
}

#[cfg(test)]
mod conversion_tests {
    use super::*;

    #[test]
    fn test_from_i32() {
        let d = D64::from_i32(42);
        assert_eq!(d.to_i64(), 42);

        let d = D64::from(42i32);
        assert_eq!(d.to_i64(), 42);
    }

    #[test]
    fn test_from_i64() {
        assert_eq!(D64::from_i64(100).unwrap().to_i64(), 100);
        assert!(D64::from_i64(i64::MAX).is_none()); // Would overflow when scaled
    }

    #[test]
    fn test_to_i64_truncate() {
        let d = D64::from_raw(250_000_000); // 2.5
        assert_eq!(d.to_i64(), 2); // Truncates
    }

    #[test]
    fn test_to_i64_round() {
        let d1 = D64::from_raw(250_000_000); // 2.5
        assert_eq!(d1.to_i64_round(), 2); // Banker's rounding: round to even

        let d2 = D64::from_raw(350_000_000); // 3.5
        assert_eq!(d2.to_i64_round(), 4); // Banker's rounding: round to even

        let d3 = D64::from_raw(260_000_000); // 2.6
        assert_eq!(d3.to_i64_round(), 3); // Normal rounding
    }

    #[test]
    fn test_from_f64() {
        let d = D64::from_f64(3.14159265).unwrap();
        let f = d.to_f64();
        assert!((f - 3.14159265).abs() < 1e-8);
    }

    #[test]
    fn test_from_f64_edge_cases() {
        assert!(D64::from_f64(f64::NAN).is_none());
        assert!(D64::from_f64(f64::INFINITY).is_none());
        assert!(D64::from_f64(f64::NEG_INFINITY).is_none());
    }

    #[test]
    fn test_try_from() {
        assert!(D64::try_from(42i64).is_ok());
        assert!(D64::try_from(i64::MAX).is_err());
        assert!(D64::try_from(3.14f64).is_ok());
        assert!(D64::try_from(f64::NAN).is_err());
    }

    #[test]
    fn test_small_int_conversions() {
        let d1: D64 = 42i8.into();
        let d2: D64 = 42u8.into();
        let d3: D64 = 42i16.into();
        let d4: D64 = 42u16.into();
        let d5: D64 = 42i32.into();
        let d6: D64 = 42u32.into();

        assert_eq!(d1.to_i64(), 42);
        assert_eq!(d2.to_i64(), 42);
        assert_eq!(d3.to_i64(), 42);
        assert_eq!(d4.to_i64(), 42);
        assert_eq!(d5.to_i64(), 42);
        assert_eq!(d6.to_i64(), 42);
    }
}

#[cfg(test)]
mod comparison_tests {
    use super::*;

    #[test]
    fn test_min() {
        let a = D64::from_raw(100_000_000); // 1.0
        let b = D64::from_raw(200_000_000); // 2.0

        assert_eq!(a.min(b), a);
        assert_eq!(b.min(a), a);
    }

    #[test]
    fn test_max() {
        let a = D64::from_raw(100_000_000); // 1.0
        let b = D64::from_raw(200_000_000); // 2.0

        assert_eq!(a.max(b), b);
        assert_eq!(b.max(a), b);
    }

    #[test]
    fn test_clamp() {
        let min = D64::from_raw(100_000_000); // 1.0
        let max = D64::from_raw(300_000_000); // 3.0

        let below = D64::from_raw(50_000_000); // 0.5
        let within = D64::from_raw(200_000_000); // 2.0
        let above = D64::from_raw(400_000_000); // 4.0

        assert_eq!(below.clamp(min, max), min);
        assert_eq!(within.clamp(min, max), within);
        assert_eq!(above.clamp(min, max), max);
    }

    #[test]
    #[should_panic(expected = "min must be less than or equal to max")]
    fn test_clamp_panic() {
        let min = D64::from_raw(300_000_000);
        let max = D64::from_raw(100_000_000);
        let _ = D64::ZERO.clamp(min, max);
    }
}

#[cfg(test)]
mod rounding_tests {
    use super::*;

    #[test]
    fn test_floor() {
        assert_eq!(D64::from_raw(250_000_000).floor().to_raw(), 200_000_000); // 2.5 -> 2.0
        assert_eq!(D64::from_raw(-250_000_000).floor().to_raw(), -300_000_000); // -2.5 -> -3.0
        assert_eq!(D64::from_raw(200_000_000).floor().to_raw(), 200_000_000); // 2.0 -> 2.0
    }

    #[test]
    fn test_ceil() {
        assert_eq!(D64::from_raw(250_000_000).ceil().to_raw(), 300_000_000); // 2.5 -> 3.0
        assert_eq!(D64::from_raw(-250_000_000).ceil().to_raw(), -200_000_000); // -2.5 -> -2.0
        assert_eq!(D64::from_raw(200_000_000).ceil().to_raw(), 200_000_000); // 2.0 -> 2.0
    }

    #[test]
    fn test_trunc() {
        assert_eq!(D64::from_raw(250_000_000).trunc().to_raw(), 200_000_000); // 2.5 -> 2.0
        assert_eq!(D64::from_raw(-250_000_000).trunc().to_raw(), -200_000_000); // -2.5 -> -2.0
    }

    #[test]
    fn test_fract() {
        assert_eq!(D64::from_raw(250_000_000).fract().to_raw(), 50_000_000); // 2.5 -> 0.5
        assert_eq!(D64::from_raw(200_000_000).fract().to_raw(), 0); // 2.0 -> 0.0
    }

    #[test]
    fn test_round() {
        assert_eq!(D64::from_raw(250_000_000).round().to_raw(), 200_000_000); // 2.5 -> 2.0 (banker's)
        assert_eq!(D64::from_raw(350_000_000).round().to_raw(), 400_000_000); // 3.5 -> 4.0 (banker's)
        assert_eq!(D64::from_raw(260_000_000).round().to_raw(), 300_000_000); // 2.6 -> 3.0
    }

    #[test]
    fn test_round_dp() {
        let val = D64::from_raw(123_456_789); // 1.23456789

        assert_eq!(val.round_dp(0).to_raw(), 100_000_000); // 1.0
        assert_eq!(val.round_dp(2).to_raw(), 123_000_000); // 1.23
        assert_eq!(val.round_dp(4).to_raw(), 123_460_000); // 1.2346
        assert_eq!(val.round_dp(8).to_raw(), 123_456_789); // 1.23456789 (unchanged)
    }

    #[test]
    #[should_panic(expected = "decimal_places must be <= DECIMALS")]
    fn test_round_dp_panic() {
        core::hint::black_box(D64::ZERO.round_dp(9));
    }
}

#[cfg(test)]
mod math_tests {
    use super::*;

    #[test]
    fn test_recip() {
        let two = D64::from_raw(200_000_000); // 2.0
        let half = two.recip().unwrap();
        assert_eq!(half.to_raw(), 50_000_000); // 0.5

        assert_eq!(D64::ZERO.recip(), None);
    }

    #[test]
    fn test_powi_positive() {
        let two = D64::from_raw(200_000_000); // 2.0

        assert_eq!(two.powi(0).unwrap().to_raw(), 100_000_000); // 2^0 = 1.0
        assert_eq!(two.powi(1).unwrap().to_raw(), 200_000_000); // 2^1 = 2.0
        assert_eq!(two.powi(2).unwrap().to_raw(), 400_000_000); // 2^2 = 4.0
        assert_eq!(two.powi(3).unwrap().to_raw(), 800_000_000); // 2^3 = 8.0
    }

    #[test]
    fn test_powi_negative() {
        let two = D64::from_raw(200_000_000); // 2.0

        assert_eq!(two.powi(-1).unwrap().to_raw(), 50_000_000); // 2^-1 = 0.5
        assert_eq!(two.powi(-2).unwrap().to_raw(), 25_000_000); // 2^-2 = 0.25
    }

    #[test]
    fn test_powi_compound_interest() {
        // 1.05^10 for compound interest calculation
        let rate = D64::from_raw(105_000_000); // 1.05 (5% interest)
        let result = rate.powi(10).unwrap();

        // 1.05^10 ≈ 1.62889463
        let expected = D64::from_raw(162_889_463);

        // Allow small rounding difference
        assert!((result.to_raw() - expected.to_raw()).abs() < 100);
    }

    #[test]
    fn test_sqrt_perfect_squares() {
        let four = D64::from_raw(400_000_000); // 4.0
        let sqrt_four = four.sqrt().unwrap();
        assert_eq!(sqrt_four.to_raw(), 200_000_000); // 2.0

        let nine = D64::from_raw(900_000_000); // 9.0
        let sqrt_nine = nine.sqrt().unwrap();
        assert_eq!(sqrt_nine.to_raw(), 300_000_000); // 3.0
    }

    #[test]
    fn test_sqrt_non_perfect() {
        let two = D64::from_raw(200_000_000); // 2.0
        let sqrt_two = two.sqrt().unwrap();

        // sqrt(2) ≈ 1.41421356
        let expected = D64::from_raw(141_421_356);

        // Check accuracy within reasonable tolerance
        assert!((sqrt_two.to_raw() - expected.to_raw()).abs() < 10);
    }

    #[test]
    fn test_sqrt_edge_cases() {
        assert_eq!(D64::ZERO.sqrt().unwrap(), D64::ZERO);
        assert_eq!(D64::ONE.sqrt().unwrap(), D64::ONE);

        let neg = D64::from_raw(-100_000_000);
        assert_eq!(neg.sqrt(), None);
    }

    #[test]
    fn test_sqrt_verify() {
        // Test that sqrt(x)^2 ≈ x
        let x = D64::from_raw(500_000_000); // 5.0
        let sqrt_x = x.sqrt().unwrap();
        let squared = sqrt_x.checked_mul(sqrt_x).unwrap();

        // Should be very close to original
        assert!((squared.to_raw() - x.to_raw()).abs() < 100);
    }
}

#[cfg(test)]
mod result_tests {
    use super::*;

    #[test]
    fn test_try_add() {
        let a = D64::from_raw(100_000_000);
        let b = D64::from_raw(200_000_000);

        assert!(a.try_add(b).is_ok());
        assert_eq!(a.try_add(b).unwrap().to_raw(), 300_000_000);

        assert!(D64::MAX.try_add(D64::ONE).is_err());
    }

    #[test]
    fn test_try_sub() {
        let a = D64::from_raw(300_000_000);
        let b = D64::from_raw(100_000_000);

        assert!(a.try_sub(b).is_ok());
        assert!(D64::MIN.try_sub(D64::ONE).is_err());
    }

    #[test]
    fn test_try_mul() {
        let a = D64::from_raw(200_000_000);
        let b = D64::from_raw(300_000_000);

        assert!(a.try_mul(b).is_ok());
        assert!(D64::MAX.try_mul(D64::from_raw(200_000_000)).is_err());
    }

    #[test]
    fn test_try_div() {
        let a = D64::from_raw(600_000_000);
        let b = D64::from_raw(200_000_000);

        assert!(a.try_div(b).is_ok());

        let result = D64::ONE.try_div(D64::ZERO);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), DecimalError::DivisionByZero));
    }

    #[test]
    fn test_try_neg() {
        assert!(D64::ONE.try_neg().is_ok());
        assert!(D64::MIN.try_neg().is_err());
    }

    #[test]
    fn test_try_abs() {
        assert!(D64::from_raw(-100_000_000).try_abs().is_ok());
        assert!(D64::MIN.try_abs().is_err());
    }

    #[test]
    fn test_try_recip() {
        assert!(D64::from_raw(200_000_000).try_recip().is_ok());

        let result = D64::ZERO.try_recip();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), DecimalError::DivisionByZero));
    }

    #[test]
    fn test_try_powi() {
        let two = D64::from_raw(200_000_000);
        assert!(two.try_powi(3).is_ok());
        assert!(D64::MAX.try_powi(2).is_err());
    }

    #[test]
    fn test_try_sqrt() {
        let four = D64::from_raw(400_000_000);
        assert!(four.try_sqrt().is_ok());

        let neg = D64::from_raw(-100_000_000);
        assert!(neg.try_sqrt().is_err());
    }
}

#[cfg(test)]
mod string_tests {
    use super::*;

    #[test]
    fn test_from_str_integer() {
        assert_eq!(D64::from_str_exact("123").unwrap().to_raw(), 12_300_000_000);
        assert_eq!(D64::from_str_exact("0").unwrap().to_raw(), 0);
        assert_eq!(
            D64::from_str_exact("-456").unwrap().to_raw(),
            -45_600_000_000
        );
    }

    #[test]
    fn test_from_str_decimal() {
        assert_eq!(
            D64::from_str_exact("123.45").unwrap().to_raw(),
            12_345_000_000
        );
        assert_eq!(D64::from_str_exact("0.00000001").unwrap().to_raw(), 1);
        assert_eq!(
            D64::from_str_exact("-123.45").unwrap().to_raw(),
            -12_345_000_000
        );
    }

    #[test]
    fn test_from_str_leading_decimal() {
        assert_eq!(D64::from_str_exact("0.5").unwrap().to_raw(), 50_000_000);
        assert_eq!(D64::from_str_exact(".5").unwrap().to_raw(), 50_000_000);
    }

    #[test]
    fn test_from_str_trailing_zeros() {
        assert_eq!(
            D64::from_str_exact("123.45000000").unwrap().to_raw(),
            12_345_000_000
        );
        assert_eq!(D64::from_str_exact("1.10").unwrap().to_raw(), 110_000_000);
    }

    #[test]
    fn test_from_str_plus_sign() {
        assert_eq!(
            D64::from_str_exact("+123.45").unwrap().to_raw(),
            12_345_000_000
        );
    }

    #[test]
    fn test_from_str_whitespace() {
        assert_eq!(
            D64::from_str_exact("  123.45  ").unwrap().to_raw(),
            12_345_000_000
        );
    }

    #[test]
    fn test_from_str_errors() {
        assert!(D64::from_str_exact("").is_err());
        assert!(D64::from_str_exact("abc").is_err());
        assert!(D64::from_str_exact("12.34.56").is_err());
        assert!(matches!(
            D64::from_str_exact("123.123456789"),
            Err(DecimalError::PrecisionLoss)
        ));
    }

    #[test]
    fn test_from_str_overflow() {
        assert!(matches!(
            D64::from_str_exact("999999999999999"),
            Err(DecimalError::Overflow)
        ));
    }

    #[test]
    fn test_from_str_trait() {
        let d: D64 = "123.45".parse().unwrap();
        assert_eq!(d.to_raw(), 12_345_000_000);
    }

    #[test]
    fn test_from_str_edge_cases() {
        // Just decimal point
        assert!(D64::from_str_exact(".").is_err());

        // Multiple signs
        assert!(D64::from_str_exact("--123").is_err());

        // Sign after number
        assert!(D64::from_str_exact("123-").is_err());
    }
}

#[cfg(test)]
mod const_new_tests {
    use super::*;

    #[test]
    fn test_new_basic() {
        let d = D64::new(123, 45_000_000);
        assert_eq!(d, D64::from_str_exact("123.45").unwrap());
    }

    #[test]
    fn test_new_zero() {
        let d = D64::new(0, 0);
        assert_eq!(d, D64::ZERO);
    }

    #[test]
    fn test_new_integer_only() {
        let d = D64::new(42, 0);
        assert_eq!(d, D64::from_str_exact("42").unwrap());
    }

    #[test]
    fn test_new_fractional_only() {
        let d = D64::new(0, 50_000_000);
        assert_eq!(d, D64::from_str_exact("0.5").unwrap());
    }

    #[test]
    fn test_new_negative_integer() {
        let d = D64::new(-123, 45_000_000);
        assert_eq!(d, D64::from_str_exact("-123.45").unwrap());
    }

    #[test]
    fn test_new_max_fractional() {
        let d = D64::new(1, 99_999_999);
        assert_eq!(d, D64::from_str_exact("1.99999999").unwrap());
    }

    #[test]
    fn test_new_const() {
        const RATE: D64 = D64::new(2, 50_000_000); // 2.5%
        assert_eq!(RATE, D64::from_str_exact("2.5").unwrap());
    }

    #[test]
    fn test_new_large_values() {
        let d = D64::new(1_000_000, 12_345_678);
        assert_eq!(d, D64::from_str_exact("1000000.12345678").unwrap());
    }
}

#[cfg(test)]
mod mul_i64_tests {
    use super::*;

    #[test]
    fn test_mul_i64_basic() {
        let price = D64::from_str_exact("10.50").unwrap();
        let quantity = 5;

        let total = price.mul_i64(quantity).unwrap();
        assert_eq!(total, D64::from_str_exact("52.50").unwrap());
    }

    #[test]
    fn test_mul_i64_zero() {
        let price = D64::from_str_exact("100.00").unwrap();
        let total = price.mul_i64(0).unwrap();
        assert_eq!(total, D64::ZERO);
    }

    #[test]
    fn test_mul_i64_one() {
        let price = D64::from_str_exact("42.42").unwrap();
        let total = price.mul_i64(1).unwrap();
        assert_eq!(total, price);
    }

    #[test]
    fn test_mul_i64_negative_quantity() {
        let price = D64::from_str_exact("10.00").unwrap();
        let total = price.mul_i64(-5).unwrap();
        assert_eq!(total, D64::from_str_exact("-50.00").unwrap());
    }

    #[test]
    fn test_mul_i64_negative_price() {
        let price = D64::from_str_exact("-25.50").unwrap();
        let total = price.mul_i64(4).unwrap();
        assert_eq!(total, D64::from_str_exact("-102.00").unwrap());
    }

    #[test]
    fn test_mul_i64_both_negative() {
        let price = D64::from_str_exact("-10.00").unwrap();
        let total = price.mul_i64(-3).unwrap();
        assert_eq!(total, D64::from_str_exact("30.00").unwrap());
    }

    #[test]
    fn test_mul_i64_overflow() {
        let price = D64::MAX;
        let result = price.mul_i64(2);
        assert!(result.is_none());
    }

    #[test]
    fn test_mul_i64_large_quantity() {
        let price = D64::from_str_exact("0.01").unwrap(); // 1 cent
        let quantity = 1_000_000; // 1 million

        let total = price.mul_i64(quantity).unwrap();
        assert_eq!(total, D64::from_str_exact("10000.00").unwrap());
    }

    #[test]
    fn test_try_mul_i64_success() {
        let price = D64::from_str_exact("5.25").unwrap();
        let total = price.try_mul_i64(10).unwrap();
        assert_eq!(total, D64::from_str_exact("52.50").unwrap());
    }

    #[test]
    fn test_try_mul_i64_overflow_error() {
        let price = D64::MAX;
        let result = price.try_mul_i64(2);
        assert!(matches!(result, Err(DecimalError::Overflow)));
    }

    #[test]
    fn test_mul_i64_fractional_result() {
        let price = D64::from_str_exact("3.333333").unwrap();
        let total = price.mul_i64(3).unwrap();
        assert_eq!(total, D64::from_str_exact("9.999999").unwrap());
    }
}

#[cfg(test)]
mod fixed_point_str_tests {
    use std::println;

    use super::*;

    #[test]
    fn test_from_fixed_point_str_2_decimals() {
        // Common for currencies: "12345" with 2 decimals → 123.45
        let d = D64::from_fixed_point_str("12345", 2).unwrap();
        assert_eq!(d, D64::from_str_exact("123.45").unwrap());
    }

    #[test]
    fn test_from_fixed_point_str_0_decimals() {
        // No decimals: "123" with 0 decimals → 123.00
        let d = D64::from_fixed_point_str("123", 0).unwrap();
        assert_eq!(d, D64::from_str_exact("123").unwrap());
    }

    #[test]
    fn test_from_fixed_point_str_8_decimals() {
        // All decimals: "12345678" with 8 decimals → 0.12345678
        let d = D64::from_fixed_point_str("12345678", 8).unwrap();
        assert_eq!(d, D64::from_str_exact("0.12345678").unwrap());
    }

    #[test]
    fn test_from_fixed_point_str_4_decimals() {
        // Mid-range: "1234567" with 4 decimals → 123.4567
        let d = D64::from_fixed_point_str("1234567", 4).unwrap();
        assert_eq!(d, D64::from_str_exact("123.4567").unwrap());
    }

    #[test]
    fn test_from_fixed_point_str_negative() {
        let d = D64::from_fixed_point_str("-12345", 2).unwrap();
        assert_eq!(d, D64::from_str_exact("-123.45").unwrap());
    }

    #[test]
    fn test_from_fixed_point_str_zero() {
        let d = D64::from_fixed_point_str("0", 2).unwrap();
        assert_eq!(d, D64::ZERO);
    }

    #[test]
    fn test_from_fixed_point_str_leading_zeros() {
        // "00123" with 2 decimals → 1.23
        let d = D64::from_fixed_point_str("00123", 2).unwrap();
        assert_eq!(d, D64::from_str_exact("1.23").unwrap());
    }

    #[test]
    fn test_from_fixed_point_str_invalid_format() {
        let result = D64::from_fixed_point_str("not_a_number", 2);
        assert!(matches!(result, Err(DecimalError::InvalidFormat)));
    }

    #[test]
    fn test_from_fixed_point_str_overflow() {
        // This number fits in i64 (9223372036854775807 is i64::MAX)
        // but when we try to scale it up, it will overflow
        let result = D64::from_fixed_point_str("92233720368547758", 2);
        assert!(matches!(result, Err(DecimalError::Overflow)));
    }

    #[test]
    fn test_from_fixed_point_str_parse_error() {
        // Number too large to even parse into i64
        let result = D64::from_fixed_point_str("99999999999999999999", 2);
        assert!(matches!(result, Err(DecimalError::InvalidFormat)));
    }

    #[test]
    fn test_from_fixed_point_str_fix_protocol() {
        // FIX protocol often uses 5 decimals for prices
        let d = D64::from_fixed_point_str("10050000", 5).unwrap();
        assert_eq!(d, D64::from_str_exact("100.50000").unwrap());
    }
}

#[cfg(test)]
mod basis_points_tests {
    use super::*;

    #[test]
    fn test_from_basis_points_basic() {
        // 100 bps = 1%
        let d = D64::from_basis_points(100).unwrap();
        assert_eq!(d, D64::from_str_exact("0.01").unwrap());
    }

    #[test]
    fn test_from_basis_points_zero() {
        let d = D64::from_basis_points(0).unwrap();
        assert_eq!(d, D64::ZERO);
    }

    #[test]
    fn test_from_basis_points_one() {
        // 1 bp = 0.0001
        let d = D64::from_basis_points(1).unwrap();
        assert_eq!(d, D64::from_str_exact("0.0001").unwrap());
    }

    #[test]
    fn test_from_basis_points_50() {
        // 50 bps = 0.5%
        let d = D64::from_basis_points(50).unwrap();
        assert_eq!(d, D64::from_str_exact("0.005").unwrap());
    }

    #[test]
    fn test_from_basis_points_10000() {
        // 10000 bps = 100%
        let d = D64::from_basis_points(10000).unwrap();
        assert_eq!(d, D64::from_str_exact("1").unwrap());
    }

    #[test]
    fn test_from_basis_points_negative() {
        // -25 bps
        let d = D64::from_basis_points(-25).unwrap();
        assert_eq!(d, D64::from_str_exact("-0.0025").unwrap());
    }

    #[test]
    fn test_to_basis_points_basic() {
        // 1% = 100 bps
        let d = D64::from_str_exact("0.01").unwrap();
        assert_eq!(d.to_basis_points(), 100);
    }

    #[test]
    fn test_to_basis_points_zero() {
        assert_eq!(D64::ZERO.to_basis_points(), 0);
    }

    #[test]
    fn test_to_basis_points_one() {
        let d = D64::from_str_exact("0.0001").unwrap();
        assert_eq!(d.to_basis_points(), 1);
    }

    #[test]
    fn test_to_basis_points_50() {
        let d = D64::from_str_exact("0.005").unwrap();
        assert_eq!(d.to_basis_points(), 50);
    }

    #[test]
    fn test_to_basis_points_negative() {
        let d = D64::from_str_exact("-0.0025").unwrap();
        assert_eq!(d.to_basis_points(), -25);
    }

    #[test]
    fn test_to_basis_points_fractional_truncates() {
        // 0.00015 = 1.5 bps, truncates to 1
        let d = D64::from_str_exact("0.00015").unwrap();
        assert_eq!(d.to_basis_points(), 1);
    }

    #[test]
    fn test_basis_points_round_trip() {
        let original_bps = 250; // 2.5%
        let d = D64::from_basis_points(original_bps).unwrap();
        let back_to_bps = d.to_basis_points();
        assert_eq!(original_bps, back_to_bps);
    }

    #[test]
    fn test_basis_points_interest_rate() {
        // Fed funds rate move of 25 bps
        let rate_change = D64::from_basis_points(25).unwrap();
        let old_rate = D64::from_str_exact("5.25").unwrap(); // 5.25%
        let new_rate = old_rate + rate_change;
        assert_eq!(new_rate, D64::from_str_exact("5.2525").unwrap());
    }

    #[test]
    fn test_basis_points_spread() {
        // Credit spread of 150 bps over treasuries
        let spread = D64::from_basis_points(150).unwrap();
        assert_eq!(spread, D64::from_str_exact("0.015").unwrap());
    }

    #[test]
    fn test_from_basis_points_overflow() {
        // Very large number that will overflow
        let result = D64::from_basis_points(i64::MAX);
        assert!(result.is_none());
    }
}

#[cfg(test)]
mod byte_tests {
    use super::*;

    #[test]
    fn test_to_le_bytes() {
        let d = D64::from_raw(0x0123456789ABCDEF_u64 as i64);
        let bytes = d.to_le_bytes();
        assert_eq!(bytes, [0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01]);
    }

    #[test]
    fn test_from_le_bytes() {
        let bytes = [0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01];
        let d = D64::from_le_bytes(bytes);
        assert_eq!(d.to_raw(), 0x0123456789ABCDEF_u64 as i64);
    }

    #[test]
    fn test_to_be_bytes() {
        let d = D64::from_raw(0x0123456789ABCDEF_u64 as i64);
        let bytes = d.to_be_bytes();
        assert_eq!(bytes, [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF]);
    }

    #[test]
    fn test_from_be_bytes() {
        let bytes = [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF];
        let d = D64::from_be_bytes(bytes);
        assert_eq!(d.to_raw(), 0x0123456789ABCDEF_u64 as i64);
    }

    #[test]
    fn test_round_trip_le() {
        let original = D64::from_raw(123_456_789);
        let bytes = original.to_le_bytes();
        let restored = D64::from_le_bytes(bytes);
        assert_eq!(original, restored);
    }

    #[test]
    fn test_round_trip_be() {
        let original = D64::from_raw(-987_654_321);
        let bytes = original.to_be_bytes();
        let restored = D64::from_be_bytes(bytes);
        assert_eq!(original, restored);
    }

    #[test]
    fn test_round_trip_ne() {
        let original = D64::from_str_exact("123.45678").unwrap();
        let bytes = original.to_ne_bytes();
        let restored = D64::from_ne_bytes(bytes);
        assert_eq!(original, restored);
    }

    #[test]
    fn test_bytes_constant() {
        assert_eq!(D64::BYTES, 8);
    }

    #[test]
    fn test_zero_bytes() {
        let zero_bytes = D64::ZERO.to_le_bytes();
        assert_eq!(zero_bytes, [0u8; 8]);
        assert_eq!(D64::from_le_bytes(zero_bytes), D64::ZERO);
    }
}

#[cfg(test)]
mod buffer_tests {
    use super::*;

    #[test]
    fn test_write_read_le_bytes() {
        let d = D64::from_str_exact("123.45").unwrap();
        let mut buf = [0u8; 16];

        d.write_le_bytes(&mut buf[4..]);
        let restored = D64::read_le_bytes(&buf[4..]);

        assert_eq!(d, restored);
    }

    #[test]
    fn test_write_read_be_bytes() {
        let d = D64::from_str_exact("-987.654321").unwrap();
        let mut buf = [0u8; 20];

        d.write_be_bytes(&mut buf[8..]);
        let restored = D64::read_be_bytes(&buf[8..]);

        assert_eq!(d, restored);
    }

    #[test]
    fn test_write_read_ne_bytes() {
        let d = D64::from_raw(999_888_777);
        let mut buf = [0u8; 8];

        d.write_ne_bytes(&mut buf);
        let restored = D64::read_ne_bytes(&buf);

        assert_eq!(d, restored);
    }

    #[test]
    #[should_panic]
    fn test_write_le_bytes_panic_short_buffer() {
        let d = D64::ONE;
        let mut buf = [0u8; 4];
        d.write_le_bytes(&mut buf);
    }

    #[test]
    #[should_panic]
    fn test_read_le_bytes_panic_short_buffer() {
        let buf = [0u8; 4];
        let _ = D64::read_le_bytes(&buf);
    }

    #[test]
    fn test_try_write_le_bytes_success() {
        let d = D64::from_raw(123_456_789);
        let mut buf = [0u8; 10];

        assert!(d.try_write_le_bytes(&mut buf[2..]).is_some());
        assert_eq!(D64::read_le_bytes(&buf[2..]), d);
    }

    #[test]
    fn test_try_write_le_bytes_failure() {
        let d = D64::ONE;
        let mut buf = [0u8; 4];

        assert!(d.try_write_le_bytes(&mut buf).is_none());
    }

    #[test]
    fn test_try_read_le_bytes_success() {
        let d = D64::from_str_exact("42.42").unwrap();
        let bytes = d.to_le_bytes();

        assert_eq!(D64::try_read_le_bytes(&bytes), Some(d));
    }

    #[test]
    fn test_try_read_le_bytes_failure() {
        let buf = [0u8; 4];
        assert!(D64::try_read_le_bytes(&buf).is_none());
    }

    #[test]
    fn test_multiple_writes_to_buffer() {
        let prices = [
            D64::from_str_exact("100.50").unwrap(),
            D64::from_str_exact("200.75").unwrap(),
            D64::from_str_exact("300.25").unwrap(),
        ];

        let mut buf = [0u8; 24];

        for (i, price) in prices.iter().enumerate() {
            price.write_le_bytes(&mut buf[i * 8..]);
        }

        for (i, expected) in prices.iter().enumerate() {
            let actual = D64::read_le_bytes(&buf[i * 8..]);
            assert_eq!(*expected, actual);
        }
    }

    #[test]
    fn test_buffer_at_exact_size() {
        let d = D64::from_raw(-12345);
        let mut buf = [0u8; 8];

        d.write_le_bytes(&mut buf);
        assert_eq!(D64::read_le_bytes(&buf), d);
    }
}

#[cfg(test)]
mod display_tests {
    use std::format;

    use super::*;

    #[test]
    fn test_display_integer() {
        assert_eq!(format!("{}", D64::from_raw(100_000_000)), "1");
        assert_eq!(format!("{}", D64::from_raw(4200_000_000)), "42");
        assert_eq!(format!("{}", D64::ZERO), "0");
    }

    #[test]
    fn test_display_decimal() {
        assert_eq!(format!("{}", D64::from_raw(123_450_000)), "1.2345");
        assert_eq!(format!("{}", D64::from_raw(100_000_001)), "1.00000001");
    }

    #[test]
    fn test_display_negative() {
        assert_eq!(format!("{}", D64::from_raw(-100_000_000)), "-1");
        assert_eq!(format!("{}", D64::from_raw(-123_450_000)), "-1.2345");
    }

    #[test]
    fn test_display_trailing_zeros_stripped() {
        assert_eq!(format!("{}", D64::from_raw(123_000_000)), "1.23");
        assert_eq!(format!("{}", D64::from_raw(100_100_000)), "1.001");
    }

    #[test]
    fn test_display_precision() {
        let d = D64::from_raw(123_456_789); // 1.23456789

        assert_eq!(format!("{:.0}", d), "1");
        assert_eq!(format!("{:.2}", d), "1.23");
        assert_eq!(format!("{:.4}", d), "1.2346"); // Rounds
        assert_eq!(format!("{:.8}", d), "1.23456789");
    }

    #[test]
    fn test_display_precision_rounding() {
        let d = D64::from_raw(125_500_000); // 1.255
        assert_eq!(format!("{:.2}", d), "1.26"); // Rounds up

        let d = D64::from_raw(125_400_000); // 1.254
        assert_eq!(format!("{:.2}", d), "1.25"); // Rounds down
    }

    #[test]
    fn test_display_zero() {
        assert_eq!(format!("{}", D64::ZERO), "0");
        assert_eq!(format!("{:.2}", D64::ZERO), "0");
    }
}

#[cfg(test)]
mod utf8_bytes_tests {
    use super::*;

    #[test]
    fn test_from_utf8_bytes_integer() {
        let bytes = b"123";
        let d = D64::from_utf8_bytes(bytes).unwrap();
        assert_eq!(d, D64::from_str_exact("123").unwrap());
    }

    #[test]
    fn test_from_utf8_bytes_decimal() {
        let bytes = b"123.45";
        let d = D64::from_utf8_bytes(bytes).unwrap();
        assert_eq!(d, D64::from_str_exact("123.45").unwrap());
    }

    #[test]
    fn test_from_utf8_bytes_negative() {
        let bytes = b"-987.654321";
        let d = D64::from_utf8_bytes(bytes).unwrap();
        assert_eq!(d, D64::from_str_exact("-987.654321").unwrap());
    }

    #[test]
    fn test_from_utf8_bytes_with_whitespace() {
        let bytes = b"  42.42  ";
        let d = D64::from_utf8_bytes(bytes).unwrap();
        assert_eq!(d, D64::from_str_exact("42.42").unwrap());
    }

    #[test]
    fn test_from_utf8_bytes_invalid_utf8() {
        let bytes = &[0xFF, 0xFE, 0xFD]; // Invalid UTF-8
        let result = D64::from_utf8_bytes(bytes);
        assert!(matches!(result, Err(DecimalError::InvalidFormat)));
    }

    #[test]
    fn test_from_utf8_bytes_invalid_decimal() {
        let bytes = b"not a number";
        let result = D64::from_utf8_bytes(bytes);
        assert!(matches!(result, Err(DecimalError::InvalidFormat)));
    }

    #[test]
    fn test_from_utf8_bytes_empty() {
        let bytes = b"";
        let result = D64::from_utf8_bytes(bytes);
        assert!(matches!(result, Err(DecimalError::InvalidFormat)));
    }

    #[test]
    fn test_from_utf8_bytes_zero() {
        let bytes = b"0";
        let d = D64::from_utf8_bytes(bytes).unwrap();
        assert_eq!(d, D64::ZERO);
    }

    #[test]
    fn test_from_utf8_bytes_from_network_buffer() {
        // Simulate reading from a network packet
        let packet = b"PRICE:100.50;QTY:1000";
        let price_bytes = &packet[6..12]; // "100.50"

        let price = D64::from_utf8_bytes(price_bytes).unwrap();
        assert_eq!(price, D64::from_str_exact("100.50").unwrap());
    }
}

#[cfg(test)]
mod percentage_tests {
    use super::*;

    #[test]
    fn test_percent_of_basic() {
        let amount = D64::from_str_exact("1000").unwrap();
        let percent = D64::from_str_exact("5").unwrap(); // 5%

        let result = amount.percent_of(percent).unwrap();
        assert_eq!(result, D64::from_str_exact("50").unwrap()); // 5% of 1000 = 50
    }

    #[test]
    fn test_percent_of_decimal() {
        let amount = D64::from_str_exact("250.50").unwrap();
        let percent = D64::from_str_exact("10").unwrap(); // 10%

        let result = amount.percent_of(percent).unwrap();
        assert_eq!(result, D64::from_str_exact("25.05").unwrap()); // 10% of 250.50 = 25.05
    }

    #[test]
    fn test_percent_of_fractional_percent() {
        let amount = D64::from_str_exact("1000").unwrap();
        let percent = D64::from_str_exact("2.5").unwrap(); // 2.5%

        let result = amount.percent_of(percent).unwrap();
        assert_eq!(result, D64::from_str_exact("25").unwrap()); // 2.5% of 1000 = 25
    }

    #[test]
    fn test_percent_of_zero() {
        let amount = D64::from_str_exact("1000").unwrap();
        let percent = D64::ZERO;

        let result = amount.percent_of(percent).unwrap();
        assert_eq!(result, D64::ZERO);
    }

    #[test]
    fn test_percent_of_hundred() {
        let amount = D64::from_str_exact("500").unwrap();
        let percent = D64::from_str_exact("100").unwrap(); // 100%

        let result = amount.percent_of(percent).unwrap();
        assert_eq!(result, D64::from_str_exact("500").unwrap()); // 100% of 500 = 500
    }

    #[test]
    fn test_percent_of_negative_amount() {
        let amount = D64::from_str_exact("-1000").unwrap();
        let percent = D64::from_str_exact("5").unwrap();

        let result = amount.percent_of(percent).unwrap();
        assert_eq!(result, D64::from_str_exact("-50").unwrap());
    }

    #[test]
    fn test_percent_of_overflow() {
        let amount = D64::MAX;
        let percent = D64::from_str_exact("200").unwrap();

        let result = amount.percent_of(percent);
        assert!(result.is_none());
    }

    #[test]
    fn test_add_percent_basic() {
        let amount = D64::from_str_exact("1000").unwrap();
        let percent = D64::from_str_exact("5").unwrap(); // Add 5%

        let result = amount.add_percent(percent).unwrap();
        assert_eq!(result, D64::from_str_exact("1050").unwrap()); // 1000 + 5% = 1050
    }

    #[test]
    fn test_add_percent_decimal() {
        let amount = D64::from_str_exact("200").unwrap();
        let percent = D64::from_str_exact("10").unwrap(); // Add 10%

        let result = amount.add_percent(percent).unwrap();
        assert_eq!(result, D64::from_str_exact("220").unwrap()); // 200 + 10% = 220
    }

    #[test]
    fn test_add_percent_fractional() {
        let amount = D64::from_str_exact("1000").unwrap();
        let percent = D64::from_str_exact("2.5").unwrap(); // Add 2.5%

        let result = amount.add_percent(percent).unwrap();
        assert_eq!(result, D64::from_str_exact("1025").unwrap()); // 1000 + 2.5% = 1025
    }

    #[test]
    fn test_add_percent_zero() {
        let amount = D64::from_str_exact("1000").unwrap();
        let percent = D64::ZERO;

        let result = amount.add_percent(percent).unwrap();
        assert_eq!(result, amount); // Adding 0% returns original
    }

    #[test]
    fn test_add_percent_negative() {
        let amount = D64::from_str_exact("1000").unwrap();
        let percent = D64::from_str_exact("-10").unwrap(); // Subtract 10%

        let result = amount.add_percent(percent).unwrap();
        assert_eq!(result, D64::from_str_exact("900").unwrap()); // 1000 - 10% = 900
    }

    #[test]
    fn test_add_percent_negative_amount() {
        let amount = D64::from_str_exact("-1000").unwrap();
        let percent = D64::from_str_exact("5").unwrap();

        let result = amount.add_percent(percent).unwrap();
        assert_eq!(result, D64::from_str_exact("-1050").unwrap());
    }

    #[test]
    fn test_add_percent_overflow() {
        let amount = D64::MAX;
        let percent = D64::from_str_exact("50").unwrap();

        let result = amount.add_percent(percent);
        assert!(result.is_none());
    }

    #[test]
    fn test_add_percent_hundred() {
        let amount = D64::from_str_exact("500").unwrap();
        let percent = D64::from_str_exact("100").unwrap(); // Add 100% = double

        let result = amount.add_percent(percent).unwrap();
        assert_eq!(result, D64::from_str_exact("1000").unwrap());
    }

    #[test]
    fn test_percent_commission_calculation() {
        // Real-world example: $10,000 trade with 0.1% commission
        let trade_value = D64::from_str_exact("10000").unwrap();
        let commission_rate = D64::from_str_exact("0.1").unwrap();

        let commission = trade_value.percent_of(commission_rate).unwrap();
        assert_eq!(commission, D64::from_str_exact("10").unwrap());
    }

    #[test]
    fn test_percent_tax_calculation() {
        // Real-world example: $99.99 item with 8.5% tax
        let price = D64::from_str_exact("99.99").unwrap();
        let tax_rate = D64::from_str_exact("8.5").unwrap();

        let total = price.add_percent(tax_rate).unwrap();
        // 99.99 * 1.085 = 108.48915
        // Rounded to 2 decimal places (cents) = 108.49
        assert_eq!(total.round_dp(2), D64::from_str_exact("108.49").unwrap());
    }

    #[test]
    fn test_percent_discount_calculation() {
        // Real-world example: $299 item with 15% discount
        let price = D64::from_str_exact("299").unwrap();
        let discount_rate = D64::from_str_exact("-15").unwrap(); // Negative for discount

        let final_price = price.add_percent(discount_rate).unwrap();
        assert_eq!(final_price, D64::from_str_exact("254.15").unwrap());
    }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use super::*;

    #[test]
    fn test_serialize() {
        let d = D64::from_str("123.45").unwrap();
        let json = serde_json::to_string(&d).unwrap();
        assert_eq!(json, r#""123.45""#);
    }

    #[test]
    fn test_deserialize() {
        let json = r#""123.45""#;
        let d: D64 = serde_json::from_str(json).unwrap();
        assert_eq!(d, D64::from_str("123.45").unwrap());
    }

    #[test]
    fn test_round_trip() {
        let original = D64::from_str("123.456789").unwrap();
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: D64 = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_deserialize_integer() {
        let json = r#""42""#;
        let d: D64 = serde_json::from_str(json).unwrap();
        assert_eq!(d, D64::from_i32(42));
    }

    #[test]
    fn test_serialize_zero() {
        let json = serde_json::to_string(&D64::ZERO).unwrap();
        assert_eq!(json, r#""0""#);
    }

    #[test]
    fn test_serialize_negative() {
        let d = D64::from_str("-123.45").unwrap();
        let json = serde_json::to_string(&d).unwrap();
        assert_eq!(json, r#""-123.45""#);
    }
}
