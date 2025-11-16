use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::str::FromStr;

use crate::DecimalError;

/// 64-bit fixed-point decimal with 8 decimal places of precision.
///
/// Range: ±92,233,720,368.54775807
/// Precision: 0.00000001
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct D64 {
    value: i64,
}

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

    /// One cent (0.01) - useful for USD and similar currencies
    pub const CENT: Self = Self {
        value: Self::SCALE / 100,
    };

    /// One basis point (0.0001) - common in finance for interest rates
    pub const BASIS_POINT: Self = Self {
        value: Self::SCALE / 10_000,
    };

    /// Creates a new D64 from a raw scaled value.
    ///
    /// # Safety
    /// The caller must ensure the value is properly scaled by 10^8.
    #[inline]
    pub const fn from_raw(value: i64) -> Self {
        Self { value }
    }

    /// Returns the raw internal value (scaled by 10^8).
    #[inline]
    pub const fn to_raw(self) -> i64 {
        self.value
    }

    // ===== Addition =====

    /// Checked addition. Returns `None` if overflow occurred.
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn checked_add(self, rhs: Self) -> Option<Self> {
        if let Some(result) = self.value.checked_add(rhs.value) {
            Some(Self { value: result })
        } else {
            None
        }
    }

    /// Saturating addition. Clamps on overflow.
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn saturating_add(self, rhs: Self) -> Self {
        Self {
            value: self.value.saturating_add(rhs.value),
        }
    }

    /// Wrapping addition. Wraps on overflow.
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn wrapping_add(self, rhs: Self) -> Self {
        Self {
            value: self.value.wrapping_add(rhs.value),
        }
    }

    // ===== Subtraction =====

    /// Checked subtraction. Returns `None` if overflow occurred.
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn checked_sub(self, rhs: Self) -> Option<Self> {
        if let Some(result) = self.value.checked_sub(rhs.value) {
            Some(Self { value: result })
        } else {
            None
        }
    }

    /// Saturating subtraction. Clamps on overflow.
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn saturating_sub(self, rhs: Self) -> Self {
        Self {
            value: self.value.saturating_sub(rhs.value),
        }
    }

    /// Wrapping subtraction. Wraps on overflow.
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn wrapping_sub(self, rhs: Self) -> Self {
        Self {
            value: self.value.wrapping_sub(rhs.value),
        }
    }

    // ===== Multiplication =====

    /// Checked multiplication. Returns `None` if overflow occurred.
    ///
    /// Internally widens to i128 to prevent intermediate overflow.
    #[inline]
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
    #[inline]
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
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn wrapping_mul(self, rhs: Self) -> Self {
        let a = self.value as i128;
        let b = rhs.value as i128;

        let result = a * b / Self::SCALE as i128;

        Self {
            value: result as i64,
        }
    }

    // ===== Division =====

    /// Checked division. Returns `None` if `rhs` is zero or overflow occurred.
    ///
    /// Internally widens to i128 to maintain precision.
    #[inline]
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
    #[inline]
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
    #[inline]
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

    // ===== Negation =====

    /// Checked negation. Returns `None` if the result would overflow.
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn checked_neg(self) -> Option<Self> {
        if let Some(result) = self.value.checked_neg() {
            Some(Self { value: result })
        } else {
            None
        }
    }

    /// Saturating negation. Clamps on overflow.
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn saturating_neg(self) -> Self {
        Self {
            value: self.value.saturating_neg(),
        }
    }

    /// Wrapping negation. Wraps on overflow.
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn wrapping_neg(self) -> Self {
        Self {
            value: self.value.wrapping_neg(),
        }
    }

    // ===== Absolute Value =====

    /// Returns the absolute value of `self`.
    #[inline]
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
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn checked_abs(self) -> Option<Self> {
        if self.value == i64::MIN {
            None
        } else {
            Some(self.abs())
        }
    }

    /// Saturating absolute value. Clamps on overflow.
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn saturating_abs(self) -> Self {
        if self.value == i64::MIN {
            Self::MAX
        } else {
            self.abs()
        }
    }

    /// Wrapping absolute value. Wraps on overflow.
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn wrapping_abs(self) -> Self {
        Self {
            value: self.value.wrapping_abs(),
        }
    }

    // ===== Sign Operations =====

    /// Returns `true` if `self` is positive.
    #[inline]
    pub const fn is_positive(self) -> bool {
        self.value > 0
    }

    /// Returns `true` if `self` is negative.
    #[inline]
    pub const fn is_negative(self) -> bool {
        self.value < 0
    }

    /// Returns `true` if `self` is zero.
    #[inline]
    pub const fn is_zero(self) -> bool {
        self.value == 0
    }

    /// Returns the sign of `self` as -1, 0, or 1.
    #[inline]
    pub const fn signum(self) -> i32 {
        if self.value > 0 {
            1
        } else if self.value < 0 {
            -1
        } else {
            0
        }
    }

    // ===== Integer Conversions =====

    /// Creates a D64 from an i64 integer.
    #[inline]
    pub const fn from_i64(value: i64) -> Option<Self> {
        // Check if multiplication would overflow
        match value.checked_mul(Self::SCALE) {
            Some(scaled) => Some(Self { value: scaled }),
            None => None,
        }
    }

    /// Creates a D64 from an i32 integer (always succeeds).
    #[inline]
    pub const fn from_i32(value: i32) -> Self {
        Self {
            value: value as i64 * Self::SCALE,
        }
    }

    /// Creates a D64 from a u32 integer (always succeeds).
    #[inline]
    pub const fn from_u32(value: u32) -> Self {
        Self {
            value: value as i64 * Self::SCALE,
        }
    }

    /// Creates a D64 from a u64 integer.
    #[inline]
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
    #[inline]
    pub const fn to_i64(self) -> i64 {
        self.value / Self::SCALE
    }

    /// Converts to i64, rounding to nearest (banker's rounding on ties).
    #[inline]
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

    // ===== Float Conversions =====

    /// Creates a D64 from an f64.
    ///
    /// Returns `None` if the value is NaN, infinite, or out of range.
    #[inline]
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
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.value as f64 / Self::SCALE as f64
    }

    /// Creates a D64 from an f32.
    #[inline]
    pub fn from_f32(value: f32) -> Option<Self> {
        Self::from_f64(value as f64)
    }

    /// Converts to f32.
    ///
    /// Note: May lose precision.
    #[inline]
    pub fn to_f32(self) -> f32 {
        self.to_f64() as f32
    }

    // ===== Fallible Conversions with Error Type =====

    /// Creates a D64 from an i64, returning an error on overflow.
    #[inline]
    pub const fn try_from_i64(value: i64) -> crate::Result<Self> {
        match Self::from_i64(value) {
            Some(v) => Ok(v),
            None => Err(DecimalError::Overflow),
        }
    }

    /// Creates a D64 from a u64, returning an error on overflow.
    #[inline]
    pub const fn try_from_u64(value: u64) -> crate::Result<Self> {
        match Self::from_u64(value) {
            Some(v) => Ok(v),
            None => Err(DecimalError::Overflow),
        }
    }

    /// Creates a D64 from an f64, returning an error if invalid.
    #[inline]
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

    // ===== Comparison Utilities =====

    /// Returns the minimum of two values.
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn min(self, other: Self) -> Self {
        if self.value < other.value {
            self
        } else {
            other
        }
    }

    /// Returns the maximum of two values.
    #[inline]
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
    #[inline]
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

    // ===== Rounding Operations =====

    /// Returns the largest integer less than or equal to `self`.
    #[inline]
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
    #[inline]
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
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn trunc(self) -> Self {
        Self {
            value: (self.value / Self::SCALE) * Self::SCALE,
        }
    }

    /// Returns the fractional part of `self`.
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn fract(self) -> Self {
        Self {
            value: self.value % Self::SCALE,
        }
    }

    /// Rounds to the nearest integer, using banker's rounding (round half to even).
    #[inline]
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
    #[inline]
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

    // ===== Mathematical Operations =====

    /// Returns the reciprocal (multiplicative inverse) of `self`.
    ///
    /// Returns `None` if `self` is zero.
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn recip(self) -> Option<Self> {
        if self.value == 0 {
            None
        } else {
            Self::ONE.checked_div(self)
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

    // ===== Arithmetic Operations with Result =====

    /// Checked addition. Returns an error if overflow occurred.
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn try_add(self, rhs: Self) -> crate::Result<Self> {
        match self.checked_add(rhs) {
            Some(result) => Ok(result),
            None => Err(DecimalError::Overflow),
        }
    }

    /// Checked subtraction. Returns an error if overflow occurred.
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn try_sub(self, rhs: Self) -> crate::Result<Self> {
        match self.checked_sub(rhs) {
            Some(result) => Ok(result),
            None => Err(DecimalError::Overflow),
        }
    }

    /// Checked multiplication. Returns an error if overflow occurred.
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn try_mul(self, rhs: Self) -> crate::Result<Self> {
        match self.checked_mul(rhs) {
            Some(result) => Ok(result),
            None => Err(DecimalError::Overflow),
        }
    }

    /// Checked division. Returns an error if `rhs` is zero or overflow occurred.
    #[inline]
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

    /// Checked negation. Returns an error if overflow occurred.
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn try_neg(self) -> crate::Result<Self> {
        match self.checked_neg() {
            Some(result) => Ok(result),
            None => Err(DecimalError::Overflow),
        }
    }

    /// Checked absolute value. Returns an error if overflow occurred.
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn try_abs(self) -> crate::Result<Self> {
        match self.checked_abs() {
            Some(result) => Ok(result),
            None => Err(DecimalError::Overflow),
        }
    }

    /// Checked reciprocal. Returns an error if `self` is zero.
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn try_recip(self) -> crate::Result<Self> {
        match self.recip() {
            Some(result) => Ok(result),
            None => Err(DecimalError::DivisionByZero),
        }
    }

    /// Checked integer power. Returns an error if overflow occurred.
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn try_powi(self, exp: i32) -> crate::Result<Self> {
        match self.powi(exp) {
            Some(result) => Ok(result),
            None => Err(DecimalError::Overflow),
        }
    }

    /// Checked square root. Returns an error if `self` is negative.
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn try_sqrt(self) -> crate::Result<Self> {
        match self.sqrt() {
            Some(result) => Ok(result),
            None => Err(DecimalError::InvalidFormat), // or create a new error variant like NegativeSqrt
        }
    }

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
    pub fn from_str(s: &str) -> crate::Result<Self> {
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

        let value = if is_negative {
            // Handle the case where integer_part might already be negative
            if integer_part < 0 {
                // Integer part was negative (like "-0" edge case)
                -abs_value
            } else {
                -abs_value
            }
        } else {
            abs_value
        };

        Ok(Self { value })
    }
}

impl FromStr for D64 {
    type Err = DecimalError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str(s)
    }
}

// ===== Operator Overloading =====
// These use checked operations and panic on overflow (matching std behavior)
impl Add for D64 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.checked_add(rhs).expect("attempt to add with overflow")
    }
}

impl Sub for D64 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self.checked_sub(rhs)
            .expect("attempt to subtract with overflow")
    }
}

impl Mul for D64 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.checked_mul(rhs)
            .expect("attempt to multiply with overflow")
    }
}

impl Div for D64 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self.checked_div(rhs)
            .expect("attempt to divide by zero or overflow")
    }
}

impl Neg for D64 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self.checked_neg().expect("attempt to negate with overflow")
    }
}

// Compound assignment operators
impl AddAssign for D64 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for D64 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for D64 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl DivAssign for D64 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

// ===== TryFrom implementations =====

impl TryFrom<i64> for D64 {
    type Error = DecimalError;

    #[inline]
    fn try_from(value: i64) -> crate::Result<Self> {
        Self::try_from_i64(value)
    }
}

impl TryFrom<u64> for D64 {
    type Error = DecimalError;

    #[inline]
    fn try_from(value: u64) -> crate::Result<Self> {
        Self::try_from_u64(value)
    }
}

impl TryFrom<f64> for D64 {
    type Error = DecimalError;

    #[inline]
    fn try_from(value: f64) -> crate::Result<Self> {
        Self::try_from_f64(value)
    }
}

impl TryFrom<f32> for D64 {
    type Error = DecimalError;

    #[inline]
    fn try_from(value: f32) -> crate::Result<Self> {
        Self::try_from_f64(value as f64)
    }
}

// ===== From implementations for types that always succeed =====

impl From<i32> for D64 {
    #[inline]
    fn from(value: i32) -> Self {
        Self::from_i32(value)
    }
}

impl From<u32> for D64 {
    #[inline]
    fn from(value: u32) -> Self {
        Self::from_u32(value)
    }
}

impl From<i16> for D64 {
    #[inline]
    fn from(value: i16) -> Self {
        Self::from_i32(value as i32)
    }
}

impl From<u16> for D64 {
    #[inline]
    fn from(value: u16) -> Self {
        Self::from_u32(value as u32)
    }
}

impl From<i8> for D64 {
    #[inline]
    fn from(value: i8) -> Self {
        Self::from_i32(value as i32)
    }
}

impl From<u8> for D64 {
    #[inline]
    fn from(value: u8) -> Self {
        Self::from_u32(value as u32)
    }
}

// ===== Helper Functions =====

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
        assert_eq!(D64::from_str("123").unwrap().to_raw(), 12_300_000_000);
        assert_eq!(D64::from_str("0").unwrap().to_raw(), 0);
        assert_eq!(D64::from_str("-456").unwrap().to_raw(), -45_600_000_000);
    }

    #[test]
    fn test_from_str_decimal() {
        assert_eq!(D64::from_str("123.45").unwrap().to_raw(), 12_345_000_000);
        assert_eq!(D64::from_str("0.00000001").unwrap().to_raw(), 1);
        assert_eq!(D64::from_str("-123.45").unwrap().to_raw(), -12_345_000_000);
    }

    #[test]
    fn test_from_str_leading_decimal() {
        assert_eq!(D64::from_str("0.5").unwrap().to_raw(), 50_000_000);
        assert_eq!(D64::from_str(".5").unwrap().to_raw(), 50_000_000);
    }

    #[test]
    fn test_from_str_trailing_zeros() {
        assert_eq!(
            D64::from_str("123.45000000").unwrap().to_raw(),
            12_345_000_000
        );
        assert_eq!(D64::from_str("1.10").unwrap().to_raw(), 110_000_000);
    }

    #[test]
    fn test_from_str_plus_sign() {
        assert_eq!(D64::from_str("+123.45").unwrap().to_raw(), 12_345_000_000);
    }

    #[test]
    fn test_from_str_whitespace() {
        assert_eq!(
            D64::from_str("  123.45  ").unwrap().to_raw(),
            12_345_000_000
        );
    }

    #[test]
    fn test_from_str_errors() {
        assert!(D64::from_str("").is_err());
        assert!(D64::from_str("abc").is_err());
        assert!(D64::from_str("12.34.56").is_err());
        assert!(matches!(
            D64::from_str("123.123456789"),
            Err(DecimalError::PrecisionLoss)
        ));
    }

    #[test]
    fn test_from_str_overflow() {
        assert!(matches!(
            D64::from_str("999999999999999"),
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
        assert!(D64::from_str(".").is_err());

        // Multiple signs
        assert!(D64::from_str("--123").is_err());

        // Sign after number
        assert!(D64::from_str("123-").is_err());
    }
}

