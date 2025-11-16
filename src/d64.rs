use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// 64-bit fixed-point decimal with 8 decimal places of precision.
///
/// Range: ±92,233,720,368.54775807
/// Precision: 0.00000001
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct D64 {
    value: i64,
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

