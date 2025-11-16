/// 128-bit fixed-point decimal with 18 decimal places of precision.
///
/// Range: ±170,141,183,460,469,231.731687303715884105727
/// Precision: 0.000000000000000001
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct D128 {
    value: i128,
}

impl D128 {
    /// The scale factor: 10^18
    pub const SCALE: i128 = 1_000_000_000_000_000_000;

    /// The number of decimal places
    pub const DECIMALS: u8 = 18;

    /// Maximum value: ~170 trillion
    pub const MAX: Self = Self { value: i128::MAX };

    /// Minimum value: ~-170 trillion
    pub const MIN: Self = Self { value: i128::MIN };

    /// Zero
    pub const ZERO: Self = Self { value: 0 };

    /// One (1.0)
    pub const ONE: Self = Self { value: Self::SCALE };

    /// Creates a new D128 from a raw scaled value.
    ///
    /// # Safety
    /// The caller must ensure the value is properly scaled by 10^18.
    #[inline]
    pub const fn from_raw(value: i128) -> Self {
        Self { value }
    }

    /// Returns the raw internal value (scaled by 10^18).
    #[inline]
    pub const fn to_raw(self) -> i128 {
        self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_d128_constants() {
        assert_eq!(D128::ZERO.to_raw(), 0);
        assert_eq!(D128::ONE.to_raw(), 1_000_000_000_000_000_000);
        assert_eq!(D128::SCALE, 1_000_000_000_000_000_000);
    }
}
