use criterion::{Criterion, criterion_group, criterion_main};
use rust_decimal::{Decimal, MathematicalOps};
use std::hint::black_box;
use std::str::FromStr;

fn bench_addition(c: &mut Criterion) {
    c.bench_function("rust_decimal_addition", |b| {
        let x = Decimal::from_str("123.456789").unwrap();
        let y = Decimal::from_str("987.654321").unwrap();
        b.iter(|| black_box(black_box(x) + black_box(y)));
    });
}

fn bench_subtraction(c: &mut Criterion) {
    c.bench_function("rust_decimal_subtraction", |b| {
        let x = Decimal::from_str("987.654321").unwrap();
        let y = Decimal::from_str("123.456789").unwrap();
        b.iter(|| black_box(black_box(x) - black_box(y)));
    });
}

fn bench_multiplication(c: &mut Criterion) {
    c.bench_function("rust_decimal_multiplication", |b| {
        let x = Decimal::from_str("123.456789").unwrap();
        let y = Decimal::from_str("9.876543").unwrap();
        b.iter(|| black_box(black_box(x) * black_box(y)));
    });
}

fn bench_division(c: &mut Criterion) {
    c.bench_function("rust_decimal_division", |b| {
        let x = Decimal::from_str("123.456789").unwrap();
        let y = Decimal::from_str("9.876543").unwrap();
        b.iter(|| black_box(black_box(x) / black_box(y)));
    });
}

fn bench_parsing(c: &mut Criterion) {
    c.bench_function("rust_decimal_parsing", |b| {
        b.iter(|| black_box(Decimal::from_str("123.456789").unwrap()));
    });
}

fn bench_formatting(c: &mut Criterion) {
    c.bench_function("rust_decimal_formatting", |b| {
        let d = Decimal::from_str("123.456789").unwrap();
        b.iter(|| black_box(format!("{}", d)));
    });
}

fn bench_price_times_quantity(c: &mut Criterion) {
    c.bench_function("rust_decimal_price_times_quantity", |b| {
        let price = Decimal::from_str("123.45").unwrap();
        let quantity = Decimal::from(1000i64);
        b.iter(|| black_box(black_box(price) * black_box(quantity)));
    });
}

fn bench_sum(c: &mut Criterion) {
    c.bench_function("rust_decimal_sum_1000_values", |b| {
        let values: Vec<Decimal> = (0..1000)
            .map(|i| Decimal::from_str(&format!("{}.{:02}", i, i % 100)).unwrap())
            .collect();
        b.iter(|| black_box(values.iter().copied().sum::<Decimal>()));
    });
}

fn bench_rounding(c: &mut Criterion) {
    c.bench_function("rust_decimal_round_to_2_decimals", |b| {
        let d = Decimal::from_str("123.456789").unwrap();
        b.iter(|| black_box(black_box(d).round_dp(2)));
    });
}

fn bench_comparison(c: &mut Criterion) {
    c.bench_function("rust_decimal_comparison", |b| {
        let x = Decimal::from_str("123.456789").unwrap();
        let y = Decimal::from_str("123.456790").unwrap();
        b.iter(|| black_box(black_box(x) < black_box(y)));
    });
}

fn bench_sqrt(c: &mut Criterion) {
    c.bench_function("rust_decimal_sqrt", |b| {
        let d = Decimal::from_str("123.456789").unwrap();
        b.iter(|| black_box(black_box(d).sqrt().unwrap()));
    });
}

fn bench_powi(c: &mut Criterion) {
    c.bench_function("rust_decimal_powi", |b| {
        let d = Decimal::from_str("1.05").unwrap();
        b.iter(|| black_box(black_box(d).powi(10)));
    });
}

criterion_group!(
    benches,
    bench_addition,
    bench_subtraction,
    bench_multiplication,
    bench_division,
    bench_parsing,
    bench_formatting,
    bench_price_times_quantity,
    bench_sum,
    bench_rounding,
    bench_comparison,
    bench_sqrt,
    bench_powi,
);

criterion_main!(benches);
