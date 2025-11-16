use std::hint::black_box;
use std::str::FromStr;

use criterion::{Criterion, criterion_group, criterion_main};
use fixdec::D96; // Replace with your actual crate name

fn bench_addition(c: &mut Criterion) {
    c.bench_function("d96_addition", |b| {
        let x = D96::from_str("123.456789").unwrap();
        let y = D96::from_str("987.654321").unwrap();
        b.iter(|| black_box(black_box(x) + black_box(y)));
    });
}

fn bench_subtraction(c: &mut Criterion) {
    c.bench_function("d96_subtraction", |b| {
        let x = D96::from_str("987.654321").unwrap();
        let y = D96::from_str("123.456789").unwrap();
        b.iter(|| black_box(black_box(x) - black_box(y)));
    });
}

fn bench_multiplication(c: &mut Criterion) {
    c.bench_function("d96_multiplication", |b| {
        let x = D96::from_str("123.456789").unwrap();
        let y = D96::from_str("9.876543").unwrap();
        b.iter(|| black_box(black_box(x) * black_box(y)));
    });
}

fn bench_division(c: &mut Criterion) {
    c.bench_function("d96_division", |b| {
        let x = D96::from_str("123.456789").unwrap();
        let y = D96::from_str("9.876543").unwrap();
        b.iter(|| black_box(black_box(x) / black_box(y)));
    });
}

fn bench_parsing(c: &mut Criterion) {
    c.bench_function("d96_parsing", |b| {
        b.iter(|| black_box(D96::from_str("123.456789").unwrap()));
    });
}

fn bench_formatting(c: &mut Criterion) {
    c.bench_function("d96_formatting", |b| {
        let d = D96::from_str("123.456789").unwrap();
        b.iter(|| black_box(format!("{}", d)));
    });
}

fn bench_price_times_quantity_mul_i64(c: &mut Criterion) {
    c.bench_function("d96_price_times_quantity_mul_i64", |b| {
        let price = D96::from_str("123.45").unwrap();
        let quantity = 1000i128;
        b.iter(|| black_box(price.mul_i128(black_box(quantity)).unwrap()));
    });
}

fn bench_price_times_quantity_mul_d96(c: &mut Criterion) {
    c.bench_function("d96_price_times_quantity_mul_d96", |b| {
        let price = D96::from_str("123.45").unwrap();
        let quantity = D96::from_i128(1000).unwrap();
        b.iter(|| black_box(black_box(price) * black_box(quantity)));
    });
}

fn bench_sum(c: &mut Criterion) {
    c.bench_function("d96_sum_1000_values", |b| {
        let values: Vec<D96> = (0..1000)
            .map(|i| D96::from_str(&format!("{}.{:02}", i, i % 100)).unwrap())
            .collect();
        b.iter(|| black_box(values.iter().copied().sum::<D96>()));
    });
}

fn bench_rounding(c: &mut Criterion) {
    c.bench_function("d96_round_to_2_decimals", |b| {
        let d = D96::from_str("123.456789").unwrap();
        b.iter(|| black_box(black_box(d).round_dp(2)));
    });
}

fn bench_binary_write_read(c: &mut Criterion) {
    c.bench_function("d96_binary_write_read", |b| {
        let d = D96::from_str("123.456789").unwrap();
        let mut buf = [0u8; 16];
        b.iter(|| {
            d.write_le_bytes(&mut buf);
            black_box(D96::read_le_bytes(&buf))
        });
    });
}

fn bench_comparison(c: &mut Criterion) {
    c.bench_function("d96_comparison", |b| {
        let x = D96::from_str("123.456789").unwrap();
        let y = D96::from_str("123.456790").unwrap();
        b.iter(|| black_box(black_box(x) < black_box(y)));
    });
}

fn bench_sqrt(c: &mut Criterion) {
    c.bench_function("d96_sqrt", |b| {
        let d = D96::from_str("123.456789").unwrap();
        b.iter(|| black_box(black_box(d).sqrt().unwrap()));
    });
}

fn bench_powi(c: &mut Criterion) {
    c.bench_function("d96_powi", |b| {
        let d = D96::from_str("1.05").unwrap();
        b.iter(|| black_box(black_box(d).powi(10).unwrap()));
    });
}

fn bench_percentage_of(c: &mut Criterion) {
    c.bench_function("d96_percent_of", |b| {
        let amount = D96::from_str("1000").unwrap();
        let percent = D96::from_str("5").unwrap();
        b.iter(|| black_box(black_box(amount).percent_of(black_box(percent)).unwrap()));
    });
}

fn bench_add_percent(c: &mut Criterion) {
    c.bench_function("d96_add_percent", |b| {
        let amount = D96::from_str("1000").unwrap();
        let percent = D96::from_str("5").unwrap();
        b.iter(|| black_box(black_box(amount).add_percent(black_box(percent)).unwrap()));
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
    bench_price_times_quantity_mul_i64,
    bench_price_times_quantity_mul_d96,
    bench_sum,
    bench_rounding,
    bench_binary_write_read,
    bench_comparison,
    bench_sqrt,
    bench_powi,
    bench_percentage_of,
    bench_add_percent,
);

criterion_main!(benches);
