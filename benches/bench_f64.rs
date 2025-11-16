use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn bench_addition(c: &mut Criterion) {
    c.bench_function("f64_addition", |b| {
        let x = 123.456789_f64;
        let y = 987.654321_f64;
        b.iter(|| black_box(black_box(x) + black_box(y)));
    });
}

fn bench_subtraction(c: &mut Criterion) {
    c.bench_function("f64_subtraction", |b| {
        let x = 987.654321_f64;
        let y = 123.456789_f64;
        b.iter(|| black_box(black_box(x) - black_box(y)));
    });
}

fn bench_multiplication(c: &mut Criterion) {
    c.bench_function("f64_multiplication", |b| {
        let x = 123.456789_f64;
        let y = 9.876543_f64;
        b.iter(|| black_box(black_box(x) * black_box(y)));
    });
}

fn bench_division(c: &mut Criterion) {
    c.bench_function("f64_division", |b| {
        let x = 123.456789_f64;
        let y = 9.876543_f64;
        b.iter(|| black_box(black_box(x) / black_box(y)));
    });
}

fn bench_parsing(c: &mut Criterion) {
    c.bench_function("f64_parsing", |b| {
        b.iter(|| black_box("123.456789".parse::<f64>().unwrap()));
    });
}

fn bench_formatting(c: &mut Criterion) {
    c.bench_function("f64_formatting", |b| {
        let d = 123.456789_f64;
        b.iter(|| black_box(format!("{}", d)));
    });
}

fn bench_price_times_quantity(c: &mut Criterion) {
    c.bench_function("f64_price_times_quantity", |b| {
        let price = 123.45_f64;
        let quantity = 1000_f64;
        b.iter(|| black_box(black_box(price) * black_box(quantity)));
    });
}

fn bench_sum(c: &mut Criterion) {
    c.bench_function("f64_sum_1000_values", |b| {
        let values: Vec<f64> = (0..1000)
            .map(|i| format!("{}.{:02}", i, i % 100).parse().unwrap())
            .collect();
        b.iter(|| black_box(values.iter().copied().sum::<f64>()));
    });
}

fn bench_rounding(c: &mut Criterion) {
    c.bench_function("f64_round_to_2_decimals", |b| {
        let d = 123.456789_f64;
        b.iter(|| black_box((black_box(d) * 100.0).round() / 100.0));
    });
}

fn bench_binary_write_read(c: &mut Criterion) {
    c.bench_function("f64_binary_write_read", |b| {
        let d = 123.456789_f64;
        let mut buf = [0u8; 8];
        b.iter(|| {
            buf.copy_from_slice(&d.to_le_bytes());
            black_box(f64::from_le_bytes(buf))
        });
    });
}

fn bench_comparison(c: &mut Criterion) {
    c.bench_function("f64_comparison", |b| {
        let x = 123.456789_f64;
        let y = 123.456790_f64;
        b.iter(|| black_box(black_box(x) < black_box(y)));
    });
}

fn bench_sqrt(c: &mut Criterion) {
    c.bench_function("f64_sqrt", |b| {
        let d = 123.456789_f64;
        b.iter(|| black_box(black_box(d).sqrt()));
    });
}

fn bench_powi(c: &mut Criterion) {
    c.bench_function("f64_powi", |b| {
        let d = 1.05_f64;
        b.iter(|| black_box(black_box(d).powi(10)));
    });
}

fn bench_percentage_of(c: &mut Criterion) {
    c.bench_function("f64_percent_of", |b| {
        let amount = 1000.0_f64;
        let percent = 5.0_f64;
        b.iter(|| black_box(black_box(amount) * (black_box(percent) / 100.0)));
    });
}

fn bench_add_percent(c: &mut Criterion) {
    c.bench_function("f64_add_percent", |b| {
        let amount = 1000.0_f64;
        let percent = 5.0_f64;
        b.iter(|| black_box(black_box(amount) * (1.0 + black_box(percent) / 100.0)));
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
    bench_binary_write_read,
    bench_comparison,
    bench_sqrt,
    bench_powi,
    bench_percentage_of,
    bench_add_percent,
);

criterion_main!(benches);
