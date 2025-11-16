use criterion::{Criterion, criterion_group, criterion_main};
use fixdec::{D64, D96};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::{hint::black_box, str::FromStr};

// ============================================================================
// JSON Serialization/Deserialization
// ============================================================================

fn bench_d64_serialize_json(c: &mut Criterion) {
    c.bench_function("d64_serialize_json", |b| {
        let d = D64::from_str("123.456789").unwrap();
        b.iter(|| black_box(serde_json::to_string(&black_box(d)).unwrap()));
    });
}

fn bench_d64_deserialize_json(c: &mut Criterion) {
    c.bench_function("d64_deserialize_json", |b| {
        let json = r#""123.456789""#;
        b.iter(|| black_box(serde_json::from_str::<D64>(black_box(json)).unwrap()));
    });
}

fn bench_d64_roundtrip_json(c: &mut Criterion) {
    c.bench_function("d64_roundtrip_json", |b| {
        let d = D64::from_str("123.456789").unwrap();
        b.iter(|| {
            let json = serde_json::to_string(&black_box(d)).unwrap();
            black_box(serde_json::from_str::<D64>(&json).unwrap())
        });
    });
}

fn bench_d96_serialize_json(c: &mut Criterion) {
    c.bench_function("d96_serialize_json", |b| {
        let d = D96::from_str("123.456789").unwrap();
        b.iter(|| black_box(serde_json::to_string(&black_box(d)).unwrap()));
    });
}

fn bench_d96_deserialize_json(c: &mut Criterion) {
    c.bench_function("d96_deserialize_json", |b| {
        let json = r#""123.456789""#;
        b.iter(|| black_box(serde_json::from_str::<D96>(black_box(json)).unwrap()));
    });
}

fn bench_d96_roundtrip_json(c: &mut Criterion) {
    c.bench_function("d96_roundtrip_json", |b| {
        let d = D96::from_str("123.456789").unwrap();
        b.iter(|| {
            let json = serde_json::to_string(&black_box(d)).unwrap();
            black_box(serde_json::from_str::<D96>(&json).unwrap())
        });
    });
}

fn bench_rust_decimal_serialize_json(c: &mut Criterion) {
    c.bench_function("rust_decimal_serialize_json", |b| {
        let d = Decimal::from_str_exact("123.456789").unwrap();
        b.iter(|| black_box(serde_json::to_string(&black_box(d)).unwrap()));
    });
}

fn bench_rust_decimal_deserialize_json(c: &mut Criterion) {
    c.bench_function("rust_decimal_deserialize_json", |b| {
        let json = r#""123.456789""#;
        b.iter(|| black_box(serde_json::from_str::<Decimal>(black_box(json)).unwrap()));
    });
}

fn bench_rust_decimal_roundtrip_json(c: &mut Criterion) {
    c.bench_function("rust_decimal_roundtrip_json", |b| {
        let d = Decimal::from_str_exact("123.456789").unwrap();
        b.iter(|| {
            let json = serde_json::to_string(&black_box(d)).unwrap();
            black_box(serde_json::from_str::<Decimal>(&json).unwrap())
        });
    });
}

// ============================================================================
// Struct with Multiple Decimals (Realistic Scenario)
// ============================================================================

#[derive(Serialize, Deserialize)]
struct TradeD64 {
    price: D64,
    quantity: D64,
    commission: D64,
}

#[derive(Serialize, Deserialize)]
struct TradeD96 {
    price: D96,
    quantity: D96,
    commission: D96,
}

#[derive(Serialize, Deserialize)]
struct TradeRustDecimal {
    price: Decimal,
    quantity: Decimal,
    commission: Decimal,
}

fn bench_d64_struct_serialize_json(c: &mut Criterion) {
    c.bench_function("d64_struct_serialize_json", |b| {
        let trade = TradeD64 {
            price: D64::from_str("123.45").unwrap(),
            quantity: D64::from_str("1000").unwrap(),
            commission: D64::from_str("2.50").unwrap(),
        };
        b.iter(|| black_box(serde_json::to_string(&black_box(&trade)).unwrap()));
    });
}

fn bench_d64_struct_deserialize_json(c: &mut Criterion) {
    c.bench_function("d64_struct_deserialize_json", |b| {
        let json = r#"{"price":"123.45","quantity":"1000","commission":"2.50"}"#;
        b.iter(|| black_box(serde_json::from_str::<TradeD64>(black_box(json)).unwrap()));
    });
}

fn bench_d96_struct_serialize_json(c: &mut Criterion) {
    c.bench_function("d96_struct_serialize_json", |b| {
        let trade = TradeD96 {
            price: D96::from_str("123.45").unwrap(),
            quantity: D96::from_str("1000").unwrap(),
            commission: D96::from_str("2.50").unwrap(),
        };
        b.iter(|| black_box(serde_json::to_string(&black_box(&trade)).unwrap()));
    });
}

fn bench_d96_struct_deserialize_json(c: &mut Criterion) {
    c.bench_function("d96_struct_deserialize_json", |b| {
        let json = r#"{"price":"123.45","quantity":"1000","commission":"2.50"}"#;
        b.iter(|| black_box(serde_json::from_str::<TradeD96>(black_box(json)).unwrap()));
    });
}

fn bench_rust_decimal_struct_serialize_json(c: &mut Criterion) {
    c.bench_function("rust_decimal_struct_serialize_json", |b| {
        let trade = TradeRustDecimal {
            price: Decimal::from_str_exact("123.45").unwrap(),
            quantity: Decimal::from_str_exact("1000").unwrap(),
            commission: Decimal::from_str_exact("2.50").unwrap(),
        };
        b.iter(|| black_box(serde_json::to_string(&black_box(&trade)).unwrap()));
    });
}

fn bench_rust_decimal_struct_deserialize_json(c: &mut Criterion) {
    c.bench_function("rust_decimal_struct_deserialize_json", |b| {
        let json = r#"{"price":"123.45","quantity":"1000","commission":"2.50"}"#;
        b.iter(|| black_box(serde_json::from_str::<TradeRustDecimal>(black_box(json)).unwrap()));
    });
}

// ============================================================================
// Bincode (Binary Serialization)
// ============================================================================

fn bench_d64_serialize_bincode(c: &mut Criterion) {
    c.bench_function("d64_serialize_bincode", |b| {
        let d = D64::from_str("123.456789").unwrap();
        b.iter(|| black_box(bincode::serialize(&black_box(d)).unwrap()));
    });
}

fn bench_d64_deserialize_bincode(c: &mut Criterion) {
    c.bench_function("d64_deserialize_bincode", |b| {
        let d = D64::from_str("123.456789").unwrap();
        let bytes = bincode::serialize(&d).unwrap();
        b.iter(|| black_box(bincode::deserialize::<D64>(black_box(&bytes)).unwrap()));
    });
}

fn bench_d96_serialize_bincode(c: &mut Criterion) {
    c.bench_function("d96_serialize_bincode", |b| {
        let d = D96::from_str("123.456789").unwrap();
        b.iter(|| black_box(bincode::serialize(&black_box(d)).unwrap()));
    });
}

fn bench_d96_deserialize_bincode(c: &mut Criterion) {
    c.bench_function("d96_deserialize_bincode", |b| {
        let d = D96::from_str("123.456789").unwrap();
        let bytes = bincode::serialize(&d).unwrap();
        b.iter(|| black_box(bincode::deserialize::<D96>(black_box(&bytes)).unwrap()));
    });
}

fn bench_rust_decimal_serialize_bincode(c: &mut Criterion) {
    c.bench_function("rust_decimal_serialize_bincode", |b| {
        let d = Decimal::from_str_exact("123.456789").unwrap();
        b.iter(|| black_box(bincode::serialize(&black_box(d)).unwrap()));
    });
}

fn bench_rust_decimal_deserialize_bincode(c: &mut Criterion) {
    c.bench_function("rust_decimal_deserialize_bincode", |b| {
        let d = Decimal::from_str_exact("123.456789").unwrap();
        let bytes = bincode::serialize(&d).unwrap();
        b.iter(|| black_box(bincode::deserialize::<Decimal>(black_box(&bytes)).unwrap()));
    });
}

criterion_group!(
    benches,
    bench_d64_serialize_json,
    bench_d64_deserialize_json,
    bench_d64_roundtrip_json,
    bench_d96_serialize_json,
    bench_d96_deserialize_json,
    bench_d96_roundtrip_json,
    bench_rust_decimal_serialize_json,
    bench_rust_decimal_deserialize_json,
    bench_rust_decimal_roundtrip_json,
    bench_d64_struct_serialize_json,
    bench_d64_struct_deserialize_json,
    bench_d96_struct_serialize_json,
    bench_d96_struct_deserialize_json,
    bench_rust_decimal_struct_serialize_json,
    bench_rust_decimal_struct_deserialize_json,
    bench_d64_serialize_bincode,
    bench_d64_deserialize_bincode,
    bench_d96_serialize_bincode,
    bench_d96_deserialize_bincode,
    bench_rust_decimal_serialize_bincode,
    bench_rust_decimal_deserialize_bincode,
);

criterion_main!(benches);
