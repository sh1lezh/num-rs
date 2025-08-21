#![allow(unused)]
use criterion::{criterion_group, criterion_main, Criterion};
use num_rs::{nr_arange, nr_reshape_new, nr_random, nr_add, nr_matmul, nr_mul, Array};
use std::hint::black_box;

// --- Benchmark for f32 operations ---
fn benchmarks_num_rs_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("num-rs-f32");
    group.warm_up_time(std::time::Duration::from_secs(3));
    group.measurement_time(std::time::Duration::from_secs(5));
    group.sample_size(100);

    // nr_arange
    group.bench_function("nr_arange_large_f32", |b| {
        b.iter(|| nr_arange(black_box(1.0f32), black_box(100000.0), black_box(1.0)));
    });

    // nr_random
    group.bench_function("nr_random_large_f32", |b| {
        b.iter(|| nr_random::<f32>(black_box(&[1000, 1000]), black_box(2)));
    });

    // nr_reshape_new
    let arr_large_reshape = nr_arange(1.0f32, 10001.0, 1.0);
    group.bench_function("nr_reshape_new_large_f32", |b| {
        b.iter(|| nr_reshape_new(black_box(&arr_large_reshape), black_box(&[100, 100]), black_box(2)));
    });

    // nr_add
    let arr1_large_add = nr_random::<f32>(&[1000, 1000], 2);
    let arr2_large_add = nr_random::<f32>(&[1000, 1000], 2);
    group.bench_function("nr_add_large_f32", |b| {
        b.iter(|| nr_add(black_box(&arr1_large_add), black_box(&arr2_large_add)));
    });

    // nr_mul
    let arr1_large_mul = nr_random::<f32>(&[1000, 1000], 2);
    let arr2_large_mul = nr_random::<f32>(&[1000, 1000], 2);
    group.bench_function("nr_mul_large_f32", |b| {
        b.iter(|| nr_mul(black_box(&arr1_large_mul), black_box(&arr2_large_mul)));
    });

    // nr_matmul
    let mat1_large_matmul = nr_random::<f32>(&[200, 300], 2);
    let mat2_large_matmul = nr_random::<f32>(&[300, 200], 2);
    group.bench_function("nr_matmul_large_f32", |b| {
        b.iter(|| nr_matmul(black_box(&mat1_large_matmul), black_box(&mat2_large_matmul)));
    });

    group.finish();
}

// --- Benchmark for f64 operations ---
fn benchmarks_num_rs_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("num-rs-f64");
    group.warm_up_time(std::time::Duration::from_secs(3));
    group.measurement_time(std::time::Duration::from_secs(5));
    group.sample_size(100);

    // nr_arange
    group.bench_function("nr_arange_large_f64", |b| {
        b.iter(|| nr_arange(black_box(1.0f64), black_box(100000.0), black_box(1.0)));
    });

    // nr_random
    group.bench_function("nr_random_large_f64", |b| {
        b.iter(|| nr_random::<f64>(black_box(&[1000, 1000]), black_box(2)));
    });

    // nr_reshape_new
    let arr_large_reshape = nr_arange(1.0f64, 10001.0, 1.0);
    group.bench_function("nr_reshape_new_large_f64", |b| {
        b.iter(|| nr_reshape_new(black_box(&arr_large_reshape), black_box(&[100, 100]), black_box(2)));
    });

    // nr_add
    let arr1_large_add = nr_random::<f64>(&[1000, 1000], 2);
    let arr2_large_add = nr_random::<f64>(&[1000, 1000], 2);
    group.bench_function("nr_add_large_f64", |b| {
        b.iter(|| nr_add(black_box(&arr1_large_add), black_box(&arr2_large_add)));
    });

    // nr_mul
    let arr1_large_mul = nr_random::<f64>(&[1000, 1000], 2);
    let arr2_large_mul = nr_random::<f64>(&[1000, 1000], 2);
    group.bench_function("nr_mul_large_f64", |b| {
        b.iter(|| nr_mul(black_box(&arr1_large_mul), black_box(&arr2_large_mul)));
    });

    // nr_matmul
    let mat1_large_matmul = nr_random::<f64>(&[200, 300], 2);
    let mat2_large_matmul = nr_random::<f64>(&[300, 200], 2);
    group.bench_function("nr_matmul_large_f64", |b| {
        b.iter(|| nr_matmul(black_box(&mat1_large_matmul), black_box(&mat2_large_matmul)));
    });

    group.finish();
}

criterion_group!(benches, benchmarks_num_rs_f32, benchmarks_num_rs_f64);
criterion_main!(benches);