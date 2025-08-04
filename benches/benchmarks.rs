use criterion::{criterion_group, criterion_main, Criterion};
use num_rs::{nr_arange, nr_reshape_new, nr_random, nr_add, nr_matmul, nr_mul};
use std::hint::black_box;

fn benchmarks_num_rs(c: &mut Criterion) {
    let mut group = c.benchmark_group("num-rs");
    
    // Configure the benchmark group for reliable measurements
    group.warm_up_time(std::time::Duration::from_secs(3));
    group.measurement_time(std::time::Duration::from_secs(5));
    group.sample_size(100);

    // Test Case: Array Creation with nr_arange (small, medium, large)
    group.bench_function("nr_arange_small_1_to_7", |b| {
        b.iter(|| nr_arange(black_box(1.0), black_box(7.0), black_box(1.0)));
    });

    group.bench_function("nr_arange_medium_1_to_1000", |b| {
        b.iter(|| nr_arange(black_box(1.0), black_box(1000.0), black_box(1.0)));
    });

    group.bench_function("nr_arange_large_1_to_100000", |b| {
        b.iter(|| nr_arange(black_box(1.0), black_box(100000.0), black_box(1.0)));
    });

    // Test Case: Random Array Creation with nr_random
    group.bench_function("nr_random_small_10x10", |b| {
        b.iter(|| nr_random(black_box(&[10, 10]), black_box(2)));
    });

    group.bench_function("nr_random_medium_100x100", |b| {
        b.iter(|| nr_random(black_box(&[100, 100]), black_box(2)));
    });

    group.bench_function("nr_random_large_1000x1000", |b| {
        b.iter(|| nr_random(black_box(&[1000, 1000]), black_box(2)));
    });

    // Test Case: Reshaping with nr_reshape_new
    let arr_small = nr_arange(1.0, 7.0, 1.0);
    let arr_medium = nr_arange(1.0, 1001.0, 1.0);
    let arr_large = nr_arange(1.0, 10001.0, 1.0);

    group.bench_function("nr_reshape_new_small_2x3", |b| {
        b.iter(|| nr_reshape_new(black_box(&arr_small), black_box(&[2, 3]), black_box(2)));
    });

    group.bench_function("nr_reshape_new_medium_10x100", |b| {
        b.iter(|| nr_reshape_new(black_box(&arr_medium), black_box(&[10, 100]), black_box(2)));
    });

    group.bench_function("nr_reshape_new_large_100x100", |b| {
        b.iter(|| nr_reshape_new(black_box(&arr_large), black_box(&[100, 100]), black_box(2)));
    });

    // Test Case: Addition with nr_add
    let arr1_small = nr_arange(1.0, 7.0, 1.0);
    let arr2_small = nr_arange(1.0, 7.0, 1.0);
    let arr1_medium = nr_random(&[100, 100], 2);
    let arr2_medium = nr_random(&[100, 100], 2);
    let arr1_large = nr_random(&[1000, 1000], 2);
    let arr2_large = nr_random(&[1000, 1000], 2);

    group.bench_function("nr_add_small_6", |b| {
        b.iter(|| nr_add(black_box(&arr1_small), black_box(&arr2_small)));
    });

    group.bench_function("nr_add_medium_100x100", |b| {
        b.iter(|| nr_add(black_box(&arr1_medium), black_box(&arr2_medium)));
    });

    group.bench_function("nr_add_large_1000x1000", |b| {
        b.iter(|| nr_add(black_box(&arr1_large), black_box(&arr2_large)));
    });

    // Test Case: Multiplication with nr_mul
    group.bench_function("nr_mul_small_6", |b| {
        b.iter(|| nr_mul(black_box(&arr1_small), black_box(&arr2_small)));
    });

    group.bench_function("nr_mul_medium_100x100", |b| {
        b.iter(|| nr_mul(black_box(&arr1_medium), black_box(&arr2_medium)));
    });

    group.bench_function("nr_mul_large_1000x1000", |b| {
        b.iter(|| nr_mul(black_box(&arr1_large), black_box(&arr2_large)));
    });

    // Test Case: Matrix Multiplication with nr_matmul
    let mat1_small = nr_reshape_new(&nr_arange(1.0, 7.0, 1.0), &[2, 3], 2);
    let mat2_small = nr_reshape_new(&nr_arange(1.0, 7.0, 1.0), &[3, 2], 2);
    let mat1_medium = nr_random(&[50, 100], 2);
    let mat2_medium = nr_random(&[100, 50], 2);
    let mat1_large = nr_random(&[200, 300], 2);
    let mat2_large = nr_random(&[300, 200], 2);

    group.bench_function("nr_matmul_small_2x3_3x2", |b| {
        b.iter(|| nr_matmul(black_box(&mat1_small), black_box(&mat2_small)));
    });

    group.bench_function("nr_matmul_medium_50x100_100x50", |b| {
        b.iter(|| nr_matmul(black_box(&mat1_medium), black_box(&mat2_medium)));
    });

    group.bench_function("nr_matmul_large_200x300_300x200", |b| {
        b.iter(|| nr_matmul(black_box(&mat1_large), black_box(&mat2_large)));
    });

    group.finish();
}

criterion_group!(benches, benchmarks_num_rs);
criterion_main!(benches);