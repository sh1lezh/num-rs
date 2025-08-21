use num_rs::{nr_add, nr_arange, nr_matmul, nr_mul, nr_random, nr_reshape_new, Array};

fn main() {
    println!("--- Welcome to the num-rs demonstration! ---");

    // --- 1. Array Creation ---
    println!("\n## 1. Array Creation ##");

    // Using nr_arange with f32
    println!("\nCreating a 1D array of f32 from 1.0 to 8.0:");
    let arr_f32: Array<f32> = nr_arange(1.0, 9.0, 1.0);
    arr_f32.nr_show();

    // Using nr_arange with i32
    println!("\nCreating a 1D array of i32 from -3 to 3:");
    let arr_i32: Array<i32> = nr_arange(-3, 4, 1);
    arr_i32.nr_show();

    // Using nr_random with f64
    println!("\nCreating a 2x4 array with random f64 values:");
    let rand_arr: Array<f64> = nr_random(&[2, 4], 2);
    rand_arr.nr_show();

    // --- 2. Reshaping and Info ---
    println!("\n## 2. Reshaping and Inspecting an Array ##");
    println!("\nReshaping the f32 array into a 2x4 matrix:");
    let reshaped_arr = nr_reshape_new(&arr_f32, &[2, 4], 2);
    reshaped_arr.nr_show();
    println!("Array Info:");
    reshaped_arr.nr_print_info();

    // --- 3. Element-wise Operations ---
    println!("\n## 3. Element-wise Operations ##");

    // nr_add (element-wise addition)
    println!("\nAdding two 2x4 arrays:");
    let a_add = nr_reshape_new(&nr_arange(1, 9, 1), &[2, 4], 2);
    let b_add = nr_reshape_new(&nr_arange(10, 18, 1), &[2, 4], 2);
    let c_add = nr_add(&a_add, &b_add);
    println!("Array A:");
    a_add.nr_show();
    println!("Array B:");
    b_add.nr_show();
    println!("Result (A + B):");
    c_add.nr_show();

    // nr_add with broadcasting
    println!("\nAdding a 1x3 array to a 2x1 array (Broadcasting):");
    let a_broadcast = nr_reshape_new(&nr_arange(1.0, 4.0, 1.0), &[1, 3], 2); // Shape [1, 3]
    let b_broadcast = nr_reshape_new(&nr_arange(1.0, 3.0, 1.0), &[2, 1], 2); // Shape [2, 1]
    let c_broadcast = nr_add(&a_broadcast, &b_broadcast);
    println!("Array A (1x3):");
    a_broadcast.nr_show();
    println!("Array B (2x1):");
    b_broadcast.nr_show();
    println!("Result after broadcasting (2x3):");
    c_broadcast.nr_show();

    // nr_mul (element-wise multiplication)
    println!("\nMultiplying two 2x2 arrays:");
    let a_mul = nr_reshape_new(&nr_arange(1, 5, 1), &[2, 2], 2);
    let b_mul = nr_reshape_new(&nr_arange(5, 9, 1), &[2, 2], 2);
    let c_mul = nr_mul(&a_mul, &b_mul);
    println!("Array A:");
    a_mul.nr_show();
    println!("Array B:");
    b_mul.nr_show();
    println!("Result (A * B):");
    c_mul.nr_show();

    // --- 4. Matrix Multiplication ---
    println!("\n## 4. Matrix Multiplication (MatMul) ##");
    println!("\nMultiplying a 2x3 matrix with a 3x2 matrix:");
    let mat1: Array<f32> = nr_reshape_new(&nr_arange(1.0, 7.0, 1.0), &[2, 3], 2);
    let mat2: Array<f32> = nr_reshape_new(&nr_arange(7.0, 13.0, 1.0), &[3, 2], 2);
    let mat_result = nr_matmul(&mat1, &mat2);
    println!("Matrix 1 (2x3):");
    mat1.nr_show();
    println!("Matrix 2 (3x2):");
    mat2.nr_show();
    println!("Result of MatMul (2x2):");
    mat_result.nr_show();
}