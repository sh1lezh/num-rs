# Num-RS

`num-rs` is a lightweight Rust library for multidimensional array operations, inspired by Python's NumPy. It provides a robust foundation for numerical computing, featuring array creation, reshaping, element-wise operations, broadcasting, and matrix multiplication. Designed for safety, performance, and extensibility, `num-rs` leverages Rust's type system and zero-cost abstractions to deliver efficient numerical computations.

**Note**: This project is under active development. Current features focus on core array operations, with plans to expand to advanced indexing and optimized linear algebra routines.

## Features

- **Multidimensional Arrays**: Create and manipulate n-dimensional arrays with flexible shapes and C-order memory layout.
- **Array Creation**:
  - `nr_arange`: Generate arrays with a sequence of values (similar to `np.arange`).
  - `nr_random`: Create arrays with random values (similar to `np.random.uniform`).
  - `nr_create`: Initialize arrays with zeros for a specified shape.
- **Array Operations**:
  - `nr_reshape_new`: Reshape arrays to new dimensions (similar to `np.reshape`).
  - `nr_add` and `nr_mul`: Perform element-wise addition and multiplication with broadcasting support (similar to `np.add`, `np.multiply`).
  - `nr_matmul`: Execute matrix multiplication for n-dimensional arrays (similar to `np.matmul`).
- **Display and Debugging**:
  - `nr_show`: Pretty-print arrays in a NumPy-like format for easy visualization.
  - `nr_print_info`: Display array metadata, including shape, strides, and memory layout.
- **Performance**: Utilizes `rayon` for parallelized operations in `nr_add`, ensuring efficient computation on multi-core systems.
- **Safety**: Built with Rust’s memory safety guarantees, using `Vec` for dynamic memory management.
- **Generic Types**: Supports multiple data types (`f32`, `f64`, `i32`, etc.) using Rust generics for flexible numerical computations.
- **Benchmarking**: Includes a `benchmarks.rs` file with `criterion` for performance testing of key operations.

## Installation

### Prerequisites

- Rust (edition 2021 or later)
- Cargo for dependency management
- Optional: Criterion for running benchmarks

### Adding `num-rs` to Your Project

1. Clone the repository:

   ```bash
   git clone https://github.com/sh1lezh/num-rs.git
   cd num-rs
   ```

2. Add `num-rs` as a dependency in your `Cargo.toml`:

   ```toml
   [dependencies]
   num-rs = { path = "./num-rs" }
   ```

3. Build and run:

   ```bash
   cargo build
   cargo test
   ```

### Dependencies

- `rand = "0.9.2"`: Enables random number generation for `nr_random`.
- `rayon = "1.10.0"`: Provides parallel processing for `nr_add`.
- `num-traits = "0.2"`: Enables generic numerical operations across different data types.
- Optional: `criterion = "0.7.0"` for running benchmarks (used in `benchmarks.rs`).

## Usage

Below are examples demonstrating key `num-rs` functionality, with NumPy equivalents for reference.

### Creating and Reshaping Arrays

```rust
use num_rs::{nr_arange, nr_reshape_new, nr_show};

fn main() {
    let arr = nr_arange(1.0, 7.0, 1.0); // Creates [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    let reshaped = nr_reshape_new(&arr, &[2, 3], 2); // Reshapes to [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    reshaped.nr_show();
}
```

**NumPy Equivalent**:

```python
import numpy as np
arr = np.arange(1, 7)
reshaped = arr.reshape(2, 3)
print(reshaped)
```

**Output**:

```
1.000 2.000 3.000
4.000 5.000 6.000
```

### Element-wise Addition with Broadcasting

```rust
use num_rs::{nr_arange, nr_reshape_new, nr_add, nr_show};

fn main() {
    let a = nr_reshape_new(&nr_arange(1.0, 4.0, 1.0), &[1, 3], 2); // [[1.0, 2.0, 3.0]]
    let b = nr_reshape_new(&nr_arange(1.0, 3.0, 1.0), &[2, 1], 2); // [[1.0], [2.0]]
    let c = nr_add(&a, &b); // [[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]
    c.nr_show();
}
```

**NumPy Equivalent**:

```python
a = np.arange(1, 4).reshape(1, 3)
b = np.arange(1, 3).reshape(2, 1)
c = a + b
print(c)
```

**Output**:

```
2.000 3.000 4.000
3.000 4.000 5.000
```

### Matrix Multiplication

```rust
use num_rs::{nr_arange, nr_reshape_new, nr_matmul, nr_show};

fn main() {
    let a = nr_reshape_new(&nr_arange(1.0, 5.0, 1.0), &[2, 2], 2); // [[1.0, 2.0], [3.0, 4.0]]
    let b = nr_reshape_new(&nr_arange(5.0, 9.0, 1.0), &[2, 2], 2); // [[5.0, 6.0], [7.0, 8.0]]
    let c = nr_matmul(&a, &b); // [[19.0, 22.0], [43.0, 50.0]]
    c.nr_show();
}
```

**NumPy Equivalent**:

```python
a = np.arange(1, 5).reshape(2, 2)
b = np.arange(5, 9).reshape(2, 2)
c = a @ b
print(c)
```

**Output**:

```
19.000 22.000
43.000 50.000
```

### Random Array Creation

```rust
use num_rs::{nr_random, nr_show};

fn main() {
    let arr = nr_random(&[2, 3], 2); // Creates a 2x3 array with random values in [0.0, 1.0)
    arr.nr_show();
}
```

**NumPy Equivalent**:

```python
arr = np.random.uniform(0, 1, (2, 3))
print(arr)
```

## Project Structure

```
num-rs/
├── Cargo.toml       # Project configuration and dependencies
├── benches/
│   ├── benchmarks.rs    # Performance benchmarks using Criterion
├── src/
│   ├── array.rs     # Core Array struct and indexing logic
│   ├── lib.rs       # Library entry point, re-exports public APIs
│   ├── main.rs      # Example usage of the library
│   ├── ops.rs       # Array operations (arange, add, mul, matmul, random)
│   ├── util.rs      # Utility functions (random number generation)
├── README.md        # Project documentation
```

## Testing

Run unit tests to verify functionality:

```bash
cargo test
```

Tests cover:

- `nr_arange`: Validates array creation with a range of values.
- `nr_add`: Verifies element-wise addition.
- `nr_reshape_new`: Ensures reshaping preserves data integrity.

## Benchmarking

Run performance benchmarks to evaluate operation efficiency:

```bash
cargo bench
```

Benchmarks include:

- Array creation (`nr_arange`) for small, medium, and large sizes.
- Random array generation (`nr_random`) for 10x10, 100x100, and 1000x1000 arrays.
- Reshaping (`nr_reshape_new`) for various array sizes.
- Element-wise operations (`nr_add`, `nr_mul`) for small, medium, and large arrays.
- Matrix multiplication (`nr_matmul`) for different matrix dimensions.

## Roadmap

`num-rs` aims to evolve into a comprehensive numerical computing library, approaching the functionality of Rust’s `ndarray` or NumPy. Planned enhancements include:

- **Generic Types**: *Completed* - Support for multiple data types (`f32`, `f64`, `i32`, etc.) using Rust generics.
- **Advanced Indexing**: Implement NumPy-like slicing (e.g., `arr[1:3, :, 2]`) and boolean indexing.
- **Expanded Operations**: Add statistical functions (`sum`, `mean`, `max`), trigonometric functions (`sin`, `cos`), and more.
- **Shape Manipulation**: Support `transpose`, `expand_dims`, `squeeze`, and `concatenate`.
- **Performance Optimizations**: Integrate SIMD, BLAS/LAPACK, and zero-copy views for enhanced performance.
- **Python Interoperability**: Add `pyo3` bindings for seamless integration with Python.
- **Robust Error Handling**: Transition from `panic!` to `Result` for better error management.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request with a clear description of your changes.

Please include tests for new features and update the README as necessary.

## Learning Resources

- **Rust**: The Rust Programming Language Book
- **NumPy**: NumPy Documentation
- **ndarray**: Rust ndarray Crate

## License

This project is licensed under the MIT License. The license file is pending finalization.

## Contact

For questions, suggestions, or issues, please open an issue on GitHub or contact the maintainer at [contachsh1lezh@gmail.com].