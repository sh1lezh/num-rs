# Num-RS

A lightweight Rust library for multidimensional array operations, inspired by Python's NumPy. `num-rs` provides a foundation for numerical computing with support for array creation, reshaping, element-wise operations, broadcasting, and matrix multiplication. It is designed to be safe, performant, and extensible, leveraging Rust's type system and zero-cost abstractions.

**Note**: This is a work-in-progress project. Current functionality covers basic array operations, with plans to add more features like generic types, advanced indexing, and optimized linear algebra.

## Features

- **Multidimensional Arrays**: Create and manipulate n-dimensional arrays with flexible shapes and memory layouts (C-order).
- **Array Creation**:
  - `nr_arange`: Generate arrays with a range of values (like `np.arange`).
  - `nr_random`: Create arrays with random values (like `np.random.uniform`).
  - `nr_create`: Initialize arrays with zeros for a given shape.
- **Array Operations**:
  - `nr_reshape_new`: Reshape arrays to new dimensions (like `np.reshape`).
  - `nr_add` and `nr_mul`: Element-wise addition and multiplication with broadcasting (like `np.add`, `np.multiply`).
  - `nr_matmul`: Matrix multiplication for n-dimensional arrays (like `np.matmul`).
- **Display and Debugging**:
  - `nr_show`: Pretty-print arrays in a NumPy-like format.
  - `nr_print_info`: Display metadata (shape, strides, memory layout).
- **Performance**: Uses `rayon` for parallelized operations in `nr_add`.
- **Safety**: Built with Rust’s memory safety guarantees, using `Vec` for dynamic memory management.

## Installation

### Prerequisites
- [Rust](https://www.rust-lang.org/tools/install) (edition 2021 or later)
- [Cargo](https://doc.rust-lang.org/cargo/) for dependency management

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
- `rand = "0.9.2"`: For random number generation in `nr_random`.
- `rayon = "1.10.0"`: For parallelized operations in `nr_add`.

## Usage

Below are examples demonstrating `num-rs` functionality, with NumPy equivalents for comparison.

### Creating and Reshaping Arrays
```rust
use num_rs::{nr_arange, nr_reshape_new, nr_show};

fn main() {
    let arr = nr_arange(1.0, 7.0, 1.0); // [1, 2, 3, 4, 5, 6]
    let reshaped = nr_reshape_new(&arr, &[2, 3], 2); // [[1, 2, 3], [4, 5, 6]]
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
    let a = nr_reshape_new(&nr_arange(1.0, 4.0, 1.0), &[1, 3], 2); // [[1, 2, 3]]
    let b = nr_reshape_new(&nr_arange(1.0, 3.0, 1.0), &[2, 1], 2); // [[1], [2]]
    let c = nr_add(&a, &b); // [[2, 3, 4], [3, 4, 5]]
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
    let a = nr_reshape_new(&nr_arange(1.0, 5.0, 1.0), &[2, 2], 2); // [[1, 2], [3, 4]]
    let b = nr_reshape_new(&nr_arange(5.0, 9.0, 1.0), &[2, 2], 2); // [[5, 6], [7, 8]]
    let c = nr_matmul(&a, &b); // [[19, 22], [43, 50]]
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
    let arr = nr_random(&[2, 3], 2); // Random 2x3 array
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
├── src/
│   ├── lib.rs       # Library entry point, re-exports public APIs
│   ├── array.rs     # Array struct and core functionality
│   ├── ops.rs       # Array operations (arange, add, mul, matmul)
│   ├── util.rs      # Utility functions (random number generation)
│   ├── main.rs      # Example usage
├── README.md        # Project documentation
```

## Testing
Run tests to verify functionality:
```bash
cargo test
```
Current tests cover:
- `nr_arange`: Verifies array creation with a range.
- `nr_add`: Tests element-wise addition.
- `nr_reshape_new`: Ensures reshaping preserves data.

## Roadmap
`num-rs` is a work-in-progress aiming to approach the functionality of Rust’s `ndarray` or NumPy. Planned features include:
- **Generic Types**: Support for `f64`, `i32`, etc., using Rust generics.
- **Advanced Indexing**: NumPy-like slicing (e.g., `arr[1:3, :, 2]`), boolean indexing.
- **More Operations**: Statistical functions (`sum`, `mean`, `max`), trigonometric functions (`sin`, `cos`).
- **Shape Manipulation**: `transpose`, `expand_dims`, `squeeze`, `concatenate`.
- **Performance Optimizations**: SIMD, BLAS/LAPACK integration, zero-copy views.
- **Python Bindings**: Integration with `pyo3` for Python interoperability.
- **Error Handling**: Replace `panic!` with `Result` for robust error management.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please include tests for new features and update this README as needed.

## Learning Resources
- **Rust**: [The Rust Programming Language Book](https://doc.rust-lang.org/book/)
- **NumPy**: [NumPy Documentation](https://numpy.org/doc/stable/)
- **ndarray**: [Rust ndarray Crate](https://crates.io/crates/ndarray)

## License
This project is licensed under the MIT License. Havent updated the license yet.

## Contact
For questions or suggestions, open an issue or contact the maintainer at [contachsh1lezh@gmail.com] 
