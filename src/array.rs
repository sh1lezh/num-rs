#![allow(unused)]
use std::mem;
use num_traits::Zero;
use std::fmt::{Debug, Display};

// ArrayIndices for n-dimensional indexing
#[derive(Debug)]
pub struct ArrayIndices {
    pub indices: Vec<Vec<i32>>, // 2D vector for index combinations
    pub count: usize,           // Total number of index combinations
}

// LinearIndices for 1D equivalent of n-dimensional indices
#[derive(Debug)]
pub struct LinearIndices {
    pub indices: Vec<usize>, // 1D indices
    pub count: usize,        // Total number of indices
}

// Main Array struct
#[derive(Debug)]
pub struct Array<T> {
    pub data: Vec<T>,         // Data buffer
    pub shape: Vec<i32>,        // Shape of the array
    pub strides: Vec<i32>,      // Strides for each dimension
    pub backstrides: Vec<i32>,  // Backstrides for reverse traversal
    pub ndim: usize,            // Number of dimensions
    pub itemsize: usize,        // Size of one element (f32)
    pub totalsize: usize,       // Total number of elements
    pub idxs: ArrayIndices,     // n-dimensional indices
    pub lidxs: LinearIndices,   // Linear indices
    pub c_order: bool,          // C-contiguous flag
    pub f_order: bool,          // F-contiguous flag
}


impl<T> Array<T>
where
    T: Copy + Zero,
{
    // Create a new Array with given shape and number of dimensions
    pub fn nr_create(shape: &[i32], ndim: usize) -> Array<T> {
        if ndim == 0 || shape.len() != ndim {
            panic!("Cannot initialize Array with ndim {} or invalid shape", ndim);
        }

        let itemsize = std::mem::size_of::<T>();
        let mut totalsize = 1;
        for &dim in shape {
            if dim <= 0 {
                panic!("Shape dimensions must be positive");
            }
            totalsize *= dim as usize;
        }

        // Initialize data with zeros
        let data = vec![T::zero(); totalsize];  
        let shape = shape.to_vec();
        let mut strides = vec![0; ndim];
        let mut backstrides = vec![0; ndim];

        // Calculate strides (C-order by default)
        strides[ndim - 1] = itemsize as i32;
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        // Calculate backstrides
        for i in (0..ndim).rev() {
            backstrides[i] = -1 * strides[i] * (shape[i] - 1);
        }

        // Create indices
        let idxs = Array::<T>::create_array_indices(&shape, ndim);
        let lidxs = Array::<T>::create_linear_indices(&shape, &strides, itemsize, totalsize);

        // Set order flags
        let c_order = strides[ndim - 1] == itemsize as i32;
        let f_order = strides[0] == itemsize as i32;

        Array {
            data,
            shape,
            strides,
            backstrides,
            ndim,
            itemsize,
            totalsize,
            idxs,
            lidxs,
            c_order,
            f_order,
        }
    }

    // Create n-dimensional indices
    pub fn create_array_indices(shape: &[i32], ndim: usize) -> ArrayIndices {
        let count = shape.iter().product::<i32>() as usize;
        let mut indices = vec![vec![0; ndim]; count];
        let mut current_index = vec![0; ndim];

        for i in 0..count {
            for j in 0..ndim {
                indices[i][j] = current_index[j];
            }
            for j in (0..ndim).rev() {
                current_index[j] += 1;
                if current_index[j] < shape[j] {
                    break;
                }
                current_index[j] = 0;
            }
        }

        ArrayIndices { indices, count }
    }

    // Create linear indices
    fn create_linear_indices(shape: &[i32], strides: &[i32], itemsize: usize, totalsize: usize) -> LinearIndices {
        let mut indices = vec![0; totalsize];
        let idxs = Array::<T>::create_array_indices(shape, shape.len());
        for i in 0..totalsize {
            let mut idx = 0;
            for j in 0..shape.len() {
                idx += idxs.indices[i][j] * strides[j];
            }
            indices[i] = (idx / itemsize as i32) as usize;
        }
        LinearIndices { indices, count: totalsize }
    }

    // Print array information
    pub fn nr_print_info(&self) {
        println!("Shape: {:?}", self.shape);
        println!("Strides: {:?}", self.strides);
        println!("Array is C-contiguous? {}", self.c_order);
        println!("Array is F-contiguous? {}", self.f_order);
    }

}

impl<T: Display + Copy> Array<T> {
    pub fn nr_show(&self) {
        fn print_recursive<T: std::fmt::Display>(data_slice: &[T], shape: &[i32]) {
            // Base case: If the shape is 1D, print all elements separated by spaces.
            if shape.len() == 1 {
                for i in 0..shape[0] as usize {
                    print!("{} ", data_slice[i]);
                }
                println!();
                return;
            }

            // Recursive step:
            let remaining_shape = &shape[1..];
            // The size of each sub-block is the product of the remaining dimensions.
            let block_size = remaining_shape.iter().product::<i32>() as usize;

            for i in 0..shape[0] as usize {
                // Calculate the slice for the current sub-block.
                let start = i * block_size;
                let end = start + block_size;
                print_recursive(&data_slice[start..end], remaining_shape);
            }
            
            // Add a newline for visual separation between higher-dimensional blocks.
            if shape.len() > 2 {
                 println!();
            }
        }
        
        if !self.data.is_empty() {
            print_recursive(&self.data, &self.shape);
        }
    }
}