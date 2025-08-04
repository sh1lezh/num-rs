#![allow(unused)]
use crate::array::{Array, ArrayIndices, LinearIndices};
use crate::util::get_random_float;
use rayon::prelude::*;

pub fn nr_arange(start: f32, end: f32, step: f32) -> Array {
    if start >= end {
        panic!("Start value must be less than end value");
    }
    if step <= 0.0 {
        panic!("Step value must be positive");
    }

    let len = ((end - start) / step).ceil() as i32;
    let shape = vec![len];
    let mut arr = Array::nr_create(&shape, 1);
    for i in 0..len as usize {
        arr.data[i] = start + i as f32 * step;
    }
    arr
}

pub fn nr_random(shape: &[i32], ndim: usize) -> Array {
    let mut arr = Array::nr_create(shape, ndim);
    for i in 0..arr.totalsize {
        arr.data[i] = get_random_float(0.0, 1.0);
    }
    arr
}

pub fn nr_reshape_new(arr: &Array, shape: &[i32], ndim: usize) -> Array {
    let new_totalsize = shape.iter().product::<i32>() as usize;
    if new_totalsize != arr.totalsize {
        panic!("Cannot reshape due to invalid shape");
    }
    let mut new_arr = Array::nr_create(shape, ndim);
    new_arr.data.copy_from_slice(&arr.data);
    new_arr
}

fn broadcast_final_shape(a: &Array, b: &Array) -> Option<Vec<i32>> {
    if a.shape == b.shape {
        return Some(a.shape.clone());
    }

    let res_ndim = a.ndim.max(b.ndim);
    let mut a_shape = vec![1; res_ndim];
    let mut b_shape = vec![1; res_ndim];

    for i in 0..a.ndim {
        a_shape[res_ndim - a.ndim + i] = a.shape[i];
    }
    for i in 0..b.ndim {
        b_shape[res_ndim - b.ndim + i] = b.shape[i];
    }

    let mut res_shape = vec![0; res_ndim];
    for i in 0..res_ndim {
        if a_shape[i] == 1 || b_shape[i] == 1 || a_shape[i] == b_shape[i] {
            res_shape[i] = a_shape[i].max(b_shape[i]);
        } else {
            return None;
        }
    }
    Some(res_shape)
}

fn broadcast_array(arr: &Array, shape: &[i32], ndim: usize) -> Array {
    let mut res = Array::nr_create(shape, ndim);
    let n_prepend = ndim - arr.ndim;

    for i in 0..res.totalsize {
        let mut src_idx = 0;
        for dim in 0..arr.ndim {
            if arr.shape[dim] > 1 {
                src_idx += (res.idxs.indices[i][n_prepend + dim] % arr.shape[dim]) * arr.strides[dim];
            }
        }
        res.data[res.lidxs.indices[i]] = arr.data[src_idx as usize / arr.itemsize];
    }
    res
}

pub fn nr_add(a: &Array, b: &Array) -> Array {
    if a.shape == b.shape {
        let mut res = Array::nr_create(&a.shape, a.ndim);
        res.data.par_iter_mut().enumerate().for_each(|(i, val)| {
            *val = a.data[a.lidxs.indices[i]] + b.data[b.lidxs.indices[i]];
        });
        return res;
    }

    let res_shape = broadcast_final_shape(a, b).expect("Cannot add arrays of non-broadcastable shapes");
    let res_ndim = res_shape.len();
    let a_final = broadcast_array(a, &res_shape, res_ndim);
    let b_final = broadcast_array(b, &res_shape, res_ndim);
    let mut res = Array::nr_create(&res_shape, res_ndim);
    for i in 0..res.totalsize {
        res.data[i] = a_final.data[i] + b_final.data[i];
    }
    res
}

pub fn nr_mul(a: &Array, b: &Array) -> Array {
    if a.shape == b.shape {
        let mut res = Array::nr_create(&a.shape, a.ndim);
        for i in 0..a.totalsize {
            res.data[i] = a.data[a.lidxs.indices[i]] * b.data[b.lidxs.indices[i]];
        }
        return res;
    }

    let res_shape = broadcast_final_shape(a, b).expect("Cannot multiply arrays of non-broadcastable shapes");
    let res_ndim = res_shape.len();
    let a_final = broadcast_array(a, &res_shape, res_ndim);
    let b_final = broadcast_array(b, &res_shape, res_ndim);
    let mut res = Array::nr_create(&res_shape, res_ndim);
    for i in 0..res.totalsize {
        res.data[i] = a_final.data[i] * b_final.data[i];
    }
    res
}

pub fn nr_matmul(a: &Array, b: &Array) -> Array {
    if a.ndim < 2 || b.ndim < 2 {
        panic!("Both arrays must have atleast 2 dimensions for matmul");
    }
    if a.shape[a.ndim - 1] != b.shape[b.ndim - 2] {
        panic!("Last dimension of first array must match second-last dimension of second array");
    }

    let result_ndim = a.ndim.max(b.ndim);
    let mut result_shape = vec![0; result_ndim];
    
    for i in 0..result_ndim - 2 {
        result_shape[i] = if i < a.ndim - 2 {
            a.shape[i]
        } else {
            1
        };

        result_shape[i] = if i < b.ndim - 2 {
            result_shape[i].max(b.shape[i])
        } else {
            result_shape[i]
        };
    }

    result_shape[result_ndim - 2] = a.shape[a.ndim - 2];
    result_shape[result_ndim - 1] = b.shape[b.ndim - 1];

    let mut result = Array::nr_create(&result_shape, result_ndim);
    let m = a.shape[a.ndim - 2] as usize;
    let n = a.shape[a.ndim - 1] as usize;
    let p = b.shape[b.ndim - 1] as usize;

    let idxs = Array::create_array_indices(&result_shape[..result_ndim -2], result_ndim - 2);
    for idx in 0..idxs.count {
        let nd_index = &idxs.indices[idx];
        for i in 0..m {
            for j in 0..p {
                let mut sum = 0.0;
                for k in 0..n {
                    let mut a_index1d = 0;
                    let mut b_index1d = 0;
                    for d in 0..a.ndim - 2 {
                        a_index1d += (nd_index[d] * a.strides[d]) as usize;
                    }
                    for d in 0..b.ndim - 2 {
                        b_index1d += (nd_index[d] * a.strides[d]) as usize;
                    }
                    a_index1d += (i * a.strides[a.ndim - 2] as usize + k * a.strides[a.ndim -1] as usize) / a.itemsize;
                    b_index1d += (k * b.strides[b.ndim - 2] as usize + j * b.strides[b.ndim - 1] as usize) / b.itemsize;
                    sum += a.data[a_index1d] * b.data[b_index1d];
                }
                let mut r_index1d = 0;
                for d in 0..result.ndim - 2 {
                    r_index1d += (nd_index[d] * result.strides[d]) as usize;
                }
                r_index1d += (i * result.strides[result.ndim - 2] as usize + j * result.strides[result.ndim - 1] as usize) / result.itemsize;
                result.data[r_index1d] = sum;
            }
        }
    }
    result
}