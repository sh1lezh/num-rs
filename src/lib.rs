mod array;
mod ops;
mod util;

pub use array::{Array, ArrayIndices, LinearIndices};
pub use ops::{nr_arange, nr_random, nr_reshape_new, nr_mul, nr_add, nr_matmul};

#[cfg(test)]
mod test {
    use super::*;

    // --- nr_arange tests ---
    #[test]
    fn test_arange_f32() {
        let arr = nr_arange(1.0f32, 7.0, 1.0);
        assert_eq!(arr.shape, vec![6]);
        assert_eq!(arr.data, vec![1.0, 2.0, 3.0, 4.0 , 5.0, 6.0]);
    }

    #[test]
    fn test_arange_i32() {
        let arr = nr_arange(1i32, 7, 1);
        assert_eq!(arr.shape, vec![6]);
        assert_eq!(arr.data, vec![1, 2, 3, 4, 5, 6]);
    }

    // --- nr_reshape_new tests ---
    #[test]
    fn test_reshape() {
        let a = nr_arange(1.0f32, 7.0, 1.0);
        let b = nr_reshape_new(&a, &[2, 3], 2);
        assert_eq!(b.shape, vec![2, 3]);
        assert_eq!(b.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // --- nr_add tests ---
    #[test]
    fn test_add_f64() {
        let a = nr_arange(1.0f64, 7.0, 1.0);
        let b = nr_arange(1.0f64, 7.0, 1.0);
        let c = nr_add(&a, &b);
        assert_eq!(c.data, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_add_with_broadcasting() {
        let a = nr_reshape_new(&nr_arange(1.0f32, 4.0, 1.0), &[1, 3], 2); // Shape [1, 3]
        let b = nr_reshape_new(&nr_arange(1.0f32, 3.0, 1.0), &[2, 1], 2); // Shape [2, 1]
        let c = nr_add(&a, &b);
        assert_eq!(c.shape, vec![2, 3]);
        assert_eq!(c.data, vec![2.0, 3.0, 4.0, 3.0, 4.0, 5.0]);
    }

    // --- nr_mul tests ---
    #[test]
    fn test_mul_i32() {
        let a = nr_arange(1i32, 7, 1); // [1, 2, 3, 4, 5, 6]
        let b = nr_arange(1i32, 7, 1); // [1, 2, 3, 4, 5, 6]
        let c = nr_mul(&a, &b);
        assert_eq!(c.data, vec![1, 4, 9, 16, 25, 36]);
    }

    // --- nr_matmul tests ---
    #[test]
    fn test_matmul_f32() {
        let a = nr_reshape_new(&nr_arange(1.0f32, 5.0, 1.0), &[2, 2], 2); // [[1, 2], [3, 4]]
        let b = nr_reshape_new(&nr_arange(5.0f32, 9.0, 1.0), &[2, 2], 2); // [[5, 6], [7, 8]]
        let c = nr_matmul(&a, &b);
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

     #[test]
    fn test_matmul_i32() {
        let a = nr_reshape_new(&nr_arange(1i32, 5, 1), &[2, 2], 2); // [[1, 2], [3, 4]]
        let b = nr_reshape_new(&nr_arange(5i32, 9, 1), &[2, 2], 2); // [[5, 6], [7, 8]]
        let c = nr_matmul(&a, &b);
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![19, 22, 43, 50]);
    }

    // --- nr_random tests ---
    #[test]
    fn test_random() {
        let arr = nr_random::<f64>(&[5, 5], 2);
        assert_eq!(arr.shape, vec![5, 5]);
        assert_eq!(arr.totalsize, 25);
        // Check if values are within the expected range [0, 1)
        for val in arr.data {
            assert!(val >= 0.0 && val < 1.0);
        }
    }
}