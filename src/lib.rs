mod array;
mod ops;
mod util;

pub use array::{Array, ArrayIndices, LinearIndices};
pub use ops::{nr_add, nr_arange, nr_mul, nr_random, nr_reshape_new, nr_matmul};

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_arrange() {
        let arr = nr_arange(1.0, 7.0, 1.0);
        assert_eq!(arr.shape, vec![6]);
        assert_eq!(arr.data, vec![1.0, 2.0, 3.0, 4.0 , 5.0, 6.0]);
    }

    #[test]
    fn test_add() {
        let a = nr_arange(1.0, 7.0, 1.0);
        let b = nr_arange(1.0, 7.0, 1.0);
        let c = nr_add(&a, &b);
        assert_eq!(c.data, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    }

    #[test] 
    fn test_reshape() {
        let a = nr_arange(1.0, 7.0, 1.0);
        let b = nr_reshape_new(&a, &[2, 3], 2);
        assert_eq!(b.shape, vec![2, 3]);
        assert_eq!(b.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}