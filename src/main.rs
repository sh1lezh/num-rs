use num_rs::{self, nr_arange, nr_reshape_new};

fn main() {
    let arr = nr_reshape_new(&nr_arange(1.0, 5.0, 1.0), &[2, 2], 2);
    arr.nr_print_info();
    println!("Indices: {:?}", arr.idxs.indices);
}