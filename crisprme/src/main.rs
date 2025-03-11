#[macro_use] extern crate shrinkwraprs;

use std::{collections::HashSet, time::Instant};
use rand::Rng;

use crisprme_cuda::{autotune, kernel, Config, KResult};

mod tree;
mod utils;

use tree::Tree;

fn memory_of_bucket(levels: &[u32], query_len: usize, bucket_len: usize) -> u32 {
    let mut result = 0;
    for l in levels {
        result += (*l as usize) * query_len;
    }
    return (result + bucket_len * 2) as u32;
}

const REF_SIZE: usize = 64;
const ANCHOR_LEN: usize = 4;

/// Example usage of kernels
fn main() -> KResult<()> {

    // Available nucleotides
    let nucleotides: &[u8] = b"ACTG";

    // Create random reference sequence
    let reference = utils::generate_test_sequence(REF_SIZE, nucleotides, 3666);
    let reference_windows = utils::split_windows(&reference, ANCHOR_LEN);

    // Iterate over all windows of size ANCHOR_LEN and insert into tree
    let mut tree: Tree<4> = Tree::new(ANCHOR_LEN);
    for seq in reference_windows {
        tree.insert(seq);
    }

    tree.print_sequences();

    // Packed anchor
    let packed_tree = tree.pack();
    println!("----------------------");
    packed_tree.print();

    // Print the all split packed trees
    let split_packed_tree = packed_tree.split_at_width::<u8>(3);
    //for split_tree in &split_packed_tree {
    //    println!("----------------------");
    //    split_tree.print();
    //}

    // How many unique sequences are stored?
    println!("Span of the tree: {}", tree.span());

    return Ok(());


    let gpu = crisprme_cuda::initialize()?;

    // The sequence to test
    let query = vec![1, 3, 2];
    // The compressed bucket of prefix-tries
    let bucket = vec![
        1,      // layer: 0
        3, 1,   // layer: 1
        3, 5, 7 // layer: 2
    ];
    // The parents at each layer
    let parents = vec![
        0,      // layer: 0
        0, 1,   // layer: 1
        0, 1, 1 // layer: 2
    ];
    // The size of each layer
    let levels = vec![1, 2, 3];
    // Cumsum of layer sizes
    let levels_cumsum = vec![0, 1, 3];
    // Offset of DP tables
    let tables = vec![0, 3 * 1, 3 * 3];

    let mut config = Config::for_num_elems(100);
    config.shared_mem_bytes = memory_of_bucket(&levels, query.len(), bucket.len());
    println!("shared_mem_bytes: {}", config.shared_mem_bytes);
    
    // Launch work on GPU and get result
    kernel::mine_global_aligment(&gpu, config,
        query,
        bucket,
        parents, 
        levels,
        levels_cumsum,
        tables
    )?;

    Ok(())
}
