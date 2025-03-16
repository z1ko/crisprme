#[macro_use] extern crate shrinkwraprs;
use prettytable::{row, Cell, Row, Table};
use needleman::needleman::needleman;
use crisprme_cuda::{autotune, kernel, Config, KResult};

mod tree;
mod utils;

use tree::Tree;

const REF_SIZE: usize = 64;
const ANCHOR_LEN: usize = 20;

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
    //println!("----------------------");
    //packed_tree.print();

    // Print the all split packed trees
    let split_packed_tree = packed_tree.split_at_width::<u32>(30);
    //split_packed_tree[0].print();
    //for split_tree in &split_packed_tree {
    //    println!("----------------------");
    //    split_tree.print();
    //}

    // How many unique sequences are stored?
    println!("Span of the tree: {}", tree.span());
    //println!("----------------------");

    let QUERY = "ACACTTGAACACACACTGAA";
    assert_eq!(QUERY.len(), ANCHOR_LEN);
    
    // Query as u32
    let query: Vec<u32> = QUERY.chars()
        .map(|x| u32::try_from(x).unwrap())
        .collect();
    
    // Letters as u32
    let bucket: Vec<u32> = utils::linearize_memory(&split_packed_tree[0].layers).iter()
        .map(|l| u32::try_from(*l).unwrap())
        .collect();

    let parents = utils::linearize_memory(&split_packed_tree[0].offset);
    let levels = split_packed_tree[0].layer_sizes();
    let levels_cumsum = split_packed_tree[0].layer_sizes_cumsum();
    let tables = split_packed_tree[0].dp_table_offsets();

    let gpu = crisprme_cuda::initialize()?;

    let mut config = Config::for_num_elems(100);
    config.shared_mem_bytes = 48000; // 48K
    println!("shared_mem_bytes: {}", config.shared_mem_bytes);

    // Launch work on GPU and get result
    let (aligments, scores) = kernel::mine_global_aligment(&gpu, config,
        query,
        bucket,
        parents, 
        levels,
        levels_cumsum,
        tables
    )?;

    // Show extracted aligment

    let mut table_query = Table::new();
    table_query.add_row(row!["Query", QUERY]);
    table_query.printstd();

    let mut table = Table::new();
    table.add_row(row!["Nth.", "Reference", "Aligment", "Score", "Correct?", "Reference aligment"]);
    for i in 0..split_packed_tree[0].span() {
        
        let seq = split_packed_tree[0].sequence_at_leaf(i);
        let cigarx: String = aligments.iter().skip(i).step_by(split_packed_tree[0].span())
            .filter(|x| **x != 0)
            .map(|e| char::from_u32(*e).unwrap())
            .rev()
            .collect();

        // Is it correct?
        let ag = needleman(QUERY, &seq, 1, -1, -2, false, 'N');
        let correct = ag.score == scores[i];

        table.add_row(row![i, seq, cigarx, scores[i], correct]);
    }
    table.printstd();
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::utils::{self};
    use super::*;

    const REF_SIZE: usize = 200;
    const ANCHOR_LEN: usize = 20;
    const RNG_SEED: u64 = 3666;

    /// Tests that the global aligment kernel returns the correct best scores compared to a reference implementation
    #[test] fn cuda_global_aligment_scores_correct() -> KResult<()> {
        let gpu = crisprme_cuda::initialize()?;

        let reference = utils::generate_test_sequence(REF_SIZE, b"ACTG", RNG_SEED);
        let reference_windows = utils::split_windows(&reference, ANCHOR_LEN);

        // Create prefix tree
        let mut tree: Tree<4> = Tree::new(ANCHOR_LEN);
        for seq in &reference_windows {
            tree.insert(seq);
        }

        // Create packed prefix tree and split it in a local tree with span 30
        let packed_tree = tree.pack();
        let split_packed_trees = packed_tree.split_at_width(30);

        // Test many times with different query sequences
        for i in 1..=500 {
            let query_base = utils::generate_test_sequence(
                ANCHOR_LEN, b"ACTG", RNG_SEED+i);

            // Query as u32
            let query: Vec<u32> = query_base.iter()
                .map(|x| u32::try_from(*x).unwrap())
                .collect();

            // Letters as u32
            let bucket: Vec<u32> = utils::linearize_memory(&split_packed_trees[0].layers).iter()
                .map(|l| u32::try_from(*l).unwrap())
                .collect();

            let parents = utils::linearize_memory(&split_packed_trees[0].offset);
            let levels = split_packed_trees[0].layer_sizes();
            let levels_cumsum = split_packed_trees[0].layer_sizes_cumsum();
            let tables = split_packed_trees[0].dp_table_offsets();

            let mut config = Config::for_num_elems(100);
            config.shared_mem_bytes = 48000; // 48K

            // Launch work on GPU and get result
            let (_aligments, scores) = kernel::mine_global_aligment(&gpu, config,
                query,
                bucket,
                parents, 
                levels,
                levels_cumsum,
                tables
            )?;

            let query_base = std::str::from_utf8(&query_base);
            for i in 0..split_packed_trees[0].span() {
                let seq = split_packed_trees[0].sequence_at_leaf(i);
                let ag = needleman(query_base.unwrap(), &seq, 1, -1, -2, false, 'N');
                assert_eq!(ag.score, scores[i]);
            }
        }

        Ok(())
    }
}