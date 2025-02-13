use std::time::Instant;

use crisprme_cuda::{autotune, kernel, Config, KResult};

fn memory_of_bucket(levels: &[u32], query_len: usize, bucket_len: usize) -> u32 {
    let mut result = 0;
    for l in levels {
        result += (*l as usize) * query_len;
    }
    return (result + bucket_len * 2) as u32;
}

/// Example usage of kernels
fn main() -> KResult<()> {
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

    /*
    // Let's find the best launch configuration
    let search = autotune::Search {
        configs: vec![
            Config::for_num_elems(1000),
            Config::for_num_elems(100),
            Config::for_num_elems(10),
        ],
    };

    println!(
        "Best configuration: {:?}",
        autotune::benchmark(&gpu, search, |gpu, c| {
            let input = vec![0.0; 1000];

            let start = Instant::now();
            kernel::example(gpu, c, input).unwrap();
            start.elapsed()
        })?
    );
    Ok(())
    */

}
