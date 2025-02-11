use std::time::Instant;

use crisprme_cuda::{autotune, kernel, Config, KResult};

/// Example usage of kernels
fn main() -> KResult<()> {
    let gpu = crisprme_cuda::initialize()?;

    let mut config = Config::for_num_elems(100);
    config.shared_mem_bytes = 1024; // TODO: Calculate
    
    // Launch work on GPU and get result
    kernel::mine_global_aligment(&gpu, config, 
        vec![1, 2, 3], 
        vec![1, 3, 6, 5, 3]
    )?;
    return Ok(());

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
}
