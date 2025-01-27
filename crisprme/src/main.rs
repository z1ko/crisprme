use std::time::Instant;

use crisprme_cuda::{autotune, kernel, Config, KResult};

/// Example usage of kernels
fn main() -> KResult<()> {
    let gpu = crisprme_cuda::initialize()?;

    // Launch work on GPU and get result
    let config = Config::for_num_elems(1000);
    let output = kernel::example(&gpu, config, vec![0.0; 1000])?;
    println!("{:?}", output);

    // Let's find the best launch configuration
    let s = autotune::Search {
        configs: vec![
            Config::for_num_elems(1000),
            Config::for_num_elems(100),
            Config::for_num_elems(10),
        ],
    };

    println!(
        "Best configuration: {:?}",
        autotune::benchmark(&gpu, s, |gpu, c| {
            let input = vec![0.0; 1000];

            let start = Instant::now();
            kernel::example(gpu, c, input).unwrap();
            start.elapsed()
        })?
    );
    Ok(())
}
