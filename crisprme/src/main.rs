use crisprme_cuda::{autotune, kernel, KResult, LaunchConfig};

/// Example usage of kernels
fn main() -> KResult<()> {
    let gpu = crisprme_cuda::initialize()?;

    // Launch work on GPU and get result
    let config = LaunchConfig::for_num_elems(1000);
    let output = kernel::example(&gpu, config, vec![0.0; 1000])?;
    println!("{:?}", output);

    // Let's find the best launch configuration
    let s = autotune::Search {
        configs: vec![
            LaunchConfig::for_num_elems(1000),
            LaunchConfig::for_num_elems(100),
            LaunchConfig::for_num_elems(10),
        ],
    };

    println!(
        "Best configuration: {:?}",
        autotune::benchmark(&gpu, s, |gpu, c| {
            kernel::example(gpu, c, vec![0.0; 1000]).unwrap();
        })?
    );
    Ok(())
}
