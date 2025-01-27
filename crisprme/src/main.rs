use crisprme_cuda::{KResult, LaunchConfig};

/// Example usage of kernels
fn main() -> KResult<()> {
    let gpu = crisprme_cuda::initialize()?;

    // Launch work on GPU and get result
    let output = crisprme_cuda::kernel::example(&gpu, vec![0.0; 1000])?;
    println!("{:?}", output);

    // Let's find the best launch configuration
    let s = crisprme_cuda::autotune::Search {
        configs: vec![
            LaunchConfig::for_num_elems(1000),
            LaunchConfig::for_num_elems(100),
            LaunchConfig::for_num_elems(10),
        ],
    };

    println!(
        "best configuration: {:?}",
        crisprme_cuda::autotune::benchmark(&gpu, s, |gpu, c| {
            crisprme_cuda::kernel::example_with_config(gpu, c, vec![0.0; 1000]).unwrap();
        })?
    );
    Ok(())
}
