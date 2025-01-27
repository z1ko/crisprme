#![allow(dead_code)]

pub use cudarc::{
    driver::{CudaDevice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig},
    nvrtc::Ptx,
};

use paste::paste;

use std::sync::Arc;
use std::time::Instant;

// CUDA errors
pub type KResult<T> = Result<T, DriverError>;

// Include generated CUDA bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// Include the compiled PTX code as string
const CUDA_KERNEL_PTX_EXAMPLE: &str = include_str!(concat!(env!("OUT_DIR"), "/example.ptx"));

/// Macro to generate boilerplater for kernel function with basic launch configuration
macro_rules! generate_basic_kernel_call {
    ($kernel_name:ident, $o:ty, $($v:tt: $t:ty),+) => { paste! {
        /// Function that will be used to invoke the kernel by the consumer with simpler parameters
        pub fn [< $kernel_name _simple >] (gpu: &Arc<CudaDevice>, n: u32, $($v: $t),+) -> KResult<$o> {
            let config = LaunchConfig::for_num_elems(n as u32);
            $kernel_name(gpu, config, n, $($v)+)
        }
    } };
}

/// All the kernels will be inside this module
pub mod kernel {
    use super::*;

    /// Function with customizable launch configuration
    pub fn example(
        gpu: &Arc<CudaDevice>,
        config: LaunchConfig,
        n: u32,
        input: Vec<f32>,
    ) -> KResult<Vec<f32>> {
        let input = gpu.htod_copy(input)?; // Copy input to GPU memory
        let f = gpu.get_func("crisprme", "example").unwrap();
        unsafe {
            f.launch(config, (&input, n))?; // Launch kernel on GPU
        }
        gpu.sync_reclaim(input) // Retrive output
    }

    generate_basic_kernel_call!(example, Vec<f32>, input: Vec<f32>);
}

/// Initialize Cuda device and load all PTX
pub fn initialize() -> KResult<Arc<CudaDevice>> {
    let gpu = CudaDevice::new(0)?;

    // Load the ptx of the kernel
    let ptx = Ptx::from_src(CUDA_KERNEL_PTX_EXAMPLE);
    gpu.load_ptx(ptx, "crisprme", &["example"])?;

    Ok(gpu)
}

/// Support function to automatize launch configuration
pub mod autotune {
    use super::*;

    /// Describe what to explore during autotuning
    pub struct Search {
        pub configs: Vec<LaunchConfig>,
    }

    impl Search {
        /// Generates all possible launch configurations
        pub fn configurations(&self) -> impl Iterator<Item = &LaunchConfig> {
            self.configs.iter()
        }
    }

    /// Analyze kernel and return best launch configuration
    pub fn benchmark<F>(
        gpu: &Arc<CudaDevice>,
        search: Search,
        mut f: F,
    ) -> KResult<(LaunchConfig, u128)>
    where
        F: Fn(&Arc<CudaDevice>, LaunchConfig),
    {
        let best_config = search
            .configurations()
            .map(|c| {
                let s = Instant::now();
                f(gpu, *c);
                (*c, s.elapsed().as_nanos())
            })
            .min_by_key(|(_, t)| *t)
            .unwrap();
        Ok(best_config)
    }
}
