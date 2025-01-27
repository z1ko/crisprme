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
// Wrapper for the CudaDevice
pub type Gpu = Arc<CudaDevice>;

// Include generated CUDA bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Macro used to create the initialize function
macro_rules! create_initialize_func {
    ($($ptx_file:literal),*) => {
        paste! {

            // Include all generated PTX files from the build system
            $(
                const [< PTX_ $ptx_file:upper >]: &str = include_str!(concat!(env!("OUT_DIR"), "/", $ptx_file, ".ptx"));
            )*

            /// Initialize Cuda device and load all PTX
            pub fn initialize() -> KResult<Arc<CudaDevice>> {
                let gpu = CudaDevice::new(0)?;

                // Load all PTX sources to the GPU
                $({
                    println!("Loading source for {}:\n{}", concat!($ptx_file, ".ptx"), [< PTX_ $ptx_file:upper >]);
                    let ptx = Ptx::from_src([< PTX_ $ptx_file:upper >]);
                    gpu.load_ptx(ptx, "crisprme", &[ $ptx_file ])?;
                })*

                Ok(gpu)
            }
        }
    }
}

create_initialize_func!("example");

/// All the kernels will be inside this module
pub mod kernel {
    use super::*;

    /// Function with customizable launch configuration
    pub fn example(gpu: &Gpu, config: LaunchConfig, input: Vec<f32>) -> KResult<Vec<f32>> {
        let n = input.len();
        let input = gpu.htod_copy(input)?; // Copy input to GPU memory
        let f = gpu.get_func("crisprme", "example").unwrap();
        unsafe {
            f.launch(config, (&input, n))?; // Launch kernel on GPU
        }
        gpu.sync_reclaim(input) // Retrive output
    }
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
        f: F,
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
