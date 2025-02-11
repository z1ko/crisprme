#![allow(dead_code)]

use cudarc::{
    driver::{CudaDevice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig},
    nvrtc::Ptx,
};

use paste::paste;
use std::sync::Arc;

pub mod autotune;

#[macro_use]
mod macros;

// CUDA errors
pub type KResult<T> = Result<T, DriverError>;
// Wrapper for the CudaDevice
pub type Gpu = Arc<CudaDevice>;
// Rename LaunchConfig to shorter Config
pub type Config = LaunchConfig;

// Include generated CUDA bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// Generate initialization function to load all generated PTX sources from the build system
create_initialize_func!("mine_global_aligment");

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


    /// Mine all aligments of a query string into a trie of sequences
    pub fn mine_global_aligment(
        gpu: &Gpu, 
        config: LaunchConfig,
        levels: Vec<u32>,
        diagonals: Vec<u32>
    ) -> KResult<()> {

        let levels = gpu.htod_copy(levels)?;
        let diagonals = gpu.htod_copy(diagonals)?;

        let f = gpu.get_func("crisprme", "mine_global_aligment").unwrap();
        unsafe {
            f.launch(config, (&levels, &diagonals))?;
        }

        Ok(())
    }
}
