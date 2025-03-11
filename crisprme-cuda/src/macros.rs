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
                    println!("Loading source for {}", concat!($ptx_file, ".ptx"));
                    let ptx = Ptx::from_src([< PTX_ $ptx_file:upper >]);
                    gpu.load_ptx(ptx, "crisprme", &[ $ptx_file ])?;
                })*

                Ok(gpu)
            }
        }
    }
}
