use std::{env, path::PathBuf, process::Command};

use bindgen::CargoCallbacks;
use regex::Regex;

// Specify the desired architecture version
const ARCH: &str = "compute_86";
const CODE: &str = "sm_86";

fn main() {
    // Tell cargo to invalidate the built crate whenever files of interest changes.
    println!("cargo:rerun-if-changed={}", "cuda");

    // Where the articats will be located
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let cuda_src = PathBuf::from("src/cuda/kernels/example.cu");
    let ptx = out_dir.join("example.ptx");

    let nvcc = Command::new("nvcc")
        .arg("-ptx")
        .arg("-o")
        .arg(&ptx)
        .arg(&cuda_src)
        .arg(format!("-arch={}", ARCH))
        .arg(format!("-code={}", CODE))
        .status()
        .unwrap();

    assert!(nvcc.success(), "Failed to compile CUDA source to PTX.");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("src/cuda/include/wrapper.h")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(CargoCallbacks))
        // we use "no_copy" and "no_debug" here because we don't know if we can safely generate them for our structs in C code (they may contain raw pointers)
        .no_copy("*")
        .no_debug("*")
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // We need to make modifications to the generated code
    let generated_bindings = bindings.to_string();

    // Regex to find raw pointers to float and replace them with CudaSlice<f32>
    // You can copy this regex to add/modify other types of pointers, for example "*mut i32"
    let pointer_regex = Regex::new(r"\*mut f32").unwrap();
    let modified_bindings = pointer_regex.replace_all(&generated_bindings, "CudaSlice<f32>");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    std::fs::write(out_path.join("bindings.rs"), modified_bindings.as_bytes())
        .expect("Failed to write bindings");
}
