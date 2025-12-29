use std::env;
use std::path::PathBuf;

fn main() {
    napi_build::setup();

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let lib_path = PathBuf::from(&manifest_dir).join("../../lib_build/lib");

    println!("cargo:rustc-link-search=native={}", lib_path.display());
    println!("cargo:rustc-link-lib=dylib=ctranslate2");

    let rpath = lib_path.canonicalize().unwrap_or(lib_path.clone());
    
    // Platform-specific rpath/library loading
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", rpath.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
    }
    
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", rpath.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    }
    
    // Windows: DLLs are found via PATH or same directory as executable
    #[cfg(target_os = "windows")]
    {
        // On Windows, the DLL needs to be in PATH or next to the .node file
        // No additional linker args needed
    }
}
