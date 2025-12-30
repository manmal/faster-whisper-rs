fn main() {
    napi_build::setup();
    
    // whisper-rs links whisper.cpp statically, no additional linking needed
    
    // On macOS, we need to link to Metal and CoreML frameworks
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-framework");
        println!("cargo:rustc-link-arg=Metal");
        println!("cargo:rustc-link-arg=-framework");
        println!("cargo:rustc-link-arg=MetalKit");
        println!("cargo:rustc-link-arg=-framework");
        println!("cargo:rustc-link-arg=Accelerate");
        println!("cargo:rustc-link-arg=-framework");
        println!("cargo:rustc-link-arg=CoreML");
        println!("cargo:rustc-link-arg=-framework");
        println!("cargo:rustc-link-arg=Foundation");
    }
}
