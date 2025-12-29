/**
 * Post-build script
 * 
 * With whisper-rs, the whisper.cpp library is statically linked into the
 * binary, so we don't need to copy any dynamic libraries.
 * 
 * This script is kept for potential future use and cleanup.
 */

console.log('✅ Post-build complete (whisper-rs uses static linking)');
