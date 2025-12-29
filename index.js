const path = require('path');
const os = require('os');

// Detect platform and architecture
const platform = os.platform();
const arch = os.arch();

let binaryName;
if (platform === 'darwin' && arch === 'arm64') {
  binaryName = 'whisper-live-engine.darwin-arm64.node';
} else if (platform === 'darwin' && arch === 'x64') {
  binaryName = 'whisper-live-engine.darwin-x64.node';
} else if (platform === 'linux' && arch === 'x64') {
  binaryName = 'whisper-live-engine.linux-x64-gnu.node';
} else if (platform === 'linux' && arch === 'arm64') {
  binaryName = 'whisper-live-engine.linux-arm64-gnu.node';
} else if (platform === 'win32' && arch === 'x64') {
  binaryName = 'whisper-live-engine.win32-x64-msvc.node';
} else {
  throw new Error(`Unsupported platform: ${platform}-${arch}`);
}

// Load the native module from crates/engine
const { Engine } = require(`./crates/engine/${binaryName}`);

module.exports = { Engine };
