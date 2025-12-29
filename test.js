/**
 * Simple smoke test - verifies basic functionality
 */
const path = require('path');
const fs = require('fs');

const modelPath = './models/tiny';
const audioPath = './tests/fixtures/hello.wav';

// Check prerequisites
if (!fs.existsSync(path.join(modelPath, 'model.bin'))) {
  console.log('⚠️  Model not found. Download with:');
  console.log('   cd models && git lfs install && git clone --depth 1 https://huggingface.co/Systran/faster-whisper-tiny tiny');
  process.exit(0);
}

if (!fs.existsSync(audioPath)) {
  console.log('⚠️  Test audio not found:', audioPath);
  process.exit(1);
}

console.log('Loading model from:', modelPath);
const { Engine } = require('./index');
const engine = new Engine(modelPath);

console.log('Transcribing:', audioPath);
const result = engine.transcribe(audioPath);

console.log('\n📝 Transcription:');
console.log(result);

// Basic validation
if (typeof result !== 'string' || result.trim().length === 0) {
  console.error('\n❌ Test failed: Empty or invalid result');
  process.exit(1);
}

console.log('\n✅ Test passed');
