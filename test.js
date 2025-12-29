/**
 * Phase 1 Test - Verifies new segment output, options, and audio handling
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

console.log('Loading faster-whisper-node...');
const { Engine, availableModels, formatTimestamp } = require('./index');

// Test 1: availableModels()
console.log('\n📋 Test 1: Available models');
const models = availableModels();
console.log('   Models:', models.join(', '));
if (!models.includes('tiny')) {
  console.error('❌ availableModels() should include "tiny"');
  process.exit(1);
}
console.log('   ✅ Passed');

// Test 2: formatTimestamp()
console.log('\n📋 Test 2: formatTimestamp()');
const ts1 = formatTimestamp(65.5);       // 01:05.500
const ts2 = formatTimestamp(65.5, true); // 00:01:05.500
const ts3 = formatTimestamp(3661.123, true); // 01:01:01.123
console.log(`   formatTimestamp(65.5) = "${ts1}"`);
console.log(`   formatTimestamp(65.5, true) = "${ts2}"`);
console.log(`   formatTimestamp(3661.123, true) = "${ts3}"`);
if (!ts1.includes('01:05')) {
  console.error('❌ formatTimestamp(65.5) should format correctly');
  process.exit(1);
}
console.log('   ✅ Passed');

// Test 3: Basic constructor
console.log('\n📋 Test 3: Engine constructor');
console.log('   Loading model from:', modelPath);
const engine = new Engine(modelPath);
console.log('   Sampling rate:', engine.samplingRate(), 'Hz');
console.log('   Is multilingual:', engine.isMultilingual());
console.log('   Num languages:', engine.numLanguages());
if (engine.samplingRate() !== 16000) {
  console.error('❌ Sampling rate should be 16000');
  process.exit(1);
}
console.log('   ✅ Passed');

// Test 4: Simple transcribe (backward compatible)
console.log('\n📋 Test 4: Simple transcribe (backward compatible)');
const simpleResult = engine.transcribe(audioPath);
console.log('   Result:', simpleResult);
if (typeof simpleResult !== 'string' || simpleResult.trim().length === 0) {
  console.error('❌ transcribe() should return non-empty string');
  process.exit(1);
}
console.log('   ✅ Passed');

// Test 5: Transcribe with segments
console.log('\n📋 Test 5: Transcribe with segments');
const segmentResult = engine.transcribeSegments(audioPath);
console.log('   Duration:', segmentResult.duration.toFixed(2), 'seconds');
console.log('   Language:', segmentResult.language);
console.log('   Full text:', segmentResult.text);
console.log('   Segments:', segmentResult.segments.length);

for (const seg of segmentResult.segments) {
  console.log(`     [${formatTimestamp(seg.start)} -> ${formatTimestamp(seg.end)}] ${seg.text}`);
}

if (!segmentResult.text || segmentResult.segments.length === 0) {
  console.error('❌ transcribeSegments() should return segments');
  process.exit(1);
}
if (typeof segmentResult.duration !== 'number' || segmentResult.duration <= 0) {
  console.error('❌ Duration should be positive');
  process.exit(1);
}
console.log('   ✅ Passed');

// Test 6: Transcribe with options
console.log('\n📋 Test 6: Transcribe with options');
const optionResult = engine.transcribeWithOptions(audioPath, {
  beamSize: 3,
  patience: 1.0,
  temperature: 0.0,
  language: 'en'
});
console.log('   Result:', optionResult);
if (typeof optionResult !== 'string' || optionResult.trim().length === 0) {
  console.error('❌ transcribeWithOptions() should return non-empty string');
  process.exit(1);
}
console.log('   ✅ Passed');

// Test 7: Transcribe from buffer
console.log('\n📋 Test 7: Transcribe from buffer');
const audioBuffer = fs.readFileSync(audioPath);
const bufferResult = engine.transcribeBuffer(audioBuffer);
console.log('   Result:', bufferResult.text);
if (typeof bufferResult.text !== 'string' || bufferResult.text.trim().length === 0) {
  console.error('❌ transcribeBuffer() should return result with text');
  process.exit(1);
}
console.log('   ✅ Passed');

// Test 8: Engine.withOptions factory
console.log('\n📋 Test 8: Engine.withOptions factory');
const engine2 = Engine.withOptions(modelPath, {
  device: 'cpu',
  computeType: 'default',
  cpuThreads: 4
});
const result2 = engine2.transcribe(audioPath);
console.log('   Result:', result2);
if (typeof result2 !== 'string' || result2.trim().length === 0) {
  console.error('❌ Engine.withOptions should work');
  process.exit(1);
}
console.log('   ✅ Passed');

console.log('\n✅ All Phase 1 tests passed!');
