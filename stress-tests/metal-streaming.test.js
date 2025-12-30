#!/usr/bin/env node
/**
 * Metal GPU Streaming Stress Test
 * 
 * Tests streaming transcription performance with Metal GPU acceleration.
 * Simulates concurrent real-time streams to measure throughput and latency.
 */

const { StreamingEngine, decodeAudio, getModelPath, isGpuAvailable, getBestDevice } = require('../index.js');
const fs = require('fs');

// Configuration
const NUM_SESSIONS = parseInt(process.argv[2] || '20');
const STREAM_DURATION_S = parseFloat(process.argv[3] || '5');
const CHUNK_SIZE_MS = parseInt(process.argv[4] || '1000');

const SAMPLE_RATE = 16000;
const CHUNK_SAMPLES = Math.floor(SAMPLE_RATE * CHUNK_SIZE_MS / 1000);

async function main() {
  console.log('═'.repeat(70));
  console.log('  METAL GPU STREAMING STRESS TEST');
  console.log('═'.repeat(70));
  console.log(`  GPU Available:      ${isGpuAvailable()}`);
  console.log(`  Best Device:        ${getBestDevice()}`);
  console.log(`  Concurrent streams: ${NUM_SESSIONS}`);
  console.log(`  Stream duration:    ${STREAM_DURATION_S}s`);
  console.log(`  Chunk size:         ${CHUNK_SIZE_MS}ms`);
  console.log('═'.repeat(70));
  
  // Load audio files
  const audioFiles = [
    './tests/fixtures/hello.wav',
    './tests/fixtures/numbers.wav',
    './tests/fixtures/sentence.wav',
  ].filter(f => fs.existsSync(f));
  
  console.log(`\n📁 Loading ${audioFiles.length} audio files...`);
  
  let allSamples = [];
  for (const f of audioFiles) {
    const samples = decodeAudio(f);
    allSamples.push(...samples);
  }
  
  // Extend to target duration
  const targetSamples = Math.ceil(STREAM_DURATION_S * SAMPLE_RATE);
  while (allSamples.length < targetSamples) {
    allSamples = allSamples.concat(allSamples);
  }
  allSamples = allSamples.slice(0, targetSamples);
  
  console.log(`  Total audio: ${(allSamples.length / SAMPLE_RATE).toFixed(2)}s per stream`);
  
  // Create engine
  console.log('\n🔧 Creating StreamingEngine...');
  const engine = new StreamingEngine(getModelPath('tiny'));
  console.log(`  Sample rate: ${engine.samplingRate()}`);
  
  // Warm up
  console.log('\n⏳ Warming up (Metal shader compilation)...');
  const warmupStart = Date.now();
  const warmupSession = engine.createSession({ language: 'en' });
  engine.processAudio(warmupSession, allSamples.slice(0, CHUNK_SAMPLES * 2));
  engine.flushSession(warmupSession);
  engine.closeSession(warmupSession);
  console.log(`  Warmup time: ${Date.now() - warmupStart}ms`);
  
  // Create sessions
  console.log(`\n🚀 Starting ${NUM_SESSIONS} concurrent streaming sessions...`);
  const sessions = [];
  for (let i = 0; i < NUM_SESSIONS; i++) {
    sessions.push({
      id: engine.createSession({ language: 'en', beamSize: 1 }), // Greedy for speed
      results: [],
      startTime: null,
      endTime: null
    });
  }
  
  const numChunks = Math.ceil(allSamples.length / CHUNK_SAMPLES);
  const testStartTime = Date.now();
  
  // Process chunks - simulating real-time arrival
  for (let chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
    const chunkStart = chunkIdx * CHUNK_SAMPLES;
    const chunkEnd = Math.min((chunkIdx + 1) * CHUNK_SAMPLES, allSamples.length);
    const chunk = allSamples.slice(chunkStart, chunkEnd);
    const chunkDurationS = chunk.length / SAMPLE_RATE;
    const audioEndTimeS = chunkEnd / SAMPLE_RATE;
    
    // Expected arrival time for this chunk (real-time simulation)
    const expectedArrivalTime = testStartTime + (chunkStart / SAMPLE_RATE * 1000);
    const now = Date.now();
    
    // Wait if we're ahead of real-time
    if (expectedArrivalTime > now) {
      await new Promise(r => setTimeout(r, expectedArrivalTime - now));
    }
    
    // Process this chunk for all sessions
    for (const session of sessions) {
      if (!session.startTime) session.startTime = Date.now();
      
      const processStart = Date.now();
      const result = engine.processAudio(session.id, chunk);
      const processTime = Date.now() - processStart;
      
      const wallClockSinceStart = Date.now() - session.startTime;
      const latencyVsRealtime = wallClockSinceStart - (audioEndTimeS * 1000);
      
      session.results.push({
        chunkIdx,
        audioEndTimeS,
        processTimeMs: processTime,
        latencyVsRealtimeMs: latencyVsRealtime,
        stableCount: result.stableSegments.length,
        hasPreview: !!result.previewText
      });
    }
    
    // Progress
    const progress = ((chunkIdx + 1) / numChunks * 100).toFixed(0);
    const elapsed = ((Date.now() - testStartTime) / 1000).toFixed(1);
    process.stdout.write(`\r  [${elapsed}s] Processing chunk ${chunkIdx + 1}/${numChunks} (${progress}%)`);
  }
  
  // Flush all sessions
  console.log('\n\n📤 Flushing sessions...');
  for (const session of sessions) {
    engine.flushSession(session.id);
    engine.closeSession(session.id);
    session.endTime = Date.now();
  }
  
  const totalTestTime = Date.now() - testStartTime;
  
  // Analyze results
  console.log('\n' + '═'.repeat(70));
  console.log('  RESULTS');
  console.log('═'.repeat(70));
  
  // Collect all chunk results
  const allChunks = sessions.flatMap(s => s.results);
  const latencies = allChunks.map(c => c.latencyVsRealtimeMs);
  const processTimes = allChunks.map(c => c.processTimeMs);
  
  // Calculate stats
  const stats = (arr) => {
    const sorted = [...arr].sort((a, b) => a - b);
    const sum = arr.reduce((a, b) => a + b, 0);
    return {
      min: sorted[0],
      max: sorted[sorted.length - 1],
      mean: sum / arr.length,
      median: sorted[Math.floor(sorted.length / 2)],
      p95: sorted[Math.floor(sorted.length * 0.95)],
      p99: sorted[Math.floor(sorted.length * 0.99)]
    };
  };
  
  const latStats = stats(latencies);
  const procStats = stats(processTimes);
  
  const totalAudioS = (allSamples.length / SAMPLE_RATE) * NUM_SESSIONS;
  const throughput = totalAudioS / (totalTestTime / 1000);
  
  console.log('\n📊 OVERVIEW');
  console.log('─'.repeat(50));
  console.log(`  Concurrent sessions:     ${NUM_SESSIONS}`);
  console.log(`  Audio per session:       ${(allSamples.length / SAMPLE_RATE).toFixed(2)}s`);
  console.log(`  Total audio processed:   ${totalAudioS.toFixed(2)}s`);
  console.log(`  Wall clock time:         ${(totalTestTime / 1000).toFixed(2)}s`);
  console.log(`  Throughput:              ${throughput.toFixed(2)}x real-time`);
  
  console.log('\n📈 LATENCY (vs real-time)');
  console.log('─'.repeat(50));
  console.log(`  Min:     ${latStats.min.toFixed(0)}ms`);
  console.log(`  Mean:    ${latStats.mean.toFixed(0)}ms`);
  console.log(`  Median:  ${latStats.median.toFixed(0)}ms`);
  console.log(`  P95:     ${latStats.p95.toFixed(0)}ms`);
  console.log(`  P99:     ${latStats.p99.toFixed(0)}ms`);
  console.log(`  Max:     ${latStats.max.toFixed(0)}ms`);
  
  // Latency buckets
  const excellent = latencies.filter(l => l < 500).length;
  const good = latencies.filter(l => l >= 500 && l < 1000).length;
  const acceptable = latencies.filter(l => l >= 1000 && l < 2000).length;
  const poor = latencies.filter(l => l >= 2000).length;
  
  console.log('\n  Distribution:');
  console.log(`    ✅ < 500ms:    ${excellent} (${(excellent/latencies.length*100).toFixed(1)}%)`);
  console.log(`    👍 500-1000ms: ${good} (${(good/latencies.length*100).toFixed(1)}%)`);
  console.log(`    ⚠️  1-2s:       ${acceptable} (${(acceptable/latencies.length*100).toFixed(1)}%)`);
  console.log(`    ❌ > 2s:       ${poor} (${(poor/latencies.length*100).toFixed(1)}%)`);
  
  console.log('\n🔧 PROCESSING TIME');
  console.log('─'.repeat(50));
  console.log(`  Min:     ${procStats.min.toFixed(0)}ms`);
  console.log(`  Mean:    ${procStats.mean.toFixed(0)}ms`);
  console.log(`  P95:     ${procStats.p95.toFixed(0)}ms`);
  console.log(`  Max:     ${procStats.max.toFixed(0)}ms`);
  
  // Verdict
  console.log('\n' + '═'.repeat(70));
  console.log('  VERDICT');
  console.log('═'.repeat(70));
  
  const checks = [];
  
  // Can we keep up with real-time?
  const realtimeRatio = throughput / NUM_SESSIONS;
  if (realtimeRatio >= 1.0) {
    checks.push({ pass: true, msg: `Can handle ${NUM_SESSIONS} real-time streams (${realtimeRatio.toFixed(2)}x per stream)` });
  } else {
    checks.push({ pass: false, msg: `Cannot keep up: only ${realtimeRatio.toFixed(2)}x real-time per stream` });
  }
  
  // Latency check
  if (latStats.p95 < 2000) {
    checks.push({ pass: true, msg: `P95 latency ${latStats.p95.toFixed(0)}ms < 2s` });
  } else {
    checks.push({ pass: false, msg: `P95 latency ${latStats.p95.toFixed(0)}ms >= 2s` });
  }
  
  // GPU check
  if (isGpuAvailable() && getBestDevice() === 'metal') {
    checks.push({ pass: true, msg: 'Metal GPU acceleration active' });
  } else {
    checks.push({ pass: false, msg: 'Metal GPU not being used' });
  }
  
  const allPassed = checks.every(c => c.pass);
  
  console.log('\n  Checks:');
  for (const c of checks) {
    console.log(`    ${c.pass ? '✅' : '❌'} ${c.msg}`);
  }
  
  // Estimate max concurrent streams
  const maxStreams = Math.floor(throughput);
  console.log(`\n  📈 Estimated max concurrent real-time streams: ~${maxStreams}`);
  
  console.log('\n  ' + (allPassed ? '🎉 ALL CHECKS PASSED!' : '⚠️  SOME CHECKS FAILED'));
  console.log('\n' + '═'.repeat(70));
  
  process.exit(allPassed ? 0 : 1);
}

main().catch(console.error);
