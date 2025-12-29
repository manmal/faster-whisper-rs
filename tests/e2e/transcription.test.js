const { describe, it, before } = require('node:test');
const assert = require('node:assert');
const path = require('node:path');
const fs = require('node:fs');

const modelPath = path.join(__dirname, '../../models/tiny');
const hasModel = fs.existsSync(path.join(modelPath, 'model.bin'));
const fixturesDir = path.join(__dirname, '../fixtures');

describe('E2E Transcription', { skip: !hasModel && 'Model not available' }, () => {
  let engine;
  
  before(() => {
    const { Engine } = require('../../index.js');
    engine = new Engine(modelPath);
  });

  describe('English transcription', () => {
    it('should transcribe "hello world"', () => {
      const result = engine.transcribe(path.join(fixturesDir, 'hello.wav'));
      
      // Whisper tiny model may not be perfectly accurate, so we check for key words
      const normalized = result.toLowerCase();
      assert.ok(
        normalized.includes('hello') || normalized.includes('world'),
        `Expected "hello" or "world" in: "${result}"`
      );
    });

    it('should transcribe numbers', () => {
      const result = engine.transcribe(path.join(fixturesDir, 'numbers.wav'));
      
      const normalized = result.toLowerCase();
      // Check for at least some numbers (words or digits)
      const hasNumbers = (
        ['one', 'two', 'three', 'four', 'five'].some(n => normalized.includes(n)) ||
        ['1', '2', '3', '4', '5'].some(n => result.includes(n))
      );
      assert.ok(hasNumbers, `Expected some numbers in: "${result}"`);
    });

    it('should transcribe longer sentences', () => {
      const result = engine.transcribe(path.join(fixturesDir, 'sentence.wav'));
      
      // Should have multiple words
      const words = result.trim().split(/\s+/);
      assert.ok(words.length >= 3, `Expected at least 3 words, got: "${result}"`);
    });
  });

  describe('Performance', () => {
    it('should transcribe in reasonable time', () => {
      const audioPath = path.join(fixturesDir, 'hello.wav');
      
      const start = Date.now();
      engine.transcribe(audioPath);
      const elapsed = Date.now() - start;
      
      // Should complete within 10 seconds for tiny model
      assert.ok(elapsed < 10000, `Transcription took too long: ${elapsed}ms`);
    });

    it('should handle rapid sequential transcriptions', () => {
      const audioPath = path.join(fixturesDir, 'hello.wav');
      
      const results = [];
      for (let i = 0; i < 3; i++) {
        results.push(engine.transcribe(audioPath));
      }
      
      // All results should be the same
      assert.ok(
        results.every(r => r === results[0]),
        'Sequential transcriptions should produce consistent results'
      );
    });
  });
});
