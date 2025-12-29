const { describe, it, before } = require('node:test');
const assert = require('node:assert');
const path = require('node:path');
const fs = require('node:fs');

// Skip if model not available (CI will download it)
const modelPath = path.join(__dirname, '../../models/tiny');
const hasModel = fs.existsSync(path.join(modelPath, 'model.bin'));

describe('Engine', { skip: !hasModel && 'Model not available' }, () => {
  let Engine;
  
  before(() => {
    const engineModule = require('../../index.js');
    Engine = engineModule.Engine;
  });

  describe('constructor', () => {
    it('should create engine with valid model path', () => {
      const engine = new Engine(modelPath);
      assert.ok(engine, 'Engine should be created');
    });

    it('should throw on invalid model path', () => {
      assert.throws(() => {
        new Engine('/nonexistent/path');
      }, /Error|not found|failed/i);
    });
  });

  describe('transcribe', () => {
    it('should transcribe audio file', () => {
      const engine = new Engine(modelPath);
      const audioPath = path.join(__dirname, '../fixtures/hello.wav');
      
      const result = engine.transcribe(audioPath);
      
      assert.ok(typeof result === 'string', 'Result should be a string');
      assert.ok(result.length > 0, 'Result should not be empty');
    });

    it('should throw on invalid audio path', () => {
      const engine = new Engine(modelPath);
      
      assert.throws(() => {
        engine.transcribe('/nonexistent/audio.wav');
      }, /Error|not found|failed/i);
    });

    it('should handle multiple transcriptions', () => {
      const engine = new Engine(modelPath);
      const audioPath = path.join(__dirname, '../fixtures/hello.wav');
      
      const result1 = engine.transcribe(audioPath);
      const result2 = engine.transcribe(audioPath);
      
      assert.strictEqual(result1, result2, 'Same audio should produce same result');
    });
  });
});
