/**
 * Whisper transcription engine powered by CTranslate2
 */
export class Engine {
  /**
   * Create a new transcription engine
   * @param modelPath - Path to the Whisper model directory (CTranslate2 format)
   */
  constructor(modelPath: string);

  /**
   * Transcribe an audio file
   * @param audioPath - Path to WAV file (16kHz, mono, 16-bit PCM)
   * @returns Transcribed text
   */
  transcribe(audioPath: string): string;
}
