const fs = require('fs');
const path = require('path');

const engineDir = path.join(__dirname, '..', 'crates', 'engine');
const libDir = path.join(__dirname, '..', 'lib_build', 'lib');

// Platform-specific library names
const platform = process.platform;
let libPatterns;

switch (platform) {
  case 'darwin':
    libPatterns = ['libctranslate2.dylib'];
    break;
  case 'linux':
    libPatterns = ['libctranslate2.so*', 'libgomp.so*'];
    break;
  case 'win32':
    libPatterns = ['*.dll'];
    break;
  default:
    console.log(`⚠️  Unknown platform: ${platform}`);
    process.exit(0);
}

// Copy libraries
let copied = false;
for (const pattern of libPatterns) {
  if (pattern.includes('*')) {
    // Glob pattern - list files
    const files = fs.readdirSync(libDir).filter(f => {
      const regex = new RegExp('^' + pattern.replace('*', '.*') + '$');
      return regex.test(f);
    });
    for (const file of files) {
      const src = path.join(libDir, file);
      const dst = path.join(engineDir, file);
      fs.copyFileSync(src, dst);
      console.log(`✅ Copied ${file} to ${engineDir}`);
      copied = true;
    }
  } else {
    // Exact name
    const src = path.join(libDir, pattern);
    if (fs.existsSync(src)) {
      const dst = path.join(engineDir, pattern);
      fs.copyFileSync(src, dst);
      console.log(`✅ Copied ${pattern} to ${engineDir}`);
      copied = true;
    }
  }
}

if (!copied) {
  console.log(`⚠️  No libraries found in ${libDir}`);
}

console.log('✅ Post-build complete');
