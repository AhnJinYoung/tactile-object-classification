// Shared by the Node tests: Python's reference data (Hydra compositions, symbol values) and the page scripts.
const { execFileSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

const REPO = path.resolve(__dirname, '..', '..');
const SRC = path.join(REPO, 'docs', 'report', 'src');

function python() {
  if (process.env.PYTHON) return process.env.PYTHON;
  const venv = path.join(REPO, '.venv', 'bin', 'python');
  return fs.existsSync(venv) ? venv : 'python3';
}

let cached = null;
function reference() {
  if (!cached) {
    const out = execFileSync(python(), ['-m', 'tacfind.report'], { cwd: REPO, maxBuffer: 64 << 20 });
    cached = JSON.parse(out.toString());
  }
  return cached;
}

const compose = require(path.join(SRC, 'compose.js'));
const symbols = require(path.join(SRC, 'symbols.js'));

module.exports = { REPO, SRC, reference, compose, symbols };
