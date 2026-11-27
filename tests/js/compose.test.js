// The page's Hydra composition and symbol table against Python (Hydra itself and tacfind/report/specs.py).
const test = require('node:test');
const assert = require('node:assert/strict');
const { reference, compose, symbols } = require('./reference.js');
const { CASES, build } = require('./export_cases.js');

const ref = reference();

test('every preset composes exactly as Hydra composes it', () => {
  for (const [name, p] of Object.entries(ref.presets)) {
    const preset = ref.sources.presets[name];
    const choices = compose.choicesWith(ref.sources, preset.groups);
    for (const [g, o] of Object.entries(choices)) assert.equal(o, p.choices[g], `${name}: group ${g}`);
    const values = Object.assign({ name: preset.name, description: preset.description }, preset.values);
    const { cfg } = compose.compose(ref.sources, choices, values);
    assert.deepStrictEqual(cfg, p.cfg, name);
  }
});

test('symbol values match Python for every preset', () => {
  for (const [name, p] of Object.entries(ref.presets)) {
    assert.deepStrictEqual(symbols.symbolValues(ref.specs, p.cfg, p.choices), p.symbols, name);
  }
});

test('expanded step shapes evaluate for every preset and module', () => {
  for (const [name, p] of Object.entries(ref.presets)) {
    const sym = symbols.symbolValues(ref.specs, p.cfg, p.choices);
    for (const m of Object.values(symbols.activeModules(ref.specs, p.choices))) {
      for (const s of symbols.expandSteps(m, p.cfg, sym)) {
        assert.doesNotMatch(symbols.formatShape(s.shape, p.cfg, s.symbols), /\?/, `${name}: ${s.name}`);
      }
    }
  }
});

test('expression grammar', () => {
  const cfg = { a: { b: [3, 5, 7] }, k: 16 };
  assert.equal(symbols.evaluate('min(k, a.b[-1]) * 2 + 1', cfg, {}), 15);
  assert.equal(symbols.evaluate('at(a.b, 0 - 1, 99) - len(a.b) // 2', cfg, {}), 98);
  assert.equal(symbols.evaluate('k', cfg, { k: 4 }, 'k'), 16);  // a symbol is never its own definition
  assert.equal(symbols.formatShape(['B', 'k', 'k * 2', 6], cfg, { k: 4 }), '(B, k=4, k * 2 = 8, 6)');
});

test('export: changed groups and keys only, ready for Hydra', () => {
  const yaml = compose.exportYaml(ref.sources, build(ref.sources, CASES[0]));
  assert.match(yaml, /^# @package _global_\ndefaults:\n  - override \/matcher: latent_head\nname: X1_group_switch\n/);
  assert.match(yaml, /\nhistory: 20\n/);
  assert.doesNotMatch(yaml, /dim:|stages:/);
  const staged = compose.exportYaml(ref.sources, build(ref.sources, CASES[2]));
  assert.match(staged, /train:\n  stages:\n    contrastive:\n      iters: 50\n      loss:\n        uniform: 0.5\n/);
  assert.match(staged, /encoder_ppo:\n      lr: 0.0001\n/);
  assert.match(staged, /\nhistory: 10\n/);  // the preset's own change is kept
  const unchanged = compose.exportYaml(ref.sources, build(ref.sources, { name: 'same', from: 'E001_baseline', groups: {}, values: { history: 30 } }));
  assert.doesNotMatch(unchanged, /history|defaults/);
});

test('switching a group drops the edits it defines', () => {
  const s = build(ref.sources, CASES[3]);
  assert.equal(s.values.attn_groups, undefined);  // E013's attention width belongs to the scale group
  const { cfg } = compose.compose(ref.sources, s.choices, s.values);
  assert.equal(cfg.dim, 64);
  assert.equal(cfg.matcher.groups, cfg.attn_groups);
});

test('command line mirrors the file', () => {
  const cmd = compose.commandLine(ref.sources, build(ref.sources, CASES[1]));
  assert.match(cmd, /^python scripts\/train\.py /);
  for (const arg of ['dim=96', "'policy_hidden=[128,64]'", 'env.label=binary', 'name=X2_scalars', "'description=\"round trip"]) {
    assert.ok(cmd.includes(` ${arg}`), arg);
  }
});

test('consistency checks', () => {
  const cfg = compose.compose(ref.sources, compose.choicesWith(ref.sources, {}), { dim: 100 }).cfg;
  assert.equal(compose.checks(cfg, compose.choicesWith(ref.sources, {}), 512).length, 1);
});
