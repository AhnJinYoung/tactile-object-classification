// Edits exported by the page and composed back with Hydra (tests/test_report.py runs this file).
//   node tests/js/export_cases.js <sources.json> <out_dir>   -> <out_dir>/<name>.yaml + expected.json
const fs = require('node:fs');
const path = require('node:path');
const compose = require('../../docs/report/src/compose.js');

// Each case: start from a preset, then group switches and value edits, as the page applies them.
const CASES = [
  { name: 'X1_group_switch', from: 'E001_baseline', groups: { matcher: 'latent_head' }, values: { history: 20 } },
  { name: 'X2_scalars', from: 'E001_baseline', groups: {}, values: { dim: 96, policy_hidden: [128, 64], 'env.label': 'binary' } },
  { name: 'X3_stage_values', from: 'E007_history_10', groups: {},
    values: { 'train.stages.contrastive.iters': 50, 'train.stages.contrastive.loss.uniform': 0.5, 'train.stages.encoder_ppo.lr': 1e-4 } },
  { name: 'X4_scheme_and_scale', from: 'E013_attention_16x32', groups: { train: 'joint', scale: 'small' },
    values: { 'train.stages.joint.detach_matcher': false, 'target_encoder.stages': [96, 48], 'touch_encoder.kernel': 3 } },
];

function build(sources, c) {
  const preset = sources.presets[c.from];
  const state = { choices: compose.choicesWith(sources, preset.groups), values: compose.clone(preset.values),
    name: c.name, description: `round trip ${c.name}: "quoted", commas, colons: ok` };
  for (const [g, o] of Object.entries(c.groups)) {
    state.choices[g] = o;
    state.values = compose.withoutGroup(sources, state.values, g);
  }
  Object.assign(state.values, c.values);
  return state;
}

if (require.main === module) {
  const sources = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  const out = process.argv[3];
  const expected = {};
  for (const c of CASES) {
    const state = build(sources, c);
    fs.writeFileSync(path.join(out, `${c.name}.yaml`), compose.exportYaml(sources, state));
    const named = Object.assign({}, state.values, { name: state.name, description: state.description });
    expected[c.name] = { choices: state.choices, cfg: compose.compose(sources, state.choices, named).cfg,
      name: state.name, description: state.description, edits: c.values, command: compose.commandLine(sources, state) };
  }
  fs.writeFileSync(path.join(out, 'expected.json'), JSON.stringify(expected));
}

module.exports = { CASES, build };
