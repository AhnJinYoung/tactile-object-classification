// "Scene and objects" facts from the composed env config of the baseline preset.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { SRC, reference } = require('./reference.js');
const { sceneFacts } = require(path.join(SRC, 'scene.js'));

test('scene facts come from the config and the YCB list', () => {
  const ref = reference();
  const env = ref.presets.E001_baseline.cfg.env;
  const facts = Object.fromEntries(sceneFacts(env, ref.ycb));
  assert.match(facts.Table, new RegExp(`^${env.table.size[0]} m deep × ${env.table.size[1]} m long`));
  assert.match(facts['Objects per scene'], new RegExp(`^${env.n_active[0]}–${env.n_active[1]},`));
  const n = Object.keys(ref.ycb).length, held = Object.values(ref.ycb).filter((v) => v[2] === 'heldout').length;
  assert.match(facts.Library, new RegExp(`^${n} YCB objects.*: ${n - held} in training worlds, ${held} held out`));
  assert.match(facts.Success, new RegExp(`${env.lift_height} m above the table for ${env.hold_time} s`));
  assert.match(facts.Reward, /^success \+10, wrong_pick -5; off: .* \(sparse\)\.$/);
});
