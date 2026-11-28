// "Scene and objects" facts, generated from the composed `env` config and the YCB list of tacfind/envs/ycb.py.
(function (root) {
  'use strict';
  const m = (x) => `${Number(x.toPrecision(3))} m`;
  const range = (r, unit = '') => `${r[0]}–${r[1]}${unit}`;

  // Returns [label, text] rows. env: composed cfg.env; ycb: {name: [mass kg, material, split]}.
  function sceneFacts(env, ycb) {
    const rows = [];
    const t = env.table, z = env.zone;
    rows.push(['Table', `${m(t.size[0])} deep × ${m(t.size[1])} long; the robot base stands at the middle of a long edge (scene.py). No bin, no walls.`]);
    rows.push(['Object zone', `${m(z.size[0])} × ${m(z.size[1])}, centred ${m(Math.abs(z.center[0]))} from the robot base: object centres are scattered here, inside the arm's reach.`]);
    rows.push(['Objects per scene', `${range(env.n_active)}, drawn uniformly at every reset; one of them is the target, the rest are distractors.`]);
    if (env.objects.set === 'ycb' && ycb && Object.keys(ycb).length) {
      const split = (s) => Object.entries(ycb).filter(([, v]) => v[2] === s).map(([k]) => k);
      const masses = Object.values(ycb).map((v) => v[0]);
      rows.push(['Library', `${Object.keys(ycb).length} YCB objects, masses from the YCB object list (${Math.round(Math.min(...masses) * 1000)}–${Math.round(Math.max(...masses) * 1000)} g): ${split('train').length} in training worlds, ${split('heldout').length} held out (${split('heldout').join(', ')}).`]);
    } else {
      rows.push(['Library', `${env.objects.n_objects} procedural objects of ${env.objects.families.length} families; ${Math.round(env.objects.heldout_frac * 100)} % held out.`]);
    }
    const r = env.randomize;
    if (r && r.enabled) {
      rows.push(['Domain randomisation', `per world at every reset: mass × ${range(r.mass_scale)}, friction × ${range(r.friction_scale)}, centre of mass ± ${Math.round(r.com_offset * 1000)} mm, fingertip force noise ${r.force_noise} N, contact threshold ${range(r.force_threshold, ' N')}.`]);
    }
    rows.push(['Success', `the target's lowest point ${m(env.lift_height)} above the table for ${env.hold_time} s. The same with a distractor is a wrong pick. Both end the episode, as do the target falling off the table (failure) and a time-out after ${env.timeout} s.`]);
    const w = env.reward, on = Object.entries(w).filter(([, v]) => v);
    const off = Object.entries(w).filter(([, v]) => !v).map(([k]) => k);
    rows.push(['Reward', `${on.map(([k, v]) => `${k} ${v > 0 ? '+' : ''}${v}`).join(', ')}${off.length ? `; off: ${off.join(', ')}` : ''}${on.every(([k]) => k === 'success' || k === 'wrong_pick') ? ' (sparse)' : ''}.`]);
    rows.push(['Control', `${Math.round(1 / (env.timestep * env.substeps))} Hz (${env.substeps} physics steps of ${Number((env.timestep * 1000).toPrecision(3))} ms); ${env.n_slots} object slots in the padded tensors.`]);
    return rows;
  }

  const api = { sceneFacts };
  root.TF = root.TF || {};
  root.TF.scene = api;
  if (typeof module === 'object' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
