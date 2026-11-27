// Hydra composition for this repository, re-implemented for the page, and the export of an edited experiment.
// Merge order (as Hydra): group options in the root defaults order (`_global_` packages at the root, others under
// their group name) < the root config (`_self_` comes last in its defaults) < experiment preset < user edits;
// then `${key}` interpolation. Dicts merge key by key, lists are replaced.
// tests/js/compose.test.js compares it with Hydra's own result for every preset.
(function (root) {
  'use strict';

  const clone = (x) => (x === undefined ? undefined : JSON.parse(JSON.stringify(x)));
  const isObj = (x) => x !== null && typeof x === 'object' && !Array.isArray(x);

  function merge(dst, src) {
    for (const [k, v] of Object.entries(src || {})) {
      if (isObj(v) && isObj(dst[k])) merge(dst[k], v);
      else dst[k] = clone(v);
    }
    return dst;
  }

  function setPath(obj, path, value) {
    const keys = path.split('.');
    let node = obj;
    for (const k of keys.slice(0, -1)) { if (!isObj(node[k])) node[k] = {}; node = node[k]; }
    node[keys[keys.length - 1]] = clone(value);
  }

  function getPath(obj, path) {
    let node = obj;
    for (const k of path.split('.')) { if (!isObj(node) || !(k in node)) return undefined; node = node[k]; }
    return node;
  }

  function flatten(tree, prefix = '', out = {}) {
    for (const [k, v] of Object.entries(tree)) {
      if (isObj(v) && Object.keys(v).length) flatten(v, `${prefix}${k}.`, out);
      else out[`${prefix}${k}`] = v;
    }
    return out;
  }

  // Selected option per group: root defaults, then the given choices.
  function choicesWith(src, chosen) {
    const out = {};
    for (const [g, o] of src.defaults) out[g] = (chosen && chosen[g]) || o;
    return out;
  }

  // Unresolved composed config.
  function composeRaw(src, choices, values) {
    const cfg = {};
    for (const [g, def] of src.defaults) {
      const group = src.groups[g];
      const content = group.options[choices[g] || def] || {};
      if (group.global) merge(cfg, content);
      else merge(cfg[g] = isObj(cfg[g]) ? cfg[g] : {}, content);
    }
    merge(cfg, src.root);
    for (const [p, v] of Object.entries(values || {})) setPath(cfg, p, v);
    return cfg;
  }

  // Resolve `${a.b}` interpolations (a whole-string interpolation keeps the value's type).
  function resolve(raw) {
    const done = new Map();
    function value(path, depth) {
      if (depth > 32) throw new Error(`interpolation cycle at ${path}`);
      return walk(getPath(raw, path), depth + 1);
    }
    function walk(node, depth) {
      if (typeof node === 'string') {
        const whole = node.match(/^\$\{([^}]+)\}$/);
        if (whole) return value(whole[1].trim(), depth);
        return node.replace(/\$\{([^}]+)\}/g, (_, p) => String(value(p.trim(), depth)));
      }
      if (Array.isArray(node)) return node.map((x) => walk(x, depth));
      if (isObj(node)) {
        if (done.has(node)) return done.get(node);
        const out = {};
        for (const [k, v] of Object.entries(node)) out[k] = walk(v, depth);
        done.set(node, out);
        return out;
      }
      return node;
    }
    return walk(raw, 0);
  }

  function compose(src, choices, values) {
    const raw = composeRaw(src, choices, values);
    return { raw, cfg: resolve(raw) };
  }

  // Keys a group option may define (for dropping edits when the group's option changes).
  function groupKeys(src, group) {
    const g = src.groups[group];
    if (!g.global) return [group];
    const keys = new Set();
    for (const opt of Object.values(g.options)) Object.keys(opt).forEach((k) => keys.add(k));
    return [...keys];
  }

  function withoutGroup(src, values, group) {
    const keys = groupKeys(src, group);
    const out = {};
    for (const [p, v] of Object.entries(values)) {
      if (!keys.some((k) => p === k || p.startsWith(`${k}.`))) out[p] = v;
    }
    return out;
  }

  // What an experiment file must state: changed groups and changed keys, relative to the root config.
  function changes(src, choices, values) {
    const groups = {};
    for (const [g, o] of src.defaults) if (choices[g] && choices[g] !== o) groups[g] = choices[g];
    const before = flatten(composeRaw(src, choices, {}));
    const after = flatten(composeRaw(src, choices, values));
    const keys = {};
    for (const [p, v] of Object.entries(after)) {
      if (p === 'name' || p === 'description') continue;
      if (JSON.stringify(v) !== JSON.stringify(before[p])) keys[p] = v;
    }
    return { groups, keys };
  }

  // ------------------------------------------------------------------------------------------------- YAML --
  const RESERVED = /^(true|false|yes|no|on|off|null|y|n|~)$/i;

  function scalar(v) {
    if (v === null || v === undefined) return 'null';
    if (typeof v === 'boolean') return String(v);
    if (typeof v === 'number') {
      if (Number.isInteger(v)) return String(v);
      const s = String(v);
      return /e/i.test(s) && !s.includes('.') ? s.replace(/e/i, '.0e') : s;
    }
    if (Array.isArray(v)) return `[${v.map(scalar).join(', ')}]`;
    const s = String(v);
    if (/^[A-Za-z_][\w./-]*$/.test(s) && !RESERVED.test(s)) return s;
    return JSON.stringify(s);
  }

  function emitTree(tree, indent) {
    let out = '';
    for (const [k, v] of Object.entries(tree)) {
      if (isObj(v) && Object.keys(v).length) out += `${indent}${k}:\n${emitTree(v, `${indent}  `)}`;
      else out += `${indent}${k}: ${scalar(v)}\n`;
    }
    return out;
  }

  // A ready-to-use experiment file (configs/experiment/<name>.yaml).
  function exportYaml(src, state) {
    const { groups, keys } = changes(src, state.choices, state.values);
    let out = '# @package _global_\n';
    if (Object.keys(groups).length) {
      out += 'defaults:\n';
      for (const [g, o] of Object.entries(groups)) out += `  - override /${g}: ${o}\n`;
    }
    out += `name: ${scalar(state.name)}\n`;
    out += `description: ${JSON.stringify(state.description || '')}\n`;
    const tree = {};
    for (const [p, v] of Object.entries(keys)) setPath(tree, p, v);
    return out + emitTree(tree, '');
  }

  function shellQuote(s) { return /^[\w.=,/:+-]+$/.test(s) ? s : `'${s.replace(/'/g, `'\\''`)}'`; }

  function cliValue(v) {
    if (Array.isArray(v)) return `[${v.map(cliValue).join(',')}]`;
    if (typeof v === 'string') return /^[A-Za-z_][\w./-]*$/.test(v) ? v : JSON.stringify(v);
    return scalar(v);
  }

  // The same experiment as training command-line overrides.
  function commandLine(src, state) {
    const { groups, keys } = changes(src, state.choices, state.values);
    const args = Object.entries(groups).map(([g, o]) => `${g}=${o}`);
    for (const [p, v] of Object.entries(keys)) args.push(`${p}=${cliValue(v)}`);
    args.push(`name=${cliValue(String(state.name))}`);
    if (state.description) args.push(`description=${JSON.stringify(state.description)}`);
    return ['python scripts/train.py', ...args.map(shellQuote)].join(' ');
  }

  // Consistency checks the networks would otherwise fail on (constructor asserts or shape errors).
  function checks(cfg, choices, nPoints) {
    const out = [];
    if (cfg.dim % cfg.attn_groups) out.push(`dim (${cfg.dim}) must be divisible by attn_groups (${cfg.attn_groups}).`);
    if (choices.touch_encoder === 'transformer' && cfg.dim % cfg.touch_encoder.heads) {
      out.push(`dim (${cfg.dim}) must be divisible by the transformer heads (${cfg.touch_encoder.heads}).`);
    }
    if (choices.touch_encoder === 'cnn1d' && cfg.touch_encoder.kernel % 2 === 0) out.push('The conv kernel must be odd.');
    if (choices.matcher === 'latent_head' && cfg.dim <= 3 * cfg.matcher.tokenizer.n_points) {
      out.push(`latent_head needs dim > 3 x n_points (${3 * cfg.matcher.tokenizer.n_points}).`);
    }
    let prev = nPoints;
    for (const s of cfg.target_encoder.stages || []) {
      if (!(s > 0 && s <= prev)) out.push(`Target encoder stages must shrink: ${s} after ${prev}.`);
      prev = s;
    }
    return out;
  }

  const api = { clone, merge, setPath, getPath, flatten, choicesWith, composeRaw, resolve, compose, groupKeys,
    withoutGroup, changes, exportYaml, commandLine, checks, scalar, emitTree };
  root.TF = root.TF || {};
  root.TF.compose = api;
  if (typeof module === 'object' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
