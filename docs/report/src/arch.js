// Architecture tab: experiment picker, stage selector, the figure with editable hyperparameters in its blocks,
// the per-module detail view (shapes and computation from tacfind/report/modules.yaml) and the YAML export.
(function (root) {
  'use strict';
  const doc = root.document;
  const C = () => root.TF.compose, Y = () => root.TF.symbols, G = () => root.TF.diagram;
  let D = null;    // page data
  let S = null;    // edit state
  let view = null; // container
  const LOSS_PILLS = { contrastive: 'L_contrastive + L_uniform', confidence: 'L_confidence', chamfer: 'L_Chamfer', ppo: 'L_PPO' };

  function h(tag, attrs, ...kids) {
    const e = doc.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === 'class') e.className = v; else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
      else if (v !== false && v !== null && v !== undefined) e.setAttribute(k, v);
    }
    for (const k of kids.flat()) if (k !== null && k !== undefined && k !== false) e.append(k);
    return e;
  }
  const fid = (path) => `f-${path.replace(/[^\w]/g, '-')}`;
  const quiet = (e) => e.stopPropagation();

  // ------------------------------------------------------------------------------------------------ state --
  function entries() {
    const presets = Object.entries(D.config.presets).map(([k, p]) => ({ key: `preset:${k}`, label: p.name, kind: 'preset',
      choices: p.groups, values: p.values, name: p.name, description: p.description }));
    const runs = D.runs.map((r) => ({ key: `run:${r.id}`, label: `${r.id} (${r.status})`, kind: 'run',
      choices: r.arch.choices, values: r.arch.values, name: r.name, description: r.description }));
    return { presets, runs };
  }

  function load(key) {
    const all = entries(), e = [...all.presets, ...all.runs].find((x) => x.key === key) || all.presets[0];
    const choices = C().choicesWith(D.config, e.choices);
    S = { key: e.key, choices, values: C().clone(e.values || {}), name: e.name, description: e.description || '',
      initial: { choices: C().clone(choices), values: C().clone(e.values || {}) }, stage: null,
      detail: S ? S.detail : 'matcher.confidence', fit: S ? S.fit : true };
  }

  function derived() {
    const { raw, cfg } = C().compose(D.config, S.choices, S.values);
    const sym = Y().symbolValues(D.specs, cfg, S.choices);
    const stages = cfg.train.stages || {};
    if (S.stage && !stages[S.stage]) S.stage = null;
    return { raw, cfg, sym, stages };
  }

  const readsZ = () => S.choices.matcher !== 'points';

  function expandTrain(list) {
    const out = new Set();
    for (const n of list || []) {
      if (n === 'touch_side') { out.add('matcher.embed'); if (readsZ()) out.add('touch_encoder'); }
      else if (n === 'matcher') { out.add('matcher.embed'); out.add('matcher.confidence'); }
      else out.add(n);
    }
    return out;
  }

  // Which modules each loss reaches by gradient (tacfind/train/trainer.py `losses`, networks/model.py `act`).
  function reach(term, detach) {
    const z = readsZ() ? ['touch_encoder'] : [];
    return {
      contrastive: ['matcher.embed', 'target_encoder', ...z], uniform: ['target_encoder'],
      confidence: ['matcher.confidence', 'matcher.embed', 'target_encoder', ...z],
      chamfer: S.choices.matcher === 'latent_head' ? ['matcher.embed', 'touch_encoder'] : [],
      ppo: ['policy', 'touch_encoder', ...(detach ? [] : ['matcher.confidence', 'matcher.embed', 'target_encoder'])],
    }[term] || [];
  }

  function lossState(stages) {
    const out = Object.fromEntries(Object.keys(LOSS_PILLS).map((k) => [k, { on: false, targets: new Set(), stages: [] }]));
    for (const [name, st] of Object.entries(stages)) {
      if (S.stage && name !== S.stage) continue;
      const trained = expandTrain(st.train);
      for (const [term, w] of Object.entries(st.loss || {})) {
        if (!w || (term === 'chamfer' && S.choices.matcher !== 'latent_head')) continue;
        const pill = out[term === 'uniform' ? 'contrastive' : term];
        if (!pill) continue;
        pill.on = true;
        if (!pill.stages.includes(name)) pill.stages.push(name);
        for (const m of reach(term, st.detach_matcher !== false)) if (trained.has(m)) pill.targets.add(m);
      }
    }
    for (const v of Object.values(out)) v.targets = [...v.targets];
    return out;
  }

  // ------------------------------------------------------------------------------------------- controls --
  function edited(path) { return JSON.stringify(S.values[path]) !== JSON.stringify(S.initial.values[path]); }

  function setValue(path, v) { S.values[path] = v; render(); }

  function parse(raw, kind) {
    if (kind === 'list') {
      const xs = raw.split(/[,\s]+/).filter(Boolean).map(Number);
      return xs.length && xs.every((x) => Number.isInteger(x) && x > 0) ? xs : null;
    }
    const x = Number(raw);
    if (raw.trim() === '' || !Number.isFinite(x)) return null;
    return kind === 'int' ? (Number.isInteger(x) ? x : null) : x;
  }

  function field(cfg, path, label, kind = 'int') {
    const v = C().getPath(cfg, path);
    if (v === undefined) return null;
    const input = h('input', { id: fid(path), type: kind === 'list' ? 'text' : 'number', step: kind === 'float' ? 'any' : '1',
      value: Array.isArray(v) ? v.join(', ') : v, class: `${kind === 'list' ? 'wide ' : ''}${edited(path) ? 'edited' : ''}`,
      title: path, 'aria-label': `${label} (${path})`, onclick: quiet, onkeydown: quiet });
    input.addEventListener('change', () => {
      const x = parse(input.value, kind);
      if (x === null) { input.value = Array.isArray(v) ? v.join(', ') : v; return; }
      setValue(path, x);
    });
    return h('label', { class: 'f' }, h('span', {}, label), input);
  }

  function groupSelect(group, label) {
    const sel = h('select', { id: fid(`group.${group}`), class: S.choices[group] !== S.initial.choices[group] ? 'edited wide' : 'wide',
      'aria-label': `${label} (config group ${group})`, onclick: quiet, onkeydown: quiet },
    Object.keys(D.config.groups[group].options).map((o) => h('option', { value: o, selected: o === S.choices[group] ? '' : false }, o)));
    sel.addEventListener('change', () => {
      S.choices[group] = sel.value;
      S.values = C().withoutGroup(D.config, S.values, group);
      render();
    });
    return h('label', { class: 'f' }, h('span', {}, label), sel);
  }

  function select(cfg, path, label, options) {
    const v = C().getPath(cfg, path);
    const sel = h('select', { id: fid(path), class: edited(path) ? 'edited wide' : 'wide', 'aria-label': `${label} (${path})`,
      onclick: quiet, onkeydown: quiet }, options.map((o) => h('option', { value: String(o), selected: o === v ? '' : false }, String(o))));
    sel.addEventListener('change', () => setValue(path, options.find((o) => String(o) === sel.value)));
    return h('label', { class: 'f' }, h('span', {}, label), sel);
  }

  // --------------------------------------------------------------------------------------------- cards --
  function card(id, kids, opts = {}) {
    const [x, y, w, hh] = G().BOX[id];
    const attrs = { class: `card ${opts.cls || ''}`, style: `left:${x}px;top:${y}px;width:${w}px;height:${hh}px` };
    if (opts.detail) {
      Object.assign(attrs, { role: 'button', tabindex: 0, 'aria-label': `Details of ${opts.label || id}`,
        onclick: () => { S.detail = opts.detail; render(); doc.getElementById('detail')?.scrollIntoView({ block: 'nearest' }); },
        onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); S.detail = opts.detail; render(); } } });
      attrs.class += ' mod-card';
    }
    return h('div', attrs, kids);
  }

  function cards(d, losses, trained) {
    const { cfg, sym, stages } = d;
    const cls = (m) => (trained ? (trained.has(m) ? '' : 'frozen') : '');
    const st = S.stage ? stages[S.stage] : null;
    const out = [];
    const stageKids = [h('div', { class: 'ttl' }, st ? `Stage: ${S.stage}` : 'Training scheme'), groupSelect('train', 'scheme')];
    if (st) {
      const p = `train.stages.${S.stage}`;
      stageKids.push(h('div', { class: 'sub' }, `acts: ${st.actor === 'explorer' ? 'scripted explorer' : 'policy'}; trains: ${[...expandTrain(st.train)].join(', ')}`),
        h('div', { class: 'fields' }, field(cfg, `${p}.iters`, 'iters'), field(cfg, `${p}.lr`, 'lr', 'float'),
          st.epochs !== undefined ? field(cfg, `${p}.epochs`, 'epochs') : null,
          ...Object.keys(st.loss || {}).map((t) => field(cfg, `${p}.loss.${t}`, `w ${t}`, 'float')),
          st.detach_matcher !== undefined ? select(cfg, `${p}.detach_matcher`, 'detach', [true, false]) : null));
    } else {
      stageKids.push(h('div', { class: 'stl' }, Object.entries(stages).flatMap(([n, s]) => [h('span', {}, n),
        h('span', { class: 'muted' }, s.actor), h('span', {}, `${s.iters} it`)])),
      h('div', { class: 'sub' }, 'Pick a stage above to edit it and see what trains.'));
    }
    out.push(card('stage', stageKids));
    out.push(card('sim', [h('div', { class: 'ttl' }, 'Simulator ground truth'),
      h('div', { class: 'sub' }, 'MuJoCo Warp, all worlds batched on one GPU'),
      field(cfg, 'n_envs', 'worlds'), select(cfg, 'env.label', 'confidence label', ['coverage', 'binary']),
      field(cfg, 'rollout_steps', 'steps / iteration'),
      h('div', { class: 'sub' }, cfg.env.label === 'coverage' ? `c_t = touched share of the target surface (last H steps) / ${cfg.env.coverage.full_at}, clipped to 1` : 'c_t = 1 while a fingertip touches the target')],
    { detail: 'sim', label: 'the simulator ground truth' }));
    for (const [id, label] of Object.entries(LOSS_PILLS)) {
      const l = losses[id];
      const why = id === 'chamfer' && S.choices.matcher !== 'latent_head' ? 'only with latent_head' : l.on ? (S.stage ? 'on' : l.stages.join(', ')) : 'off in this stage';
      out.push(card(`loss.${id}`, [h('div', { class: 'ttl' }, label), h('div', { class: 'sub' }, why)],
        { cls: `pill-card${l.on ? '' : ' off'}`, detail: `loss.${id}`, label }));
    }
    out.push(card('scale', [h('div', { class: 'row' }, groupSelect('scale', 'scale')),
      h('div', { class: 'row' }, field(cfg, 'dim', 'dim'), field(cfg, 'depth', 'depth')),
      h('div', { class: 'row' }, field(cfg, 'attn_groups', 'G'), field(cfg, 'attn_rel_dim', 'R'))]));
    out.push(card('target', [h('div', { class: 'ttl' }, 'Target point cloud'), h('div', { class: 'sub' }, 'the query: which object to find'),
      field(cfg, 'env.objects.n_points', 'points N'), h('div', { class: 'math mono' }, `N × 6 = ${sym.N} × [xyz, normal]`)],
    { detail: 'target', label: 'the target point cloud' }));
    out.push(card('obs', [h('div', { class: 'ttl' }, 'Observation history o(t−H+1 … t)'),
      h('div', { class: 'row' }, field(cfg, 'history', 'H'), h('span', { class: 'mono' }, `× D_o = ${sym.D_o}`))],
    { detail: 'obs', label: 'the observation history' }));
    out.push(card('touchpts', [h('div', { class: 'ttl' }, 'Tactile point cloud, last H steps'),
      h('div', { class: 'sub' }, `L = 5H = ${sym.L} contacts × [point, normal, force, age]`)],
    { detail: 'touchpts', label: 'the tactile point cloud' }));
    out.push(card('target_encoder', [h('div', { class: 'ttl' }, 'Target Encoder'), h('div', { class: 'sub' }, 'Point Transformer'),
      field(cfg, 'target_encoder.stages', 'stages', 'list'), field(cfg, 'target_encoder.k', 'kNN k')],
    { cls: cls('target_encoder'), detail: 'target_encoder', label: 'the target encoder' }));
    const te = S.choices.touch_encoder;
    out.push(card('touch_encoder', [h('div', { class: 'ttl' }, 'Spatial-Temporal Encoder'), groupSelect('touch_encoder', 'type'),
      te === 'cnn1d' ? field(cfg, 'touch_encoder.kernel', 'kernel') : null,
      te === 'transformer' ? field(cfg, 'touch_encoder.heads', 'heads') : null,
      h('div', { class: 'sub' }, `→ z_touch (dim = ${sym.dim})`)],
    { cls: cls('touch_encoder'), detail: 'touch_encoder', label: 'the spatial-temporal encoder' }));
    out.push(card('matcher.confidence', [h('div', { class: 'ttl' }, 'matcher.confidence'),
      h('div', { class: 'sub' }, 'grouped vector cross-attention'),
      h('div', {}, `Q = target tokens (M = ${sym.M})`), h('div', {}, `K, V = touch tokens (T = ${sym.T})`),
      h('div', { class: 'sub' }, `G = ${sym.G} groups, relation R = ${sym.R}`), h('div', {}, '→ confidence ĉ ∈ [0, 1], z_diff')],
    { cls: cls('matcher.confidence'), detail: 'matcher.confidence', label: 'matcher.confidence' }));
    const mm = S.choices.matcher, tp = 'matcher.tokenizer';
    out.push(card('matcher.embed', [h('div', { class: 'ttl' }, 'matcher.embed'), groupSelect('matcher', 'touch input'),
      mm === 'points' ? [field(cfg, `${tp}.max_tokens`, 'max tokens'), field(cfg, `${tp}.k`, 'kNN k')] : null,
      mm === 'latent' ? field(cfg, `${tp}.n_tokens`, 'tokens') : null,
      mm === 'latent_head' ? field(cfg, `${tp}.n_points`, 'points K') : null,
      h('div', { class: 'sub' }, '→ touch tokens; e_touch (contrastive)')],
    { cls: cls('matcher.embed'), detail: 'matcher.embed', label: 'matcher.embed' }));
    out.push(card('policy', [h('div', { class: 'ttl' }, 'Exploration & Retrieval Policy'), h('div', { class: 'sub' }, 'asymmetric actor-critic'),
      field(cfg, 'policy_hidden', 'hidden', 'list'), field(cfg, 'policy.init_log_std', 'init log σ', 'float'),
      h('div', {}, `actor: o_t, z_touch, z_diff, ĉ → μ, σ (A = ${sym.A})`),
      h('div', {}, `critic: the same + s_priv (P = ${sym.P}) → V`),
      h('div', { class: 'sub' }, 'trained by PPO; s_priv exists only in simulation')],
    { cls: cls('policy'), detail: 'policy', label: 'the policy' }));
    return out;
  }

  // ---------------------------------------------------------------------------------------------- detail --
  function moduleSpecId(id) {
    if (id === 'touch_encoder') return `touch_encoder.${S.choices.touch_encoder}`;
    if (id === 'matcher.embed') return `matcher.embed.${S.choices.matcher}`;
    return id;
  }

  function usedSymbols(shapes, symbols) {
    const names = new Set();
    for (const sh of shapes) for (const dd of sh) String(dd).replace(/[A-Za-z_]\w*/g, (n) => { if (n in symbols) names.add(n); return n; });
    return [...names];
  }

  function shapeCell(shape, cfg, sy) { return h('td', { class: 'shape' }, Y().formatShape(shape, cfg, sy)); }

  function detail(d) {
    const { cfg, sym } = d;
    const box = h('div', { class: 'panel detail', id: 'detail' });
    const id = S.detail;
    if (id.startsWith('loss.')) {
      const terms = id === 'loss.contrastive' ? ['contrastive', 'uniform'] : [id.slice(5)];
      for (const t of terms) {
        const l = D.specs.losses[t];
        box.append(h('h3', {}, l.title), h('p', {}, l.text), h('pre', { class: 'code' }, l.formula), h('p', { class: 'muted mono' }, l.source));
      }
      return box;
    }
    if (id === 'obs') {
      const groups = cfg.env.obs_groups;
      box.append(h('h3', {}, 'Observation o_t'), h('p', {}, `The history holds the last H = ${sym.H} observations, oldest first, zero before the episode started. Shape (B, H=${sym.H}, D_o=${sym.D_o}).`),
        h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, 'group (env.obs_groups)'), h('th', {}, 'size'))),
          h('tbody', {}, groups.map((g) => h('tr', {}, h('td', { class: 'mono' }, g), h('td', {}, String(D.specs.obs_groups[g] ?? '?')))),
            h('tr', {}, h('td', {}, h('b', {}, 'D_o')), h('td', {}, h('b', {}, String(groups.reduce((a, g) => a + (D.specs.obs_groups[g] || 0), 0))))))));
      return box;
    }
    if (id === 'touchpts' || id === 'target' || id === 'sim') {
      const text = {
        touchpts: [`Tactile point cloud: (B, L=${sym.L}, 8)`, `Every fingertip contact of the last H = ${sym.H} steps (5 fingertips per step): contact point in the robot base frame (m), outward normal of the touched surface, force magnitude (N) and age in [0, 1] (0 = now). Contacts weaker than ${cfg.env.force_threshold} N read as no contact. The points matcher keeps T = ${sym.T ?? '-'} of them; with latent_head they are only the Chamfer target.`],
        target: [`Target point cloud: (B, N=${sym.N}, 6)`, `${sym.N} surface points of the target object with outward normals, in the object frame. The library has ${cfg.env.objects.n_objects} objects of ${cfg.env.objects.families.length} families; the trainer encodes the whole library once per update.`],
        sim: ['Simulator ground truth', `Only available in simulation: the confidence label c_t (${cfg.env.label}), the library id of the touched object (contrastive target), the reward, and the privileged state s_priv (P = ${sym.P}: per object position, 6-D rotation and target flag) that only the critic sees. ${cfg.n_envs} worlds run batched on one GPU.`],
      }[id];
      box.append(h('h3', {}, text[0]), h('p', {}, text[1]));
      return box;
    }
    const m = D.specs.modules[moduleSpecId(id)];
    if (!m) { box.append(h('p', { class: 'muted' }, 'No description for this module.')); return box; }
    const steps = Y().expandSteps(m, cfg, sym);
    const trainedIn = Object.entries(d.stages).filter(([, s]) => expandTrain(s.train).has(id)).map(([n]) => n);
    box.append(h('h3', {}, m.title), h('p', {}, m.purpose),
      h('dl', { class: 'kv' }, h('dt', {}, 'class'), h('dd', { class: 'mono' }, m.class),
        h('dt', {}, 'config'), h('dd', { class: 'mono' }, `${m.config}${S.choices[m.config.split('.')[0]] ? ` (${S.choices[m.config.split('.')[0]]})` : ''}`),
        h('dt', {}, 'trains in'), h('dd', {}, trainedIn.join(', ') || 'no stage of this scheme')));
    const io = (title, rows) => [h('h3', {}, title), h('div', { class: 'scroll-x' }, h('table', {}, h('tbody', {},
      rows.map((r) => h('tr', {}, h('td', { class: 'mono' }, r.name), shapeCell(r.shape, cfg, sym), h('td', {}, r.means || ''))))))];
    box.append(...io('Inputs', m.inputs), ...io('Outputs', m.outputs));
    box.append(h('h3', {}, 'Computation'), h('div', { class: 'scroll-x' }, h('table', {},
      h('thead', {}, h('tr', {}, ['', 'step', 'formula', 'shape after'].map((t) => h('th', {}, t)))),
      h('tbody', {}, steps.map((s, i) => h('tr', {}, h('td', {}, String(i + 1)),
        h('td', {}, h('b', {}, s.name), s.text ? h('div', { class: 'muted' }, s.text) : null),
        h('td', { class: 'formula' }, s.formula || ''), shapeCell(s.shape, cfg, s.symbols)))))));
    const names = usedSymbols([...m.inputs, ...m.outputs].map((r) => r.shape).concat(steps.map((s) => s.shape)), D.specs.symbols);
    box.append(h('h3', {}, 'Symbols'), h('dl', { class: 'kv' }, names.flatMap((n) => [h('dt', { class: 'mono' }, `${n}${sym[n] !== undefined ? ` = ${sym[n]}` : ''}`),
      h('dd', {}, D.specs.symbols[n].means)])));
    const section = C().getPath(cfg, m.config);
    if (section && typeof section === 'object') box.append(h('h3', {}, `Config (${m.config}, resolved)`), h('pre', { class: 'code' }, C().emitTree(section, '')));
    return box;
  }

  // ---------------------------------------------------------------------------------------------- export --
  function copyButton(label, getText, target) {
    const status = h('span', { class: 'muted', role: 'status' });
    const btn = h('button', { type: 'button', onclick: () => {
      const done = () => { status.textContent = 'Copied.'; };
      const fallback = () => { target.focus(); if (target.select) target.select(); status.textContent = 'Selected: press Ctrl+C or Cmd+C to copy.'; };
      try { root.navigator.clipboard.writeText(getText()).then(done, fallback); } catch (e) { fallback(); }
    } }, label);
    return [btn, status];
  }

  function exportPanel(d) {
    const yaml = C().exportYaml(D.config, S), cmd = C().commandLine(D.config, S);
    const ta = h('textarea', { class: 'code', id: 'exp-yaml', readonly: '', spellcheck: 'false', 'aria-label': 'Experiment file' });
    ta.value = yaml;
    const pre = h('pre', { class: 'code', id: 'exp-cmd', tabindex: 0 }, cmd);
    const name = h('input', { id: 'exp-name', value: S.name, 'aria-label': 'Experiment name' });
    name.addEventListener('change', () => { S.name = name.value.trim() || 'custom'; render(); });
    const desc = h('input', { id: 'exp-desc', value: S.description, 'aria-label': 'One-line description' });
    desc.addEventListener('change', () => { S.description = desc.value.trim(); render(); });
    const warns = C().checks(d.cfg, S.choices, d.sym.N);
    const changed = JSON.stringify([S.choices, S.values]) !== JSON.stringify([S.initial.choices, S.initial.values]);
    if (changed && S.key.startsWith('preset:') && S.name === D.config.presets[S.key.slice(7)].name) {
      warns.push('The name is still the preset\'s: give the edited experiment its own name so it does not replace the preset file.');
    }
    return h('div', { class: 'panel', id: 'export' }, h('h3', {}, 'Export as an experiment file'),
      h('p', { class: 'muted' }, 'Edited values are outlined in gold. The file states only what differs from the root config.'),
      h('div', { class: 'form' }, h('label', { for: 'exp-name' }, 'name'), name, h('label', { for: 'exp-desc' }, 'description'), desc),
      warns.length ? h('div', { class: 'warn' }, warns.map((w) => h('div', {}, w))) : null,
      h('p', {}, 'Save as ', h('code', {}, `configs/experiment/${S.name}.yaml`), ', run with ', h('code', {}, `python scripts/train.py +experiment=${S.name}`)),
      ta, h('div', { class: 'row' }, ...copyButton('Copy file', () => ta.value, ta),
        h('button', { type: 'button', onclick: () => { const keep = S.stage; load(S.key); S.stage = keep; render(); } }, 'Reset to the selected experiment')),
      h('p', {}, 'The same as command-line overrides:'), pre, h('div', { class: 'row' }, ...copyButton('Copy command', () => cmd, pre)));
  }

  // ---------------------------------------------------------------------------------------------- render --
  function toolbar(d) {
    const all = entries();
    const pick = h('select', { id: 'arch-exp', 'aria-label': 'Experiment' },
      h('optgroup', { label: 'Presets (configs/experiment)' }, all.presets.map((e) => h('option', { value: e.key, selected: e.key === S.key ? '' : false }, e.label))),
      all.runs.length ? h('optgroup', { label: 'Runs' }, all.runs.map((e) => h('option', { value: e.key, selected: e.key === S.key ? '' : false }, e.label))) : null);
    pick.addEventListener('change', () => { load(pick.value); render(); });
    const btn = (name, label) => h('button', { type: 'button', id: `stage-${name || 'all'}`, 'aria-pressed': String(S.stage === name),
      onclick: () => { S.stage = name; render(); } }, label);
    const st = S.stage ? d.stages[S.stage] : null;
    const fit = h('button', { type: 'button', id: 'arch-fit', 'aria-pressed': String(S.fit), onclick: () => { S.fit = !S.fit; render(); } }, 'Fit to width');
    return h('div', {}, h('div', { class: 'toolbar' }, h('label', { for: 'arch-exp' }, 'Experiment'), pick, fit),
      h('div', { class: 'toolbar' }, h('span', { class: 'muted' }, 'Stage'), h('div', { class: 'stages' }, btn(null, 'All stages'),
        Object.keys(d.stages).map((n) => btn(n, n))),
      h('span', { class: 'actor' }, st ? `${st.actor === 'explorer' ? 'The scripted explorer acts' : 'The policy acts'}; ${[...expandTrain(st.train)].join(', ')} train; the rest is frozen.`
        : 'Gold dashed arrows: losses and the modules they train.')),
      h('p', { class: 'muted' }, S.description));
  }

  function obsRows(cfg) {
    const g = cfg.env.obs_groups, rows = [];
    for (let i = 0; i < g.length; i += 2) rows.push(g.slice(i, i + 2).join(', '));
    return rows.slice(0, 7);
  }

  function fitFigure() {
    const box = doc.getElementById('figbox'), sizer = doc.getElementById('figsizer'), fig = doc.getElementById('stagefig');
    if (!box || !fig) return;
    const s = S.fit ? Math.min(1, Math.max(0.6, (box.clientWidth - 2) / G().W)) : 1;
    fig.style.transform = `scale(${s})`;
    sizer.style.width = `${G().W * s}px`;
    sizer.style.height = `${G().HT * s}px`;
  }

  function render() {
    const focus = doc.activeElement && doc.activeElement.id;
    const d = derived();
    const losses = lossState(d.stages);
    const trained = S.stage ? expandTrain(d.stages[S.stage].train) : null;
    const state = (m) => (trained ? (trained.has(m) ? 'on' : 'frozen') : '');
    const fig = h('div', { class: 'stagefig', id: 'stagefig', style: `width:${G().W}px;height:${G().HT}px` });
    fig.innerHTML = G().figure({ cfg: d.cfg, choices: S.choices, sym: d.sym, state, losses,
      explorer: !!(S.stage && d.stages[S.stage].actor === 'explorer'), robotImage: D.robotImage, obsRows: obsRows(d.cfg) });
    fig.append(...cards(d, losses, trained));
    const figbox = h('div', { class: 'figbox', id: 'figbox' }, h('div', { id: 'figsizer', style: 'position:relative;overflow:hidden' }, fig));
    view.replaceChildren(h('h2', {}, 'Architecture'), toolbar(d), figbox,
      h('p', { class: 'muted' }, 'Click a block for its inputs, outputs and computation. Values in the blocks are editable; the export below collects the changes.'),
      h('div', { class: 'grid2' }, detail(d), exportPanel(d)));
    fitFigure();
    if (focus) doc.getElementById(focus)?.focus();
  }

  function init(container, data) {
    D = data; view = container;
    load(`preset:${Object.keys(D.config.presets).find((k) => k.includes('baseline')) || Object.keys(D.config.presets)[0]}`);
    render();
    root.addEventListener('resize', fitFigure);
  }

  root.TF = root.TF || {};
  root.TF.arch = { init, render: () => render() };
})(typeof window !== 'undefined' ? window : globalThis);
