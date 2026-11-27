// Results tab: table of experiments (runs and planned presets), one run's curves and videos, and a compare chart.
(function (root) {
  'use strict';
  const doc = root.document;
  const COLORS = ['--s1', '--s2', '--s3', '--s4', '--s5', '--s6', '--s7', '--s8'];
  const ORDER = ['loss', 'acc', 'conf', 'rollout', 'ppo', 'eval', 'perf'];
  const KEY = [['rollout/success_rate', 'success', 'pct'], ['rollout/wrong_pick_rate', 'wrong pick', 'pct'],
    ['acc/contrastive_top1', 'top-1', 'pct'], ['conf/mae', 'conf MAE', 'num']];

  function h(tag, attrs, ...kids) {
    const e = doc.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === 'class') e.className = v; else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v);
    }
    for (const k of kids.flat()) if (k !== null && k !== undefined && k !== false) e.append(k);
    return e;
  }
  const color = (i) => `var(${COLORS[i % COLORS.length]})`;  // follows the theme
  const fmt = (v, kind) => (v === undefined || v === null ? '–'
    : kind === 'pct' ? `${(100 * v).toFixed(1)} %` : root.TF.charts.fmt(v));

  function statusPill(s) { return h('span', { class: `pill st-${s}` }, s); }

  function stagesOf(curves) {
    const out = [];
    curves.stage.forEach((s, i) => {
      if (!out.length || out[out.length - 1].name !== s) out.push({ name: s, x0: curves.step[i] ?? i });
    });
    return out;
  }

  function metricGroups(curves) {
    const groups = new Map();
    const names = Object.keys(curves.metrics).sort((a, b) => {
      const ia = ORDER.indexOf(a.split('/')[0]), ib = ORDER.indexOf(b.split('/')[0]);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
    });
    for (const s of stagesOf(curves)) groups.set(`Stage ${s.name}`, []);
    groups.set('Across stages', []);
    for (const name of names) {
      const st = new Set(curves.metrics[name].map((v, i) => (v === null ? null : curves.stage[i])).filter((s) => s !== null));
      if (st.size) groups.get(st.size === 1 ? `Stage ${[...st][0]}` : 'Across stages').push(name);
    }
    return new Map([...groups].filter(([, v]) => v.length));
  }

  function videoFigure(src, caption) {
    return h('figure', {}, h('video', { src, controls: '', loop: '', muted: '', playsinline: '', preload: 'metadata' }),
      h('figcaption', {}, caption));
  }

  function runDetail(run) {
    const box = h('div', { class: 'panel', id: 'run-detail' });
    box.append(h('h3', {}, run.name, ' ', statusPill(run.status)));
    if (run.description) box.append(h('p', {}, run.description));
    if (run.note) box.append(h('p', { class: 'note' }, run.note));
    const meta = [['run directory', run.id], ['started', run.start_time], ['ended', run.end_time],
      ['GPU', run.gpu], ['worlds', run.n_envs], ['env steps', run.env_steps && run.env_steps.toLocaleString('en')],
      ['peak GPU memory', run.peak_gpu_mem_gb && `${run.peak_gpu_mem_gb.toFixed(2)} GB`],
      ['parameters', run.params && run.params.total && run.params.total.toLocaleString('en')]];
    box.append(h('dl', { class: 'kv' }, meta.filter((m) => m[1]).flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, String(v))])));
    const c = run.curves;
    if (!c.step.length) box.append(h('p', { class: 'muted' }, 'No metrics logged yet.'));
    const stages = stagesOf(c);
    for (const [group, names] of metricGroups(c)) {
      box.append(h('h3', {}, group));
      const grid = h('div', { class: 'charts' });
      for (const name of names) {
        grid.append(root.TF.charts.lineChart({ title: name, stages,
          series: [{ name, color: color(0), xs: c.step.map((s, i) => s ?? i), ys: c.metrics[name] }] }));
      }
      box.append(grid);
    }
    if (run.videos.length) {
      box.append(h('h3', {}, 'Rollout videos (latest per stage)'));
      box.append(h('div', { class: 'videos' }, run.videos.map((v) => videoFigure(`videos/${run.id}/${v.file}`,
        `Stage ${v.stage}${v.iter >= 0 ? `, iteration ${v.iter}` : ''}. One world; the bars show the ground-truth and predicted confidence.`))));
    }
    return box;
  }

  function compareView(runs, state, rerender) {
    const withCurves = runs.filter((r) => r.curves.step.length);
    if (!withCurves.length) return null;
    const names = [...new Set(withCurves.flatMap((r) => Object.keys(r.curves.metrics)))].sort();
    if (!names.includes(state.metric)) state.metric = names.includes('rollout/success_rate') ? 'rollout/success_rate' : names[0];
    if (!state.compare) state.compare = new Set(withCurves.slice(-4).map((r) => r.id));
    const sel = h('select', { id: 'cmp-metric', onchange: (e) => { state.metric = e.target.value; rerender(); } },
      names.map((n) => h('option', n === state.metric ? { value: n, selected: '' } : { value: n }, n)));
    const boxes = withCurves.map((r) => h('label', {}, h('input', Object.assign({ type: 'checkbox', id: `cmp-${r.id}`,
      onchange: (e) => { e.target.checked ? state.compare.add(r.id) : state.compare.delete(r.id); rerender(); } },
    state.compare.has(r.id) ? { checked: '' } : {})), ` ${r.name}`));
    const chosen = withCurves.filter((r) => state.compare.has(r.id)).slice(0, 8);
    const series = chosen.map((r, i) => ({ name: r.name, color: color(i), xs: r.curves.step.map((s, j) => s ?? j),
      ys: r.curves.metrics[state.metric] || [] }));
    return h('div', { class: 'panel' }, h('div', { class: 'row' }, h('label', { for: 'cmp-metric' }, 'Metric'), sel),
      h('div', { class: 'checks' }, boxes),
      series.length ? root.TF.charts.lineChart({ title: state.metric, series, legend: true })
        : h('p', { class: 'muted' }, 'Select runs to compare.'),
      chosen.length < state.compare.size ? h('p', { class: 'muted' }, 'At most 8 runs are drawn at once.') : null);
  }

  function table(runs, planned, state, rerender) {
    const head = h('tr', {}, ['Experiment', 'Status', 'Differs from the baseline', 'Key metrics (latest)', 'Env steps',
      'GPU', 'Comment'].map((t) => h('th', { scope: 'col' }, t)));
    const rows = runs.map((r) => {
      const select = () => { state.run = r.id; rerender(); doc.getElementById('run-detail')?.scrollIntoView({ block: 'start' }); };
      return h('tr', { class: `run${state.run === r.id ? ' sel' : ''}`, tabindex: 0, onclick: select,
        onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(); } } },
      h('td', {}, h('b', {}, r.name), h('div', { class: 'muted' }, r.description || '')),
      h('td', {}, statusPill(r.status)),
      h('td', {}, r.chips.length ? r.chips.map((c) => h('span', { class: 'chip' }, c)) : h('span', { class: 'muted' }, 'none')),
      h('td', {}, KEY.filter(([k]) => r.key_metrics[k] !== undefined)
        .map(([k, label, kind]) => h('div', { class: 'km' }, `${label} ${fmt(r.key_metrics[k], kind)}`))),
      h('td', { class: 'mono' }, r.env_steps ? r.env_steps.toLocaleString('en') : '–'),
      h('td', {}, r.gpu || '–'), h('td', {}, r.note || ''));
    });
    const plan = planned.map((p) => h('tr', {}, h('td', {}, h('b', {}, p.name), h('div', { class: 'muted' }, p.description)),
      h('td', {}, statusPill('planned')),
      h('td', {}, p.chips.length ? p.chips.map((c) => h('span', { class: 'chip' }, c)) : h('span', { class: 'muted' }, 'none (baseline)')),
      h('td', { class: 'muted' }, '–'), h('td', { class: 'muted' }, '–'), h('td', { class: 'muted' }, '–'), h('td', {}, p.note || '')));
    return h('div', { class: 'scroll-x' }, h('table', {}, h('thead', {}, head), h('tbody', {}, rows, plan)));
  }

  function render(container, data) {
    const state = render.state || (render.state = {});
    const focus = doc.activeElement && doc.activeElement.id;
    const runs = data.runs, rerender = () => render(container, data);
    if (!state.run && runs.length) state.run = (runs.find((r) => r.status === 'running') || runs[runs.length - 1]).id;
    const count = (s) => runs.filter((r) => r.status === s).length;
    const parts = [h('h2', {}, 'Experiments'),
      h('p', { class: 'muted' }, runs.length
        ? `${runs.length} run${runs.length > 1 ? 's' : ''}: ${count('done')} done, ${count('running')} running, ${count('failed')} failed; ${data.planned.length} planned without a run yet.`
        : `${data.planned.length} experiments planned.`)];
    if (!runs.length) {
      parts.push(h('div', { class: 'empty' }, h('p', {}, h('b', {}, 'No training runs yet.'),
        ' They start when GPU resources are available. Every planned experiment is listed below; its curves and rollout videos appear here once a run writes to runs/.')));
    }
    parts.push(table(runs, data.planned, state, rerender));
    const run = runs.find((r) => r.id === state.run);
    if (run) parts.push(h('h2', {}, 'Selected run'), runDetail(run));
    const cmp = compareView(runs, state, rerender);
    if (cmp) parts.push(h('h2', {}, 'Compare runs'), cmp);
    if (data.envCheck) {
      parts.push(h('h2', {}, 'Environment check'), h('div', { class: 'videos' }, videoFigure(data.envCheck.video,
        'Environment check, not a training result: the scripted touch explorer in one simulated world, with no learning. The bars show the simulator\'s confidence label and whether a fingertip touches the target.')));
    }
    container.replaceChildren(...parts);
    if (focus) doc.getElementById(focus)?.focus();
  }

  root.TF = root.TF || {};
  root.TF.results = { render, stagesOf, metricGroups };
  if (typeof module === 'object' && module.exports) module.exports = root.TF.results;
})(typeof window !== 'undefined' ? window : globalThis);
