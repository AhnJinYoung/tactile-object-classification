// Small SVG line charts with stage boundaries and a crosshair tooltip (results tab).
(function (root) {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';

  function el(tag, attrs, parent) {
    const e = root.document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  }

  function fmt(v) {
    if (v === null || v === undefined || Number.isNaN(v)) return '–';
    const a = Math.abs(v);
    if (a >= 1e6) return `${(v / 1e6).toPrecision(3)}M`;
    if (a >= 1e4) return `${(v / 1e3).toPrecision(3)}k`;
    if (a === 0) return '0';
    if (a < 1e-3) return v.toExponential(1);
    return String(Number(v.toPrecision(3)));
  }

  function ticks(lo, hi, n) {
    const span = hi - lo || Math.abs(hi) || 1;
    const step0 = span / (n + 1), mag = 10 ** Math.floor(Math.log10(step0));  // at most about n + 1 ticks
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0);
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
    return out;
  }

  // series: [{name, color (any CSS colour, e.g. var(--s1)), xs, ys}]; stages: [{name, x0}] in order. Returns a <div class="chart">.
  function lineChart({ title, series, stages = [], legend = false, xLabel = 'env steps' }) {
    const W = 360, H = 190, m = { l: 44, r: 12, t: stages.length > 1 ? 18 : 8, b: 24 };
    const box = root.document.createElement('div');
    box.className = 'chart';
    if (title) { const h = root.document.createElement('h4'); h.textContent = title; box.appendChild(h); }
    const pts = series.flatMap((s) => s.xs.map((x, i) => [x, s.ys[i]]).filter((p) => p[1] !== null && p[0] !== null));
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img',
      'aria-label': `${title || 'chart'} against ${xLabel}` }, box);
    if (!pts.length) { const t = el('text', { x: W / 2, y: H / 2, 'text-anchor': 'middle', class: 'axis' }, svg); t.textContent = 'no values yet'; return box; }
    let [x0, x1] = [Math.min(...pts.map((p) => p[0])), Math.max(...pts.map((p) => p[0]))];
    let [y0, y1] = [Math.min(...pts.map((p) => p[1])), Math.max(...pts.map((p) => p[1]))];
    if (x1 === x0) x1 = x0 + 1;
    if (y1 === y0) { y0 -= Math.abs(y0) * 0.1 || 1; y1 += Math.abs(y1) * 0.1 || 1; }
    const pad = (y1 - y0) * 0.06; y0 -= pad; y1 += pad;
    const X = (x) => m.l + ((x - x0) / (x1 - x0)) * (W - m.l - m.r);
    const Y = (y) => H - m.b - ((y - y0) / (y1 - y0)) * (H - m.t - m.b);
    for (const v of ticks(y0, y1, 4)) {
      el('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: 'grid' }, svg);
      el('text', { x: m.l - 5, y: Y(v) + 3, 'text-anchor': 'end', class: 'axis' }, svg).textContent = fmt(v);
    }
    for (const v of ticks(x0, x1, 3)) {
      el('text', { x: X(v), y: H - 8, 'text-anchor': 'middle', class: 'axis' }, svg).textContent = fmt(v);
    }
    stages.forEach((s, i) => {
      const xa = X(Math.max(s.x0, x0)), xb = i + 1 < stages.length ? X(stages[i + 1].x0) : W - m.r;
      if (i > 0) el('line', { x1: xa, x2: xa, y1: m.t - 4, y2: H - m.b, class: 'stageb' }, svg);
      if (stages.length > 1 && xb - xa > 30) {
        el('text', { x: xa + 3, y: m.t - 6, class: 'stagel' }, svg).textContent = s.name;
      }
    });
    for (const s of series) {
      let d = '', pen = false;
      s.xs.forEach((x, i) => {
        const y = s.ys[i];
        if (y === null || y === undefined) { pen = false; return; }
        d += `${pen ? 'L' : 'M'}${X(x).toFixed(1)},${Y(y).toFixed(1)}`;
        pen = true;
      });
      el('path', { d, fill: 'none', style: `stroke:${s.color}`, 'stroke-width': 2, 'stroke-linejoin': 'round' }, svg);
      const last = s.xs.map((x, i) => [x, s.ys[i]]).filter((p) => p[1] !== null).pop();
      if (last) el('circle', { cx: X(last[0]), cy: Y(last[1]), r: 3, style: `fill:${s.color}` }, svg);
    }
    const cross = el('line', { y1: m.t, y2: H - m.b, class: 'cross', visibility: 'hidden' }, svg);
    const hit = el('rect', { x: m.l, y: 0, width: W - m.l - m.r, height: H, fill: 'transparent', tabindex: 0 }, svg);
    const tip = root.document.getElementById('tip');
    function show(xv, ev) {
      cross.setAttribute('x1', X(xv)); cross.setAttribute('x2', X(xv)); cross.setAttribute('visibility', 'visible');
      tip.replaceChildren();
      const head = root.document.createElement('div');
      head.textContent = `${xLabel} ${fmt(xv)}`;
      tip.appendChild(head);
      for (const s of series) {
        let best = -1;
        s.xs.forEach((x, i) => { if (s.ys[i] !== null && (best < 0 || Math.abs(x - xv) < Math.abs(s.xs[best] - xv))) best = i; });
        if (best < 0) continue;
        const row = root.document.createElement('div');
        const k = root.document.createElement('span'); k.className = 'k'; k.style.background = s.color;
        const b = root.document.createElement('b'); b.textContent = fmt(s.ys[best]);
        row.append(k, b, ` ${s.name}`);
        tip.appendChild(row);
      }
      tip.hidden = false;
      const r = ev ? { x: ev.clientX, y: ev.clientY } : (() => { const bb = svg.getBoundingClientRect(); return { x: bb.left + bb.width / 2, y: bb.top }; })();
      tip.style.left = `${Math.min(r.x + 12, root.innerWidth - 290)}px`;
      tip.style.top = `${r.y + 12}px`;
    }
    const hide = () => { cross.setAttribute('visibility', 'hidden'); tip.hidden = true; };
    const xs = [...new Set(pts.map((p) => p[0]))].sort((a, b) => a - b);
    let cur = xs.length - 1;
    hit.addEventListener('pointermove', (ev) => {
      const bb = svg.getBoundingClientRect();
      const xv = x0 + ((ev.clientX - bb.left) * (W / bb.width) - m.l) / (W - m.l - m.r) * (x1 - x0);
      cur = xs.reduce((bi, x, i) => (Math.abs(x - xv) < Math.abs(xs[bi] - xv) ? i : bi), 0);
      show(xs[cur], ev);
    });
    hit.addEventListener('pointerleave', hide);
    hit.addEventListener('blur', hide);
    hit.addEventListener('focus', () => show(xs[cur]));
    hit.addEventListener('keydown', (ev) => {
      if (ev.key !== 'ArrowLeft' && ev.key !== 'ArrowRight') return;
      cur = Math.max(0, Math.min(xs.length - 1, cur + (ev.key === 'ArrowRight' ? 1 : -1)));
      show(xs[cur]);
      ev.preventDefault();
    });
    if (legend) {
      const lg = root.document.createElement('div');
      lg.className = 'legend';
      for (const s of series) {
        const it = root.document.createElement('span');
        const k = root.document.createElement('i'); k.style.background = s.color;
        it.append(k, s.name);
        lg.appendChild(it);
      }
      box.appendChild(lg);
    }
    return box;
  }

  const api = { lineChart, fmt, ticks };
  root.TF = root.TF || {};
  root.TF.charts = api;
  if (typeof module === 'object' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
