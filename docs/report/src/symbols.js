// Symbol table and step specs of tacfind/report/modules.yaml, evaluated in the page.
// Mirrors tacfind/report/specs.py (same grammar: numbers, names, dotted config paths, [i], + - * //, ( ),
// min max len at); tests/js/symbols.test.js checks both give the same numbers for every preset.
(function (root) {
  'use strict';

  function lookup(cfg, path) {
    let node = cfg;
    for (const key of path.split('.')) {
      if (node === null || typeof node !== 'object' || !(key in node)) throw new Error(`no config key ${path}`);
      node = node[key];
    }
    return node;
  }

  const FUNCS = {
    min: (...a) => Math.min(...a), max: (...a) => Math.max(...a), len: (s) => s.length,
    at: (s, i, d) => (i >= 0 && i < s.length ? s[i] : d),
  };

  function tokenize(src) {
    const re = /\s*(\d+|[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*|\/\/|[-+*()[\],])/y;
    const out = [];
    let m;
    re.lastIndex = 0;
    while (re.lastIndex < src.length && (m = re.exec(src))) out.push(m[1]);
    if (re.lastIndex < src.trimEnd().length) throw new Error(`bad expression: ${src}`);
    return out;
  }

  // Evaluate one expression; names are symbols (except `skip`) or config paths.
  function evaluate(expr, cfg, symbols, skip) {
    if (typeof expr === 'number') return expr;
    const tk = tokenize(String(expr));
    let i = 0;
    const peek = () => tk[i], next = () => tk[i++];
    const expect = (t) => { if (next() !== t) throw new Error(`expected ${t} in ${expr}`); };
    function primary() {
      const t = next();
      if (t === '(') { const v = sum(); expect(')'); return v; }
      if (/^\d+$/.test(t)) return Number(t);
      if (peek() === '(') {
        next();
        const args = [];
        while (peek() !== ')') { args.push(sum()); if (peek() === ',') next(); }
        next();
        if (!(t in FUNCS)) throw new Error(`unknown function ${t}`);
        return FUNCS[t](...args);
      }
      if (t in symbols && t !== skip) return symbols[t];
      return lookup(cfg, t);
    }
    function postfix() {
      let v = primary();
      while (peek() === '[') { next(); const k = sum(); expect(']'); v = v[k < 0 ? v.length + k : k]; }
      return v;
    }
    function unary() { if (peek() === '-') { next(); return -unary(); } return postfix(); }
    function product() {
      let v = unary();
      while (peek() === '*' || peek() === '//') v = next() === '*' ? v * unary() : Math.floor(v / unary());
      return v;
    }
    function sum() {
      let v = product();
      while (peek() === '+' || peek() === '-') v = next() === '+' ? v + product() : v - product();
      return v;
    }
    const v = sum();
    if (i !== tk.length) throw new Error(`trailing tokens in ${expr}`);
    return v;
  }

  // Numbers of every symbol that applies (B stays symbolic); a symbol that cannot be evaluated is left out.
  function symbolValues(specs, cfg, choices) {
    const values = {};
    for (const [key, sym] of Object.entries(specs.symbols)) {
      let expr = sym.expr;
      if (expr && typeof expr === 'object') expr = expr[choices[expr.by]];
      if (expr === null || expr === undefined) continue;
      try { values[key] = evaluate(expr, cfg, values, key); } catch (e) { /* not applicable to this config */ }
    }
    return values;
  }

  function activeModules(specs, choices) {
    const out = {};
    for (const [k, m] of Object.entries(specs.modules)) {
      if (Object.entries(m.when || {}).every(([g, o]) => choices[g] === o)) out[k] = m;
    }
    return out;
  }

  function expandSteps(module, cfg, symbols) {
    const out = [];
    for (const step of module.steps) {
      if (!step.foreach) { out.push(Object.assign({}, step, { symbols })); continue; }
      const loop = step.foreach;
      const n = evaluate(loop.over, cfg, symbols).length;
      for (let i = 0; i < n; i++) {
        const local = Object.assign({}, symbols, { [loop.index]: i });
        for (const [key, ex] of Object.entries(loop.let)) local[key] = evaluate(ex, cfg, local, key);
        const fill = (s) => s.replaceAll('{i}', String(i)).replaceAll('{n}', String(i + 1));
        for (const sub of step.steps) {
          const trace = {};
          for (const [k, v] of Object.entries(sub.trace || {})) trace[k] = typeof v === 'string' ? fill(v) : v;
          out.push(Object.assign({}, sub, { name: fill(sub.name), trace, symbols: local }));
        }
      }
    }
    return out;
  }

  // One shape entry for display: "B", "6", "H=30", "D_o + 2 * dim + 1 = 449".
  function formatDim(d, cfg, symbols) {
    if (d === 'B') return 'B';
    if (typeof d === 'number') return String(d);
    let v;
    try { v = evaluate(d, cfg, symbols); } catch (e) { return `${d}=?`; }
    return /^[A-Za-z_]\w*$/.test(d) ? `${d}=${v}` : `${d} = ${v}`;
  }

  function formatShape(shape, cfg, symbols) {
    return `(${shape.map((d) => formatDim(d, cfg, symbols)).join(', ')})`;
  }

  const api = { lookup, evaluate, symbolValues, activeModules, expandSteps, formatShape, formatDim };
  root.TF = root.TF || {};
  root.TF.symbols = api;
  if (typeof module === 'object' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
