// Architecture figure: fixed-size SVG (W x HT) in the style of the original figure. arch.js lays HTML cards with
// the editable hyperparameters over the module shapes at the coordinates of BOX.
(function (root) {
  'use strict';
  const W = 1500, HT = 910;
  // HTML card positions (x, y, w, h) over the drawing.
  const BOX = {
    stage: [44, 20, 292, 172], sim: [364, 20, 312, 172], scale: [1240, 250, 240, 84],
    'loss.contrastive': [730, 26, 150, 58], 'loss.confidence': [900, 26, 150, 58],
    'loss.chamfer': [1070, 26, 150, 58], 'loss.ppo': [1240, 26, 150, 58],
    target: [178, 258, 134, 140], obs: [46, 424, 266, 48], touchpts: [46, 732, 266, 36],
    target_encoder: [354, 288, 164, 88], touch_encoder: [354, 516, 164, 128],
    'matcher.confidence': [646, 290, 218, 140], 'matcher.embed': [646, 500, 218, 128], policy: [938, 298, 146, 406],
  };
  const PILL_X = { contrastive: 805, confidence: 975, chamfer: 1145, ppo: 1315 };
  // Gold loss arrows: vertical lane x, then the path into the module.
  const LANE = {
    target_encoder: [436, 'V262'], touch_encoder: [540, 'V530 H529'], 'matcher.confidence': [755, 'V279'],
    'matcher.embed': [888, 'V520 H877'], policy: [1010, 'V287'],
  };

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const txt = (x, y, s, cls = 't', extra = '') => `<text x="${x}" y="${y}" class="${cls}" ${extra}>${esc(s)}</text>`;
  const edge = (d, cls = 'edge') => `<path d="${d}" class="${cls}" marker-end="url(#ah-${cls.includes('loss') ? 'gold' : 'ink'})"/>`;
  const trap = (x0, x1, y0t, y0b, y1t, y1b, cls) => `<path d="M${x0},${y0t} L${x1},${y1t} L${x1},${y1b} L${x0},${y0b} Z" class="${cls}"/>`;

  function rng(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32); }

  // An L-shaped block's surface points, isometric projection, coloured by height.
  function targetCloud(x0, y0) {
    const r = rng(7), out = [];
    for (let i = 0; i < 260; i++) {
      let p;
      do { p = [r(), r(), r()]; } while (p[0] > 0.45 && p[2] > 0.45);  // L-shape: notch out one corner
      const f = Math.floor(r() * 3); p[f] = r() < 0.5 ? 0 : (f !== 1 && p[f === 0 ? 2 : 0] > 0.45 ? 0.45 : 1);
      const sx = x0 + 40 + (p[0] - p[1]) * 48, sy = y0 + 70 + (p[0] + p[1]) * 22 - p[2] * 62;
      out.push(`<circle cx="${sx.toFixed(1)}" cy="${sy.toFixed(1)}" r="1.6" class="c${1 + Math.min(2, Math.floor(p[2] * 3))}"/>`);
    }
    return out.join('');
  }

  function traces(rows, x0, y0, w) {
    const r = rng(3);
    return rows.map((label, i) => {
      const y = y0 + i * 30;
      let lines = '';
      for (let k = 0; k < 3; k++) {
        const a = r() * 6, f1 = 1 + r() * 4, f2 = 4 + r() * 8, amp = 3 + r() * 5;
        const pts = Array.from({ length: 41 }, (_, j) => {
          const t = j / 40;
          return `${(x0 + t * w).toFixed(1)},${(y + Math.sin(a + f1 * t * 6) * amp + Math.sin(f2 * t * 6) * 1.5).toFixed(1)}`;
        });
        lines += `<polyline points="${pts.join(' ')}" class="trace"/>`;
      }
      return lines + txt(x0 - 6, y + 4, label, 'ts', 'text-anchor="end"');
    }).join('');
  }

  function contacts(x0, y0) {
    const r = rng(11);
    let s = '';
    for (let i = 0; i < 60; i++) {
      const age = r();
      s += `<circle cx="${(x0 + 20 + r() * 230).toFixed(1)}" cy="${(y0 + 8 + r() * 70).toFixed(1)}" r="${(1.5 + r() * 2.5).toFixed(1)}" class="dot" opacity="${(1 - 0.8 * age).toFixed(2)}"/>`;
    }
    return s;
  }

  // ctx: {cfg, choices, sym, state(module id) -> '' | 'on' | 'frozen', losses: {id: {on, targets}}, explorer,
  //       robotImage, obsRows}
  function figure(ctx) {
    const st = (id) => ctx.state(id);
    const mode = ctx.choices.matcher, s = ctx.sym;
    const parts = [`<svg viewBox="0 0 ${W} ${HT}" width="${W}" height="${HT}" role="img" aria-label="Architecture: target and touch encoders feed the cross-modal matcher, whose confidence and z_diff drive the actor-critic policy; losses from simulator ground truth train the modules of the selected stage.">`,
      `<defs><marker id="ah-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 Z" class="head"/></marker>`,
      `<marker id="ah-gold" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 Z" class="headg"/></marker>`,
      `<clipPath id="robot-clip"><rect x="1240" y="342" width="240" height="180" rx="6"/></clipPath></defs>`,
      `<rect x="0" y="6" width="${W}" height="220" class="band-train"/><rect x="0" y="236" width="${W}" height="668" class="band-deploy"/>`,
      txt(20, 116, 'TRAINING (simulation only)', 'bandl', 'text-anchor="middle" transform="rotate(-90 20 116)"'),
      txt(20, 570, 'DEPLOYED', 'bandl', 'text-anchor="middle" transform="rotate(-90 20 570)"'),
      `<rect x="40" y="16" width="300" height="180" rx="6" class="box"/>`,
      `<rect x="360" y="16" width="320" height="180" rx="6" class="box"/>`];
    // simulator -> losses
    const feeds = [['contrastive', 108, 'touched object id'], ['confidence', 124, 'label c_t'],
      ['chamfer', 140, 'real contact points'], ['ppo', 156, 'reward r_t']];
    for (const [id, y, label] of feeds) {
      const cls = ctx.losses[id].on ? 'edge' : 'edge frozen';
      parts.push(`<g class="${ctx.losses[id].on ? '' : 'frozen'}">${edge(`M680,${y} H${PILL_X[id] - 55} V86`, cls)}${txt(686, y - 3, label, 'ts')}</g>`);
    }
    // gold loss arrows
    Object.entries(ctx.losses).forEach(([id, l], i) => {
      if (!l.on) return;
      for (const m of l.targets) {
        const [lane, tail] = LANE[m], yl = 202 + 6 * i;
        parts.push(edge(`M${PILL_X[id]},84 V${yl} H${lane} ${tail}`, 'loss'));
      }
    });
    for (const [id, x] of Object.entries(PILL_X)) {
      parts.push(`<rect x="${x - 75}" y="26" width="150" height="58" rx="29" class="lpill${ctx.losses[id].on ? '' : ' off'}"/>`);
    }
    // privileged state to the critic (training only)
    parts.push(edge('M680,180 H1105 V312 H1092', 'edge train'), txt(940, 176, 's_priv (P=' + s.P + '): critic only', 'ts'));
    // target query
    parts.push(`<rect x="40" y="252" width="276" height="150" rx="6" class="box"/>`, targetCloud(40, 262));
    // target encoder + tokens
    parts.push(`<g class="${st('target_encoder')}">${trap(346, 526, 248, 412, 280, 380, `mod m-target ${st('target_encoder')}`)}</g>`);
    parts.push(edge('M316,330 H344'), edge('M526,326 H546'));
    for (let k = 0; k < 5; k++) parts.push(`<rect x="${552 + k * 3}" y="${292 + k * 14}" width="34" height="10" rx="2" class="mod m-target ${st('target_encoder')}"/>`);
    parts.push(txt(548, 384, 'tokens M×dim', 'math'), txt(548, 398, `= ${s.M}×${s.dim}`, 'math'));
    // observation history
    parts.push(`<rect x="40" y="420" width="276" height="290" rx="6" class="box"/>`, traces(ctx.obsRows, 190, 486, 118));
    parts.push(`<g class="${st('touch_encoder')}">${trap(346, 526, 470, 690, 510, 650, `mod m-touch ${st('touch_encoder')}`)}</g>`);
    parts.push(edge('M316,580 H344'), edge('M526,580 H550'));
    const zr = rng(5);
    for (let k = 0; k < 12; k++) parts.push(`<rect x="552" y="${520 + k * 10}" width="16" height="10" class="c${1 + Math.floor(zr() * 3)}" opacity="0.7"/>`);
    parts.push(`<rect x="552" y="520" width="16" height="120" fill="none" class="mod"/>`, txt(560, 512, 'z_touch', 'math', 'text-anchor="middle"'));
    // tactile point cloud
    const ptsUsed = mode === 'points' || mode === 'latent_head';
    parts.push(`<g class="${ptsUsed ? '' : 'frozen'}"><rect x="40" y="728" width="276" height="140" rx="6" class="box"/>${contacts(40, 772)}</g>`);
    // matcher
    parts.push(`<rect x="610" y="244" width="290" height="420" rx="8" class="container"/>`, txt(622, 656, 'Point Transformer cross-modal matcher', 'ts'));
    parts.push(`<g class="${st('matcher.confidence')}">${trap(636, 874, 280, 440, 296, 424, `mod m-match ${st('matcher.confidence')}`)}</g>`);
    parts.push(`<g class="${st('matcher.embed')}">${trap(636, 874, 490, 640, 504, 626, `mod m-match ${st('matcher.embed')}`)}</g>`);
    parts.push(edge('M596,326 H634'), txt(615, 320, 'Q', 'math', 'text-anchor="middle"'));
    parts.push(edge('M755,490 V442'), txt(762, 470, `T×dim = ${s.T}×${s.dim}: K, V`, 'math'));
    if (mode === 'points') {
      parts.push(edge('M316,798 H604 V600 H634'), txt(326, 792, `contacts L×8 = ${s.L}×8`, 'math'));
    } else {
      parts.push(edge('M568,560 H634'), txt(574, 552, 'z_touch', 'math'));
      if (mode === 'latent_head') parts.push(txt(326, 792, 'contacts: Chamfer target only', 'ts'));
    }
    // policy
    parts.push(`<g class="${st('policy')}"><rect x="930" y="288" width="160" height="424" rx="8" class="mod m-policy ${st('policy')}"/></g>`);
    parts.push(edge('M874,330 H928'), txt(900, 322, 'ĉ', 'math', 'text-anchor="middle"'));
    parts.push(edge('M874,400 H928'), txt(900, 392, 'z_diff', 'math', 'text-anchor="middle"'));
    parts.push(edge('M560,640 V680 H928'), txt(760, 675, 'z_touch', 'math'));
    parts.push(edge('M316,702 H928'), txt(840, 697, 'o_t', 'math'));
    // action, controller, robot
    const act = [10, -18, 24, -8, 14, -22, 6];
    act.forEach((a, k) => parts.push(`<rect x="${1106 + k * 6}" y="${a > 0 ? 505 - a : 505}" width="4" height="${Math.abs(a)}" class="dot"/>`));
    parts.push(`<g class="${ctx.explorer ? 'frozen' : ''}">${edge('M1090,505 H1104')}</g>`, edge('M1150,505 H1160'));
    parts.push(txt(1127, 556, 'Δq target', 'math', 'text-anchor="middle"'), txt(1127, 570, `A = ${s.A}`, 'math', 'text-anchor="middle"'));
    if (ctx.explorer) parts.push(txt(1127, 456, 'scripted', 'ts', 'text-anchor="middle"'), txt(1127, 470, 'explorer acts', 'ts', 'text-anchor="middle"'));
    parts.push(`<rect x="1162" y="380" width="60" height="260" rx="6" class="mod m-ctrl"/>`,
      txt(1192, 510, 'Joint impedance (PD) controller', 't', 'text-anchor="middle" transform="rotate(-90 1196 510)"'));
    parts.push(edge('M1222,430 H1238'));
    if (ctx.robotImage) {
      parts.push(`<image href="${esc(ctx.robotImage)}" x="1240" y="342" width="240" height="180" preserveAspectRatio="xMidYMid slice" clip-path="url(#robot-clip)"/>`);
    } else {
      parts.push(`<rect x="1240" y="342" width="240" height="180" rx="6" class="box"/>`, txt(1360, 436, 'no rendered frame available', 'ts', 'text-anchor="middle"'));
    }
    parts.push(txt(1240, 538, 'UR10e + Sharpa Wave over the table', 't'), txt(1240, 552, 'frame of our simulation (scripted explorer)', 'ts'));
    // confidence over an episode (schematic)
    parts.push(txt(1240, 584, 'confidence over one episode (schematic)', 'ts'));
    const bar = (y, segs) => { let x = 1262; return segs.map(([w, c]) => { const r = `<rect x="${x}" y="${y}" width="${w}" height="12" class="${c}"/>`; x += w; return r; }).join(''); };
    parts.push(txt(1240, 600, 'c_t', 'math'), bar(591, [[60, 'm-ctrl'], [50, 'c2'], [108, 'c3']]));
    parts.push(txt(1240, 618, 'ĉ_t', 'math'), bar(609, [[66, 'm-ctrl'], [56, 'c2'], [96, 'c3']]));
    parts.push(txt(1240, 640, 'grey: no contact, orange: another object, green: target', 'ts'));
    // telemetry back to the observation history and the tactile point cloud
    parts.push(edge('M1480,430 H1492 V890 H30 V560 H38'), edge('M30,798 H38'));
    parts.push(txt(760, 884, 'tactile and proprioceptive telemetry, 20 Hz', 'ts', 'text-anchor="middle"'));
    parts.push('</svg>');
    return parts.join('');
  }

  const api = { W, HT, BOX, figure, esc };
  root.TF = root.TF || {};
  root.TF.diagram = api;
  if (typeof module === 'object' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
