// Objects tab: one card per library object with its renders and sizes, from runs/dev/objects/objects.json
// (tacfind/report/gallery.py) as inlined by tacfind/report/page.py. Nothing here is typed by hand.
(function (root) {
  'use strict';
  const VIEWS = {
    threequarter: 'Three-quarter view (perspective)',
    side: 'Side view (orthographic)',
    top: 'Top view (orthographic)',
    hand: 'Next to the hand, from above (orthographic)',
  };
  const SPLIT_ORDER = { train: 0, heldout: 1 };
  const num = (x, digits = 3) => Number(Number(x).toPrecision(digits));
  const cm = (x) => `${Number(x).toFixed(1)} cm`;
  const dims = (d) => `${d.map((x) => x.toFixed(1)).join(' × ')} cm`;

  // Cards in display order (train first, then held out; library order within a split). gallery: objects.json data.
  function objectCards(gallery) {
    const restSpeed = gallery.rest.speed;
    return gallery.objects.map((o, i) => ({ o, i }))
      .sort((a, b) => (SPLIT_ORDER[a.o.split] - SPLIT_ORDER[b.o.split]) || a.i - b.i)
      .map(({ o }) => {
        const warn = [];
        if (o.sink_mm >= 2) warn.push(`At rest the scan dips ${o.sink_mm.toFixed(1)} mm into the table: its collision hulls do not reach the scan's surface there.`);
        if (o.rest_speed >= restSpeed) warn.push(`Still moving at the end of settling (speed ${num(o.rest_speed, 2)}): the height is a snapshot, not a resting height.`);
        return {
          name: o.name, title: o.name.replace(/_/g, ' '), split: o.split, splitLabel: o.split === 'heldout' ? 'held out' : 'train',
          facts: [
            ['Mass', `${num(o.mass_g)} g${o.mass_estimate ? ' (estimate)' : ''}`],
            ['Size (PCA)', dims(o.dims_cm)],
            ['Size (scan axes)', dims(o.box_cm)],
            ['Height at rest', cm(o.rest_height_cm)],
            ['Material', o.material],
          ],
          warn,
          images: Object.keys(VIEWS).filter((v) => o.images && o.images[v]).map((v) => ({ view: v, label: VIEWS[v], src: o.images[v] })),
        };
      });
  }

  // Notes above the cards: [title, text] from the gallery's recorded parameters.
  function galleryNotes(g) {
    const s = g.scale, h = g.hand, n = g.objects.length;
    const held = g.objects.filter((o) => o.split === 'heldout').length;
    return [
      ['Hand', `Sharpa Wave right hand with every joint at 0 (flat open), measured on the model: thumb-to-index fingertip pad distance ${cm(h.thumb_index_cm)}; ${cm(h.length_cm)} long along the fingers and ${cm(h.width_cm)} wide, thumb included (collision meshes). In the hand view it lies palm down, ${num(g.hand_gap_cm)} cm from the object.`],
      ['Scale', `Every image shows grid paper with lines ${num(s.grid_cm)} cm apart (bold every ${num(s.major_cm)} cm) on the table and on a backdrop behind the object, and a ruler of ${num(s.grid_cm)} cm black and white segments. They are geoms at exact positions in the scene. Side, top and hand views are orthographic, so one scale holds across the whole image; the three-quarter view is in perspective.`],
      ['Size', `PCA: extents of the scan along the principal axes of the object's ${g.n_points}-point target cloud, smallest to largest. For objects without a dominant axis (balls, cubes, the mug) the axes are arbitrary and these extents can exceed the true size (up to the diagonal). Scan axes: the bounding box of the scan in its own file frame (YCB scans are stored centred on their oriented bounding box).`],
      ['Height at rest', `Top of the scan above the table after the object is dropped upright (scan frame, z up) ${num(g.rest.drop_mm)} mm above the table and simulated until it rests (speed below ${g.rest.speed}: m/s and rad/s of its free joint), at most ${num(g.rest.settle_s)} s. Training worlds drop objects at random orientations.`],
      ['Objects', `${n} YCB objects: ${n - held} in training worlds first, then ${held} held out. Scripted lift results per object are not stored by any run, so they are not shown.`],
    ];
  }

  function render(container, data) {
    const doc = root.document;
    const h = (tag, cls, ...kids) => { const e = doc.createElement(tag); if (cls) e.className = cls; e.append(...kids); return e; };
    const g = data.gallery;
    if (!g) {
      container.replaceChildren(h('h2', '', 'Objects'), h('p', 'empty', 'The object gallery has not been rendered yet: run python scripts/report.py --objects.'));
      return;
    }
    const notes = h('dl', 'kv');
    for (const [k, v] of galleryNotes(g)) notes.append(h('dt', '', k), h('dd', '', v));
    const grid = h('div', 'objgrid');
    for (const c of objectCards(g)) {
      const facts = h('dl', 'kv');
      for (const [k, v] of c.facts) facts.append(h('dt', '', k), h('dd', '', v));
      const figs = h('div', 'objviews');
      for (const im of c.images) {
        const img = doc.createElement('img');
        Object.assign(img, { src: im.src, alt: `${c.title}: ${im.label}`, loading: 'lazy', width: 480, height: 360 });
        figs.append(h('figure', '', img, h('figcaption', '', im.label)));
      }
      const card = h('article', 'panel objcard', h('div', 'objhead', h('h3', '', c.title), h('span', `pill sp-${c.split}`, c.splitLabel)), facts,
        ...c.warn.map((w) => h('p', 'warn', w)), figs);
      card.id = `obj-${c.name}`;
      grid.append(card);
    }
    container.replaceChildren(h('h2', '', 'Objects'), h('div', 'panel', notes), grid);
  }

  const api = { objectCards, galleryNotes, render };
  root.TF = root.TF || {};
  root.TF.gallery = api;
  if (typeof module === 'object' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
