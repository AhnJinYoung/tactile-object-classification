// Objects tab: card order, facts and notes from a small objects.json-like gallery.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { SRC } = require('./reference.js');
const { objectCards, galleryNotes } = require(path.join(SRC, 'gallery.js'));

const obj = (name, split, extra = {}) => ({ name, split, material: 'plastic', mass_g: 37.2, mass_estimate: false,
  dims_cm: [2.04, 3.1, 6.25], box_cm: [2, 3, 6], rest_height_cm: 3.14, sink_mm: 0, rest_speed: 1e-6, images: { top: 'data:x', hand: 'data:y' }, ...extra });
const GALLERY = {
  scale: { grid_cm: 1, major_cm: 5, sheet_cm: 40 }, rest: { drop_mm: 5, settle_s: 5, speed: 0.001 }, hand_gap_cm: 2, n_points: 512,
  hand: { thumb_index_cm: 12.425, length_cm: 23.4, width_cm: 17.76 }, views: ['threequarter', 'side', 'top', 'hand'],
  objects: [obj('plum', 'heldout', { rest_speed: 0.5 }), obj('a_lego_duplo', 'train', { mass_estimate: true, sink_mm: 4.86 }), obj('lemon', 'train')],
};

test('train objects come first, library order within a split', () => {
  assert.deepEqual(objectCards(GALLERY).map((c) => [c.name, c.splitLabel]),
    [['a_lego_duplo', 'train'], ['lemon', 'train'], ['plum', 'held out']]);
});

test('card facts are formatted from the data', () => {
  const [lego, lemon] = objectCards(GALLERY);
  const facts = Object.fromEntries(lego.facts);
  assert.equal(lego.title, 'a lego duplo');
  assert.equal(facts.Mass, '37.2 g (estimate)');
  assert.equal(facts['Size (PCA)'], '2.0 × 3.1 × 6.3 cm');
  assert.equal(facts['Size (scan axes)'], '2.0 × 3.0 × 6.0 cm');
  assert.equal(facts['Height at rest'], '3.1 cm');
  assert.match(lego.warn[0], /dips 4\.9 mm into the table/);
  assert.deepEqual(lemon.warn, []);
  assert.match(objectCards(GALLERY)[2].warn[0], /^Still moving/);
  assert.deepEqual(lemon.images.map((i) => i.view), ['top', 'hand']);
});

test('notes carry the hand reference and the scale', () => {
  const notes = Object.fromEntries(galleryNotes(GALLERY));
  assert.match(notes.Hand, /thumb-to-index fingertip pad distance 12\.4 cm/);
  assert.match(notes.Scale, /lines 1 cm apart \(bold every 5 cm\)/);
  assert.match(notes.Objects, /^3 YCB objects: 2 in training worlds first, then 1 held out/);
});
