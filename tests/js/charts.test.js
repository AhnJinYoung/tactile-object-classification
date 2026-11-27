// Chart helpers and the results tab's grouping of metrics by stage (no DOM needed).
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { SRC } = require('./reference.js');
const charts = require(path.join(SRC, 'charts.js'));
const results = require(path.join(SRC, 'results.js'));

test('ticks are round numbers inside the range', () => {
  assert.deepStrictEqual(charts.ticks(0, 1, 4), [0, 0.2, 0.4, 0.6, 0.8, 1]);
  assert.deepStrictEqual(charts.ticks(-0.05, 1.05, 3), [0, 0.5, 1]);
  assert.deepStrictEqual(charts.ticks(1200, 98000, 3), [25000, 50000, 75000]);
});

test('number format', () => {
  assert.equal(charts.fmt(1234567), '1.23M');
  assert.equal(charts.fmt(51234), '51.2k');
  assert.equal(charts.fmt(0.12345), '0.123');
  assert.equal(charts.fmt(null), '–');
});

test('metrics group under the stage that logs them; shared ones under "Across stages"', () => {
  const curves = { step: [1, 2, 3, 4], stage: ['a', 'a', 'b', 'b'],
    metrics: { 'loss/x': [1, 0.5, null, null], 'loss/ppo': [null, null, 2, 1], 'perf/sps': [5, 5, 6, 6], 'conf/none': [null, null, null, null] } };
  assert.deepStrictEqual([...results.metricGroups(curves)], [['Stage a', ['loss/x']], ['Stage b', ['loss/ppo']], ['Across stages', ['perf/sps']]]);
  assert.deepStrictEqual(results.stagesOf(curves), [{ name: 'a', x0: 1 }, { name: 'b', x0: 3 }]);
});
