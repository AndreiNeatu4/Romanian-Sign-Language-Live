/**
 * Feed the identical landmark inputs captured by parity_dump.py through the
 * browser feature_extractor.js and assert the 216-dim features match Python.
 *
 * Run:  node static-site/tools/parity_check.mjs
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { FrameFeatureExtractor } from '../feature_extractor.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, 'parity_fixture.json'), 'utf-8'));

const toObjs = (arr) => (arr ? arr.map(([x, y, z]) => ({ x, y, z })) : null);

const ext = new FrameFeatureExtractor({ bridgeFrames: 3, useFaceMesh: true });

let maxDiff = 0;
let worst = { frame: -1, idx: -1 };

fixture.frames.forEach((fr, t) => {
  const handLandmarksList = fr.hands.length ? fr.hands.map(toObjs) : null;
  const handednessList = fr.handedness.length ? fr.handedness.map((label) => ({ label })) : null;
  const faceLandmarks = toObjs(fr.face);
  const poseLandmarks = toObjs(fr.pose);

  const feat = ext.process({ handLandmarksList, handednessList, faceLandmarks, poseLandmarks });
  const expected = fixture.expected_features[t];

  if (feat.length !== expected.length) {
    console.error(`Length mismatch at frame ${t}: ${feat.length} vs ${expected.length}`);
    process.exit(1);
  }
  for (let i = 0; i < feat.length; i++) {
    const d = Math.abs(feat[i] - expected[i]);
    if (d > maxDiff) { maxDiff = d; worst = { frame: t, idx: i }; }
  }
});

const TOL = 1e-4;
console.log(`frames checked : ${fixture.frames.length}`);
console.log(`max |JS - Py|  : ${maxDiff.toExponential(3)}  (frame ${worst.frame}, idx ${worst.idx})`);
if (maxDiff < TOL) {
  console.log(`PARITY OK (tol ${TOL})`);
  process.exit(0);
} else {
  console.error(`PARITY FAILED (tol ${TOL})`);
  process.exit(1);
}
