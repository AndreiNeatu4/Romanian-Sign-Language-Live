/**
 * ONNX inference wrapper (onnxruntime-web). Mirrors the gating logic of
 * web interface/app/model.py GestureRecognizer, but runs 100% in the browser.
 *
 * Expects window.ort to be loaded (classic <script> in index.html).
 */
import {
  FrameFeatureExtractor,
  TOTAL_FEATURES_SIZE,
  OFF_MASK,
} from './feature_extractor.js';

const SEQUENCE_LENGTH = 45; // matches config.SEQUENCE_LENGTH / training
const MODEL_URL = './assets/model/sign_model.onnx';
const LABELS_URL = './assets/model/class_labels.json';
const ORT_WASM_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';

// MediaPipe hand skeleton connections (for the overlay).
export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

// Face anchor indices drawn on the overlay (same 20 as the feature extractor).
export const FACE_DRAW_INDICES = [
  1, 6, 133, 33, 468, 362, 263, 473, 107, 70,
  336, 300, 61, 291, 13, 14, 152, 10, 123, 352,
];

export class SignRecognizer {
  constructor() {
    this.session = null;
    this.classNames = [];
    this.modelLoaded = false;
    this.ffe = new FrameFeatureExtractor({ bridgeFrames: 3, useFaceMesh: true });
    this._buffer = [];                 // ring of Float32Array(216)
    this._wasHandVisible = false;
    this._lastPrediction = null;       // {prediction, confidence, top_predictions}
    this._inputBuf = new Float32Array(SEQUENCE_LENGTH * TOTAL_FEATURES_SIZE);
  }

  async load() {
    const ort = window.ort;
    if (!ort) throw new Error('onnxruntime-web (window.ort) not loaded');

    // Single-threaded wasm: avoids the COOP/COEP cross-origin-isolation
    // requirement of SharedArrayBuffer, so it runs on any static host as-is.
    ort.env.wasm.wasmPaths = ORT_WASM_BASE;
    ort.env.wasm.numThreads = 1;

    const labels = await (await fetch(LABELS_URL)).json();
    this.classNames = labels.classes || labels;

    this.session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['wasm'], // LSTM is fully supported on the wasm EP
      graphOptimizationLevel: 'all',
    });
    this.modelLoaded = true;
  }

  reset() {
    this._buffer = [];
    this._wasHandVisible = false;
    this._lastPrediction = null;
    this.ffe.reset();
  }

  /**
   * Drop a duplicate detection when MediaPipe labels the same physical hand as
   * both Left and Right (wrist landmark 0 nearly coincident). Mirror of
   * model.py _deduplicate_hands.
   */
  static _dedupHands(hands, handedness, threshold = 0.12) {
    if (!hands || hands.length < 2) return [hands, handedness];
    const w0 = hands[0][0], w1 = hands[1][0];
    const dx = w0.x - w1.x, dy = w0.y - w1.y;
    if (Math.sqrt(dx * dx + dy * dy) < threshold) {
      const s0 = handedness?.[0]?.score ?? 0;
      const s1 = handedness?.[1]?.score ?? 0;
      const keep = s0 >= s1 ? 0 : 1;
      return [[hands[keep]], handedness ? [handedness[keep]] : null];
    }
    return [hands, handedness];
  }

  /**
   * Process one frame's landmarks and return a result object matching the
   * shape app.js expects (same fields the old WebSocket server returned).
   *
   * @param {Object} lm
   *   lm.hands       Array<Array<{x,y,z}>>  detected hands (0..2)
   *   lm.handedness  Array<{label,score}>   parallel to hands
   *   lm.face        Array<{x,y,z}> | null  478 face landmarks
   *   lm.pose        Array<{x,y,z}> | null  33 pose landmarks
   */
  async process({ hands, handedness, face, pose }) {
    let hd = handedness;
    [hands, hd] = SignRecognizer._dedupHands(hands || null, handedness || null);

    const feat = this.ffe.process({
      handLandmarksList: hands && hands.length ? hands : null,
      handednessList: hd && hd.length ? hd.map((h) => ({ label: h.label })) : null,
      faceLandmarks: face || null,
      poseLandmarks: pose || null,
    });

    const rawHands = (hands || []).map((h) => h.map((p) => [p.x, p.y]));
    const rawFace = face ? FACE_DRAW_INDICES.map((i) => (face[i] ? [face[i].x, face[i].y] : [0, 0])) : [];
    const hasHand = rawHands.length > 0;

    // Pre-fill the buffer on hand re-entry (matches GestureRecognizer).
    if (hasHand && !this._wasHandVisible) {
      this._buffer = new Array(SEQUENCE_LENGTH).fill(feat);
    } else {
      this._buffer.push(feat);
      if (this._buffer.length > SEQUENCE_LENGTH) this._buffer.shift();
    }
    this._wasHandVisible = hasHand;

    const result = {
      buffer_fill: this._buffer.length,
      buffer_max: SEQUENCE_LENGTH,
      hands: rawHands,
      face: rawFace,
      hand_connections: HAND_CONNECTIONS,
      prediction: null,
      confidence: 0.0,
      top_predictions: [],
      frozen: false,
    };

    if (hasHand && this._buffer.length === SEQUENCE_LENGTH && this.modelLoaded) {
      // Flatten buffer (45,216) into the reusable input tensor.
      for (let t = 0; t < SEQUENCE_LENGTH; t++) {
        this._inputBuf.set(this._buffer[t], t * TOTAL_FEATURES_SIZE);
      }
      const ort = window.ort;
      const tensor = new ort.Tensor('float32', this._inputBuf, [1, SEQUENCE_LENGTH, TOTAL_FEATURES_SIZE]);
      const out = await this.session.run({ [this.session.inputNames[0]]: tensor });
      const probs = out[this.session.outputNames[0]].data; // already softmaxed in ONNX

      const idxs = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]).slice(0, 5);
      result.prediction = this.classNames[idxs[0]];
      result.confidence = probs[idxs[0]];
      result.top_predictions = idxs.map((i) => ({ label: this.classNames[i], confidence: probs[i] }));

      // ── Diagnostic log (throttled). Disable with window.SIGN_DEBUG = false. ──
      this._dbg = (this._dbg || 0) + 1;
      if (window.SIGN_DEBUG !== false && this._dbg % 15 === 0) {
        const slot0 = feat[OFF_MASK].toFixed(1);     // 1=left-slot hand present
        const slot1 = feat[OFF_MASK + 1].toFixed(1); // 1=right-slot hand present
        const poseSeen = pose ? 'pose✓' : 'pose✗';
        const faceSeen = face ? 'face✓' : 'face✗';
        console.log(
          `[sign] hands=${(hands || []).length} labels=${(hd || []).map((h) => h.label).join(',')} ` +
          `slots=[${slot0},${slot1}] ${poseSeen} ${faceSeen} | ` +
          `top: ${result.top_predictions.map((p) => `${p.label} ${(p.confidence * 100).toFixed(0)}%`).join('  ')}`
        );
      }
      this._lastPrediction = {
        prediction: result.prediction,
        confidence: result.confidence,
        top_predictions: result.top_predictions,
      };
    } else if (!hasHand && this._lastPrediction) {
      // Freeze on last good prediction while no hand is visible.
      result.prediction = this._lastPrediction.prediction;
      result.confidence = this._lastPrediction.confidence;
      result.top_predictions = this._lastPrediction.top_predictions;
      result.frozen = true;
    }

    return result;
  }
}

export default SignRecognizer;
