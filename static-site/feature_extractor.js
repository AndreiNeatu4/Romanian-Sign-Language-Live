/**
 * In-browser port of web interface/app/feature_extractor.py.
 *
 * Produces the SAME 216-dim per-frame feature vector the model was trained on,
 * so inference can run fully client-side. Keep this in exact numeric parity
 * with the Python original — see tools/parity_check.mjs.
 *
 * Per-frame layout (TOTAL_FEATURES = 216):
 *   [  0:126] hand local        2 hands x 21 lm x 3 (wrist-centered, scale-norm)
 *   [126:186] face anchors      20 x 3 (nose-centered, face-height-norm)
 *   [186:192] wrist global      2 hands x 3, body-anchored
 *   [192:198] fingertip global  2 hands x 3, body-anchored (index fingertip)
 *   [198:204] wrist velocity    first derivative of wrist global
 *   [204:210] wrist accel        second derivative of wrist global
 *   [210:212] hand-present mask  1 = detected, 0.5 = bridged, 0 = absent
 *   [212:216] reserved          kept zero
 *
 * Landmark inputs use MediaPipe Tasks Vision shapes:
 *   handLandmarksList : Array<Array<{x,y,z}>>   (21 points per hand)
 *   handednessList    : Array<{label:'Left'|'Right'}>  parallel to hands
 *   faceLandmarks     : Array<{x,y,z}> | null    (478 points)
 *   poseLandmarks     : Array<{x,y,z}> | null    (33 points)
 */

export const NUM_HAND_LANDMARKS = 21;
export const HAND_LOCAL_SIZE = 126; // 2 * 21 * 3
export const NUM_FACE_ANCHORS = 20;
export const FACE_FEATURES_SIZE = NUM_FACE_ANCHORS * 3; // 60
export const WRIST_GLOBAL_SIZE = 6;
export const FINGERTIP_GLOBAL_SIZE = 6;
export const WRIST_VELOCITY_SIZE = 6;
export const WRIST_ACCEL_SIZE = 6;
export const MASK_RESERVED_SIZE = 6;

export const OFF_HAND_LOCAL = 0;
export const OFF_FACE = OFF_HAND_LOCAL + HAND_LOCAL_SIZE;
export const OFF_WRIST_GLOBAL = OFF_FACE + FACE_FEATURES_SIZE;
export const OFF_FINGERTIP_GLOBAL = OFF_WRIST_GLOBAL + WRIST_GLOBAL_SIZE;
export const OFF_WRIST_VELOCITY = OFF_FINGERTIP_GLOBAL + FINGERTIP_GLOBAL_SIZE;
export const OFF_WRIST_ACCEL = OFF_WRIST_VELOCITY + WRIST_VELOCITY_SIZE;
export const OFF_MASK = OFF_WRIST_ACCEL + WRIST_ACCEL_SIZE;
export const TOTAL_FEATURES_SIZE = OFF_MASK + MASK_RESERVED_SIZE; // 216

// Ordered face anchors — order matters (chin/forehead indices derived below).
export const DEFAULT_FACE_ANCHORS = [
  ['nose_tip', 1], ['nose_bridge', 6],
  ['left_eye_inner', 133], ['left_eye_outer', 33], ['left_eye_center', 468],
  ['right_eye_inner', 362], ['right_eye_outer', 263], ['right_eye_center', 473],
  ['left_eyebrow_inner', 107], ['left_eyebrow_outer', 70],
  ['right_eyebrow_inner', 336], ['right_eyebrow_outer', 300],
  ['mouth_left', 61], ['mouth_right', 291], ['mouth_top', 13], ['mouth_bottom', 14],
  ['chin', 152], ['forehead', 10], ['left_cheek', 123], ['right_cheek', 352],
];

const POSE_LEFT_SHOULDER = 11;
const POSE_RIGHT_SHOULDER = 12;
const POSE_LEFT_HIP = 23;
const POSE_RIGHT_HIP = 24;
const INDEX_FINGERTIP_LM = 8;

const norm3 = (ax, ay, az) => Math.sqrt(ax * ax + ay * ay + az * az);

export class FrameFeatureExtractor {
  constructor({ bridgeFrames = 3, useFaceMesh = true } = {}) {
    this.bridgeFrames = bridgeFrames;
    this.useFaceMesh = useFaceMesh;
    this._anchorKeys = DEFAULT_FACE_ANCHORS.map((a) => a[0]);
    this._chinIdx = this._anchorKeys.indexOf('chin');
    this._foreheadIdx = this._anchorKeys.indexOf('forehead');
    this.reset();
  }

  reset() {
    // Slot 0 = LEFT hand, slot 1 = RIGHT hand. Mirror of _HandSlotState.
    this._slots = [this._newSlot(), this._newSlot()];
  }

  _newSlot() {
    return {
      lastLocal: null,            // Float32Array(63)
      lastWristGlobal: null,      // Float32Array(3)
      lastFingertipGlobal: null,  // Float32Array(3)
      lastVelocity: new Float32Array(3),
      framesSinceSeen: 1e6,
    };
  }

  // shoulder midpoint origin + shoulder-to-hip scale
  _bodyAnchor(pose) {
    if (!pose) return null;
    const ls = pose[POSE_LEFT_SHOULDER], rs = pose[POSE_RIGHT_SHOULDER];
    const lh = pose[POSE_LEFT_HIP], rh = pose[POSE_RIGHT_HIP];
    if (!ls || !rs || !lh || !rh) return null;
    const sx = (ls.x + rs.x) * 0.5, sy = (ls.y + rs.y) * 0.5, sz = (ls.z + rs.z) * 0.5;
    const hx = (lh.x + rh.x) * 0.5, hy = (lh.y + rh.y) * 0.5, hz = (lh.z + rh.z) * 0.5;
    const torso = norm3(sx - hx, sy - hy, sz - hz) + 1e-6;
    return { origin: [sx, sy, sz], scale: torso };
  }

  // nose + face-height fallback when pose is unavailable
  _faceAnchor(face) {
    if (!face) return null;
    const nose = face[1], chin = face[152], forehead = face[10];
    const faceHeight = norm3(forehead.x - chin.x, forehead.y - chin.y, forehead.z - chin.z) + 1e-6;
    return {
      origin: [nose.x, nose.y + faceHeight * 1.5, nose.z],
      scale: faceHeight * 3.0,
    };
  }

  // [leftHandOrNull, rightHandOrNull]
  _assignHandSlots(hands, handedness) {
    const slots = [null, null];
    if (!hands || hands.length === 0) return slots;

    if (handedness && handedness.length === hands.length) {
      for (let i = 0; i < hands.length; i++) {
        const idx = handedness[i].label === 'Left' ? 0 : 1;
        if (slots[idx] === null) {
          slots[idx] = hands[i];
        } else {
          const other = 1 - idx;
          if (slots[other] === null) slots[other] = hands[i];
        }
      }
    } else {
      const sorted = hands.slice(0, 2).sort((a, b) => a[0].x - b[0].x);
      for (let i = 0; i < Math.min(2, sorted.length); i++) slots[i] = sorted[i];
    }
    return slots;
  }

  // local (63, wrist-centered/scaled), wrist xyz, fingertip xyz
  _localHandshape(hand) {
    const wx = hand[0].x, wy = hand[0].y, wz = hand[0].z;
    const tip = hand[INDEX_FINGERTIP_LM];
    let maxAbs = 0;
    const centered = new Float32Array(63);
    for (let i = 0; i < NUM_HAND_LANDMARKS; i++) {
      const cx = hand[i].x - wx, cy = hand[i].y - wy, cz = hand[i].z - wz;
      centered[i * 3] = cx; centered[i * 3 + 1] = cy; centered[i * 3 + 2] = cz;
      const m = Math.max(Math.abs(cx), Math.abs(cy), Math.abs(cz));
      if (m > maxAbs) maxAbs = m;
    }
    const scale = maxAbs + 1e-7;
    const local = new Float32Array(63);
    for (let i = 0; i < 63; i++) local[i] = centered[i] / scale;
    return {
      local,
      wrist: [wx, wy, wz],
      fingertip: [tip.x, tip.y, tip.z],
    };
  }

  // 60-dim face features (nose-centered, face-height-normalized)
  _faceFeatures(face) {
    const out = new Float32Array(FACE_FEATURES_SIZE);
    if (!face) return out;
    const n = face.length;
    const anchors = new Array(NUM_FACE_ANCHORS);
    for (let a = 0; a < NUM_FACE_ANCHORS; a++) {
      let idx = DEFAULT_FACE_ANCHORS[a][1];
      if (idx >= n) {
        const name = DEFAULT_FACE_ANCHORS[a][0];
        if (name.includes('left_eye_center')) idx = 159;
        else if (name.includes('right_eye_center')) idx = 386;
        else idx = 0;
      }
      const lm = face[idx];
      anchors[a] = [lm.x, lm.y, lm.z];
    }
    const nose = anchors[0];
    const fh = anchors[this._foreheadIdx], ch = anchors[this._chinIdx];
    const faceHeight = norm3(fh[0] - ch[0], fh[1] - ch[1], fh[2] - ch[2]) + 1e-6;
    for (let a = 0; a < NUM_FACE_ANCHORS; a++) {
      out[a * 3] = (anchors[a][0] - nose[0]) / faceHeight;
      out[a * 3 + 1] = (anchors[a][1] - nose[1]) / faceHeight;
      out[a * 3 + 2] = (anchors[a][2] - nose[2]) / faceHeight;
    }
    return out;
  }

  /**
   * Build the per-frame 216-dim feature vector. Call once per frame in order.
   * @returns {Float32Array} length 216
   */
  process({ handLandmarksList = null, handednessList = null, faceLandmarks = null, poseLandmarks = null } = {}) {
    const feat = new Float32Array(TOTAL_FEATURES_SIZE);

    // Block B: face
    if (this.useFaceMesh && faceLandmarks) {
      const ff = this._faceFeatures(faceLandmarks);
      feat.set(ff, OFF_FACE);
    }

    // Body anchor (pose preferred, face fallback)
    let anchor = this._bodyAnchor(poseLandmarks);
    if (!anchor) anchor = this._faceAnchor(faceLandmarks);
    const origin = anchor ? anchor.origin : null;
    const aScale = anchor ? anchor.scale : null;

    const slots = this._assignHandSlots(handLandmarksList, handednessList);

    for (let slotIdx = 0; slotIdx < 2; slotIdx++) {
      const hand = slots[slotIdx];
      const state = this._slots[slotIdx];
      const localOff = OFF_HAND_LOCAL + slotIdx * 63;
      const wristOff = OFF_WRIST_GLOBAL + slotIdx * 3;
      const tipOff = OFF_FINGERTIP_GLOBAL + slotIdx * 3;
      const velOff = OFF_WRIST_VELOCITY + slotIdx * 3;
      const accOff = OFF_WRIST_ACCEL + slotIdx * 3;
      const maskOff = OFF_MASK + slotIdx;

      if (hand) {
        const { local, wrist, fingertip } = this._localHandshape(hand);
        let wg, tg;
        if (origin) {
          wg = [(wrist[0] - origin[0]) / aScale, (wrist[1] - origin[1]) / aScale, (wrist[2] - origin[2]) / aScale];
          tg = [(fingertip[0] - origin[0]) / aScale, (fingertip[1] - origin[1]) / aScale, (fingertip[2] - origin[2]) / aScale];
        } else {
          wg = [wrist[0] - 0.5, wrist[1] - 0.5, wrist[2] - 0.5];
          tg = [fingertip[0] - 0.5, fingertip[1] - 0.5, fingertip[2] - 0.5];
        }

        let velocity, accel;
        if (state.lastWristGlobal && state.framesSinceSeen <= this.bridgeFrames + 1) {
          velocity = [wg[0] - state.lastWristGlobal[0], wg[1] - state.lastWristGlobal[1], wg[2] - state.lastWristGlobal[2]];
          accel = [velocity[0] - state.lastVelocity[0], velocity[1] - state.lastVelocity[1], velocity[2] - state.lastVelocity[2]];
        } else {
          velocity = [0, 0, 0];
          accel = [0, 0, 0];
        }

        feat.set(local, localOff);
        feat[wristOff] = wg[0]; feat[wristOff + 1] = wg[1]; feat[wristOff + 2] = wg[2];
        feat[tipOff] = tg[0]; feat[tipOff + 1] = tg[1]; feat[tipOff + 2] = tg[2];
        feat[velOff] = velocity[0]; feat[velOff + 1] = velocity[1]; feat[velOff + 2] = velocity[2];
        feat[accOff] = accel[0]; feat[accOff + 1] = accel[1]; feat[accOff + 2] = accel[2];
        feat[maskOff] = 1.0;

        state.lastLocal = local;
        state.lastWristGlobal = Float32Array.from(wg);
        state.lastFingertipGlobal = Float32Array.from(tg);
        state.lastVelocity = Float32Array.from(velocity);
        state.framesSinceSeen = 0;
      } else {
        state.framesSinceSeen += 1;
        if (state.lastLocal && state.lastWristGlobal && state.framesSinceSeen <= this.bridgeFrames) {
          feat.set(state.lastLocal, localOff);
          feat[wristOff] = state.lastWristGlobal[0];
          feat[wristOff + 1] = state.lastWristGlobal[1];
          feat[wristOff + 2] = state.lastWristGlobal[2];
          const tg = state.lastFingertipGlobal || state.lastWristGlobal;
          feat[tipOff] = tg[0]; feat[tipOff + 1] = tg[1]; feat[tipOff + 2] = tg[2];
          // velocity/accel stay 0 during bridging
          feat[maskOff] = 0.5;
        } else {
          feat[maskOff] = 0.0;
        }
      }
    }

    return feat;
  }
}

export default FrameFeatureExtractor;
