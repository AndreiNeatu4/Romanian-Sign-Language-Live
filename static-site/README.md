# Sign Language Recognition — static, zero-server build

A fully client-side version of the recogniser. **No backend, no server inference.**
Everything runs on the visitor's device:

```
webcam → MediaPipe Tasks (Hands + Face + Pose)  [WASM/GPU]
       → feature_extractor.js (216-dim, parity-checked vs Python)
       → 45-frame buffer
       → onnxruntime-web (CNN-BiLSTM-Attention)  [WASM]
       → letter prediction
```

The camera stream **never leaves the browser**. The only network traffic is a
one-time download of the model/runtime assets (then browser-cached).

## Files

| File | Role |
|------|------|
| `index.html` | Page shell (same UI as the server version) |
| `app.js` | Camera loop + MediaPipe + overlay (ES module) |
| `model.js` | onnxruntime-web wrapper + buffering/gating logic |
| `feature_extractor.js` | Port of `web interface/app/feature_extractor.py` |
| `style.css` | Copied from the original UI |
| `assets/model/sign_model.onnx` | Exported CNN-BiLSTM-Attn model (~3.7 MB) |
| `assets/model/class_labels.json` | 30 Romanian-alphabet classes |
| `assets/alphabet/<L>/<L>.mp4` | Reference videos (one per letter) |
| `assets/videos.json` | Letter→video list (replaces the old `/videos` API) |

External libs (MediaPipe Tasks, onnxruntime-web) and the MediaPipe `.task`
models load from jsdelivr / Google's CDN — pinned versions in `app.js` / `model.js`.

## Test locally

A secure context is required for the webcam. `localhost` counts as secure, so a
plain static server works:

```bash
cd static-site
python -m http.server 8080
# open http://localhost:8080
```

Click **Start Camera**. First load fetches ~30–40 MB of models (cached after).

## Deploy to Cloudflare Pages (recommended free host)

Unlimited bandwidth, fast global CDN, automatic HTTPS (needed for the camera).

**Option A — drag & drop**
1. Cloudflare dashboard → Workers & Pages → Create → Pages → *Upload assets*.
2. Upload the **contents of `static-site/`** (not the parent folder).
3. Done — you get a `*.pages.dev` link.

**Option B — git (auto-deploy on push)**
1. Connect the repo in Pages → *Connect to Git*.
2. Build command: *(none)*. Build output directory: `static-site`.
3. Every push redeploys.

`_headers` sets long cache lifetimes for the model and videos.

Netlify / Vercel / GitHub Pages also work (set the publish dir to `static-site`),
but have a ~100 GB/month bandwidth cap vs Cloudflare's unlimited.

## Rebuilding the artifacts

Run from the project root (needs `torch`, `onnx`, `onnxruntime`):

```bash
python static-site/tools/export_onnx.py     # .pth → assets/model/sign_model.onnx
python static-site/tools/build_assets.py     # copy css + videos, write videos.json
```

## Verifying feature parity (after any change to feature_extractor.js)

```bash
python static-site/tools/parity_dump.py      # run Python extractor on random frames
node   static-site/tools/parity_check.mjs    # assert JS matches within 1e-4
```

Current parity: `max |JS − Python| ≈ 1.4e-6`.

## Notes on fidelity

- **Every-frame face/pose.** The live FastAPI server ran face/pose every 3rd
  frame as a speed hack; this build runs all three landmarkers every frame,
  matching the offline training pipeline more closely.
- **Non-mirrored frames.** MediaPipe reads the raw video buffer (the CSS mirror
  only affects display), so handedness labels match training.
- **MediaPipe Solutions vs Tasks.** The browser uses MediaPipe *Tasks*; landmark
  indices are identical to the *Solutions* API used in Python, so the feature
  space is preserved. Tiny per-landmark numeric differences between the two
  runtimes are well within the model's tolerance (98.9% test accuracy).
