# Setup & Deployment Guide — Romanian Sign Language Recognition

This document covers two separate things:
1. **Running the ML pipeline** (extract data → train → run real-time recognition)
2. **Running the web interface** (browser-based demo, locally or on a server)

---

## Project structure

```
handle keypoints/
├── alphabet/                        # Raw training videos + extracted JPG frames
│   └── A/  B/  C/  ...             # One folder per letter, contains video + frames
│
├── data/
│   ├── alphabet_augmented/          # Augmented .npy sequences (output of extract step)
│   └── alphabet_processed/          # Final dataset.pkl + class_labels.json (training input)
│
├── models/
│   └── alphabet/                    # Trained model weights
│       ├── best_model.pth           # Best checkpoint (used by recognition)
│       ├── final_model.pth          # Last epoch weights
│       ├── training_results.json    # Accuracy, class names, feature sizes
│       ├── confusion_matrix.png
│       └── training_history.png
│
├── data_preparation/
│   ├── extract_gesture_data.py      # Step 1 – extract landmarks from videos
│   ├── extract_augmented_fast.py    # Step 2 – augment the extracted sequences
│   └── prepare_augmented_dataset.py # Step 3 – build dataset.pkl for training
│
├── training/
│   └── train_model.py               # Step 4 – train the CNN-LSTM model
│
├── application/
│   └── realtime_recognition.py      # Step 5 – live webcam inference (standalone)
│
├── web interface/                   # Browser-based demo
│   ├── app/
│   │   ├── main.py                  # FastAPI backend + WebSocket
│   │   ├── model.py                 # Model wrapper used by the web app
│   │   └── static/                  # Frontend (index.html, style.css, app.js)
│   ├── requirements.txt             # Web server Python dependencies
│   └── SERVER_SETUP.md              # This file
│
├── config.py                        # All paths and hyperparameters — edit this first
├── setup/requirements.txt           # ML pipeline Python dependencies (GPU)
└── extract_frames.py                # Utility: split video into JPG frames
```

---

## Part 1 — ML pipeline (training)

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.1 support (tested on RTX 4080)
- CUDA 12.1 toolkit installed: https://developer.nvidia.com/cuda-12-1-0-download-archive

### 1.1 Install dependencies

```bash
cd "handle keypoints"
pip install -r setup/requirements.txt
```

This installs PyTorch 2.3.1 with CUDA 12.1, MediaPipe, OpenCV, scikit-learn, and matplotlib.

### 1.2 Edit config.py

Open `config.py` and verify these paths match your machine:

```python
VIDEO_DIR  = r"C:\Users\Andrei Neatu\Desktop\handle keypoints\alphabet"
OUTPUT_DIR = r"C:\Users\Andrei Neatu\Desktop\handle keypoints\data\alphabet_processed"
MODEL_DIR  = r"C:\Users\Andrei Neatu\Desktop\handle keypoints\models\alphabet"
```

Key training settings in the same file:

| Setting | Default | Notes |
|---|---|---|
| `SEQUENCE_LENGTH` | 30 | Frames per gesture sample |
| `MODEL_TYPE` | `cnn_lstm` | Options: `lstm`, `cnn_lstm`, `transformer` |
| `EPOCHS` | 150 | With early stopping (patience=25) |
| `BATCH_SIZE` | 32 | Increase to 64 if GPU memory allows |
| `EXTRACT_FACE_MESH` | `True` | Extracts 28 face anchor points (60 features) |

### 1.3 Prepare raw videos

Each letter must have its own folder under `alphabet/` with one or more `.mp4` files:

```
alphabet/
├── A/  a.mp4
├── B/  B.mp4
...
└── Ț/  Ț.mp4
```

To extract individual frames as JPGs (optional, for inspection):

```bash
python extract_frames.py
# frames saved as A_0000.jpg, A_0001.jpg ... in each letter's folder
```

### 1.4 Extract landmark sequences

```bash
python data_preparation/extract_gesture_data.py
```

Reads videos from `VIDEO_DIR`, runs MediaPipe Hands + Face Mesh on every frame, normalizes coordinates relative to wrist/face size, and saves `.npy` sequence files to `data/alphabet_augmented/`.

Each frame produces a 186-feature vector:
- 126 features — 2 hands × 21 landmarks × 3 (x, y, z)
- 60 features — 28 face anchor points × 3 (x, y, z)

### 1.5 Build the dataset

```bash
python data_preparation/prepare_augmented_dataset.py
```

Loads all `.npy` files, does a stratified 70/15/15 train/val/test split, and writes:
- `data/alphabet_processed/dataset.pkl` — all splits in one file (used by training)
- `data/alphabet_processed/class_labels.json` — class index → letter name mapping (used by recognition)

### 1.6 Train the model

```bash
python training/train_model.py
```

Trains the CNN-LSTM model on GPU with:
- Mixed precision (AMP) for ~40% faster training on RTX 4080
- `num_workers=4` + `pin_memory=True` for fast data loading
- Early stopping (patience=25 epochs)
- Learning rate reduction on plateau (factor 0.5, patience 5)

Output written to `models/alphabet/`:
- `best_model.pth` — best validation loss checkpoint
- `final_model.pth` — last epoch
- `training_results.json` — test accuracy, class names, feature sizes
- `confusion_matrix.png` / `training_history.png` — visual diagnostics

### 1.7 Run real-time recognition (standalone)

```bash
python application/realtime_recognition.py
```

Opens webcam at 60 fps. Reads `models/alphabet/best_model.pth` and `data/alphabet_processed/class_labels.json`.

Controls:

| Key | Action |
|---|---|
| `q` | Quit |
| `r` | Reset frame buffer |
| `+` / `-` | Increase / decrease motion sensitivity |
| `s` / `a` | Increase / decrease static gesture confidence threshold |

The recognizer uses two modes:
- **Motion-based** — prediction shown when motion detected + confidence ≥ 70%
- **Static** — prediction shown without motion if confidence ≥ 90%

---

## Part 2 — Web interface

The web interface is a separate FastAPI app with a browser frontend. It uses the **same trained model** but serves it over WebSocket so any device with a browser and camera can use it.

### 2a — Local development (no Docker)

```bash
cd "handle keypoints/web interface"
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

The web app expects the trained model at:
- `models/alphabet/best_model.pth`
- `data/alphabet_processed/class_labels.json`

> Note: `web interface/requirements.txt` uses a CPU-only PyTorch build. If you want GPU inference in the web app, replace the torch line with the CUDA version from `setup/requirements.txt`.

### 2b — Docker (production / server)

#### Hardware requirements

| Resource | Minimum | Recommended |
|---|---|---|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8 GB |
| Disk | 10 GB free | 20 GB free |
| GPU | not required | NVIDIA (optional) |
| Network | broadband | static IP or DDNS for internet access |

#### Install Docker

**Windows (Docker Desktop)**

1. Download from https://www.docker.com/products/docker-desktop
2. Run installer, enable WSL 2 when prompted
3. Verify:
   ```
   docker --version
   docker compose version
   ```

**Linux (Ubuntu/Debian)**

```bash
sudo apt remove docker docker-engine docker.io containerd runc
sudo apt update && sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt update && sudo apt install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER   # log out and back in after this
```

#### Build and run

```bash
cd "handle keypoints/web interface"
docker compose build          # first time: several minutes (downloads PyTorch etc.)
docker compose up -d          # start in background
docker compose logs -f        # watch logs
```

App is at **http://localhost:8000**.

#### Open firewall port 8000

**Windows:**
1. Press `Win+R` → `wf.msc`
2. Inbound Rules → New Rule → Port → TCP → 8000 → Allow

**Linux:**
```bash
sudo ufw allow 8000/tcp && sudo ufw reload
```

#### Make it reachable from the internet

**Option A — Router port forwarding**
1. Find server's local IP: `ipconfig` (Windows) or `ip a` (Linux)
2. Log into router admin panel → Port Forwarding
3. Forward external port 8000 → server's local IP:8000, TCP
4. Public URL: `http://YOUR-PUBLIC-IP:8000`

**Option B — Dynamic DNS**
Sign up at https://www.duckdns.org or https://www.noip.com, create a hostname, install their auto-update client.
Result: `http://mysignapp.duckdns.org:8000`

#### HTTPS / SSL (required for camera on non-localhost)

Browsers block camera access (`getUserMedia`) on plain HTTP for non-localhost origins.

```bash
sudo apt install -y nginx certbot python3-certbot-nginx
```

Create `/etc/nginx/sites-available/sign-language`:

```nginx
server {
    listen 80;
    server_name YOUR-DOMAIN;

    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade    $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host       $host;
        proxy_read_timeout 86400;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/sign-language /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d YOUR-DOMAIN
```

App is now at `https://YOUR-DOMAIN` with auto-renewing SSL.

**Windows alternative:** Use Caddy (https://caddyserver.com) — handles HTTPS automatically with one config line.

#### Auto-start on boot

**Linux:**
```bash
sudo systemctl enable docker
# docker-compose.yml already has restart: unless-stopped
```

**Windows:**
Docker Desktop → Settings → General → Start Docker Desktop when you log in.
Then add a Task Scheduler task on login:
```
cmd /c "cd /d C:\Users\Andrei Neatu\Desktop\handle keypoints\web interface && docker compose up -d"
```

#### Management commands

```bash
docker compose logs -f          # live logs
docker compose down             # stop
docker compose up -d --build    # rebuild after code changes
docker compose restart          # restart without rebuild
docker stats                    # resource usage
```

#### Security checklist

- [ ] Set up HTTPS before sharing the link externally (step above)
- [ ] Change default port 8000 if exposed directly without a reverse proxy
- [ ] Do not expose the Docker socket to the internet
- [ ] Keep Docker and the host OS updated
- [ ] Add HTTP Basic Auth in Nginx to restrict access if needed

---

## Quick-start summary

### Train a model from scratch

```
1. Edit config.py — verify your paths
2. Place videos in alphabet/<LETTER>/<letter>.mp4
3. pip install -r setup/requirements.txt
4. python data_preparation/extract_gesture_data.py
5. python data_preparation/prepare_augmented_dataset.py
6. python training/train_model.py
```

### Run the trained model

```
# Standalone (webcam window):
python application/realtime_recognition.py

# Browser-based:
cd "web interface"
uvicorn app.main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```
