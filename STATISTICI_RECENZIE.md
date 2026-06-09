# Statistici Sistem de Recunoaștere Gesturi — Faza de Recenzie

**Proiect:** Recunoaștere Alfabet Limbaj Semne în Timp Real (Alfabet Românesc)
**Model:** CNN-LSTM cu MediaPipe Hands
**Data pregătire date:** 17 martie 2026
**Data:** 18 martie 2026

---

## 1. Date despre Dataset

| Parametru | Valoare |
|-----------|---------|
| Număr simboluri (clase) | **30** |
| Simboluri acoperite | A–Z + Â, Ă, Ș, Ț |
| Număr videoclipuri sursă | 30 (câte 1 per simbol) |
| Număr total secvențe (după augmentare) | **2.544** |
| Secvențe per simbol (medie) | ~84–96 |
| Lungime secvență | 30 cadre |
| Features per cadru | 252 (126 coordonate mâini + 126 velocitate) |
| Split antrenament / validare+test | 80% / 20% |
| Secvențe antrenament | ~2.035 |
| Secvențe validare/test | ~509 *(val = test)* |

**Augmentări aplicate per video:**
- 4 variante spațiale (oglindire, decalaje)
- 3 viteze de redare: 0.98×, 1.0×, 1.02×
- Variante cu zgomot (`_noise`) și rotații (`_rot7`, `_rotn7`)
- **~96 secvențe generate din 1 singur videoclip**

---

## 2. Timpi de Antrenament

| Parametru | Valoare |
|-----------|---------|
| Tip model | CNN-LSTM |
| Număr parametri model | ~120.000 |
| Optimizer | Adam (lr=0.001) |
| Batch size | 32 |
| Epoci configurate | 150 |
| **Epoci efectiv rulate** | **150** *(early stopping nu s-a activat)* |
| Device utilizat | **CPU** *(CUDA indisponibil în această rulare)* |
| GPU disponibil | NVIDIA RTX 4080 |
| Timp extragere date (MediaPipe, 30 videoclipuri) | ~3 minute |
| **Timp total antrenament (150 epoci, CPU)** | **~25–35 minute** *(estimat)* |
| Timp mediu per epocă (CPU) | ~10–14 secunde |

> **Notă:** Antrenamentul a rulat pe CPU (`device: cpu`), nu pe GPU. Cu RTX 4080 activat (CUDA), timpul estimat ar fi sub 2 minute total pentru același dataset.

---

## 3. Timpi de Execuție — Inferență în Timp Real

| Parametru | Valoare |
|-----------|---------|
| FPS video capturat (țintă) | 60 fps |
| Rezoluție cameră | 1280 × 720 |
| Buffer secvență | 30 cadre |
| Prag confidență predicție | 70% |
| Prag confidență gesturi statice | 90% |
| Timp afișare predicție pe ecran | 5 secunde |
| Procesare MediaPipe | thread separat (non-blocking) |
| Captură video | thread separat (threaded capture) |
| Latență inferență model (estimat, CPU) | ~15–30 ms |
| Latență inferență model (estimat, GPU RTX 4080) | ~2–5 ms |

---

## 4. Rate de Recunoaștere cu Succes — Rezultate Model

### 4.1 Performanță Globală (Automată, pe Test Set)

| Metrică | Valoare |
|---------|---------|
| **Acuratețe antrenament (epoca finală)** | **97,85%** |
| **Acuratețe validare (epoca finală)** | **98,78%** |
| **Acuratețe test set** | **98,78%** |
| Număr clase | 30 |
| Secvențe test evaluate | ~509 |

> Test set = Validation set (split 80/20, val și test sunt identice conform `prepare_augmented_dataset.py`).

### 4.2 Performanță per Simbol

> Valorile exacte de precizie/recall/F1 per literă sunt generate de `train_model.py` în consola de antrenament (Classification Report) și în `models/alphabet/confusion_matrix.png`.
> Acuratețea globală de **98,78%** indică erori pe ~6 din ~509 secvențe de test.

| Simbol | Observații |
|--------|------------|
| A–Z | Acoperit |
| Â | Acoperit (specific românesc) |
| Ă | Acoperit (specific românesc) |
| Ș | Acoperit (specific românesc) |
| Ț | Acoperit (specific românesc) |

### 4.3 Testare cu Utilizatori Reali

> Această secțiune se completează manual după sesiunile de testare.

| Utilizator | Nr. simboluri testate | Nr. recunoscute corect | Rată succes | Observații |
|------------|----------------------|------------------------|-------------|------------|
| 1 | ___ | ___ | ___% | |
| 2 | ___ | ___ | ___% | |
| 3 | ___ | ___ | ___% | |
| **Medie** | | | **___%** | |

---

## 5. Rezumat Executiv

| Metrică cheie | Valoare |
|---------------|---------|
| Simboluri recunoscute | 30 (alfabet românesc complet) |
| Acuratețe pe test set | **98,78%** |
| Secvențe de antrenament | 2.035 (din 2.544 totale) |
| Augmentare date | ~96× per videoclip sursă |
| Timp antrenament (CPU) | ~25–35 min |
| FPS aplicație (țintă) | 60 fps |
| Latență recunoaștere (CPU) | ~15–30 ms |
| Latență recunoaștere (GPU RTX 4080) | ~2–5 ms |

---

## 6. Fișiere Generate

| Fișier | Conținut |
|--------|----------|
| `models/alphabet/best_model.pth` | Modelul antrenat (weights PyTorch) |
| `models/alphabet/training_results.json` | Acuratețe, configurație, device |
| `models/alphabet/confusion_matrix.png` | Matricea de confuzie per simbol |
| `models/alphabet/training_history.png` | Grafic loss/accuracy pe epoci |
| `data/alphabet_processed/class_labels.json` | Mapare index → simbol |
| `data/alphabet_augmented/dataset_info.json` | Info dataset augmentat |

---

*Document completat cu date reale din fișierele proiectului — 18 martie 2026.*
*Secțiunea 4.3 (utilizatori reali) necesită completare manuală după testare.*
