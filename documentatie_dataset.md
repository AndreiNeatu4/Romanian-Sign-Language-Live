# Documentatie: Crearea Dataset-ului pentru Recunoasterea Alfabetului in Limbajul Semnelor

---

## 1. Prezentare Generala

Dataset-ul este construit dintr-o serie de videoclipuri ale gesturilor din alfabetul limbajului semnelor romanesti, procesate prin trei etape succesive: extragerea caracteristicilor, augmentarea datelor si pregatirea setului de antrenament. Rezultatul final este un fisier binar (`dataset.pkl`) ce contine secvente de landmarks normalizate, gata pentru antrenarea unui model de retea neuronala.

---

## 2. Structura Datelor Brute

### 2.1 Organizarea Videoclipurilor

Videoclipurile sursa sunt organizate intr-o structura de directoare ierarhica:

```
alphabet/
├── A/
│   └── a.mp4
├── B/
│   └── b.mp4
├── ...
├── Z/
│   └── z.mp4
├── Â/
│   └── a_circonflexe.mp4
├── Ă/
│   └── a_breve.mp4
├── Ș/
│   └── s_virgula.mp4
└── Ț/
    └── t_virgula.mp4
```

- **30 clase de gesturi:** literele A–Z (26) plus caracterele speciale romanesti Â, Ă, Ș, Ț (4)
- **Format acceptat:** `.mp4`, `.avi`, `.mov`, `.mkv`
- **Continut:** fiecare videoclip contine executia unui singur gest, filmat frontal

---

## 3. Etapa 1 — Extragerea si Augmentarea Datelor

**Script:** `data_preparation/extract_augmented_fast.py`

### 3.1 Preprocesarea Cadrelor

Inainte de procesarea cu MediaPipe, fiecare cadru este **oglindit orizontal** (`cv2.flip`). Aceasta operatiune asigura coerenta dintre perspectiva de extragere (antrenament) si perspectiva camerei in timp real (inferenta), unde imaginea este afisata ca o oglinda.

### 3.2 Extragerea Landmarks-urilor cu MediaPipe

Pentru fiecare cadru se folosesc doua modele MediaPipe:

#### Maini (MediaPipe Hands)
- **Model:** `model_complexity=0` (cel mai rapid)
- **Max maini detectate:** 2
- **Landmarks per mana:** 21 puncte × 3 coordonate (x, y, z) = 63 valori
- **Total maini:** 126 valori (mana stanga + mana dreapta)
- **Identificare:** se foloseste `multi_handedness` pentru a determina care mana este stanga si care este dreapta

#### Structura vectorului de caracteristici per cadru (252 valori)

Vectorul per cadru este compus din doua sectiuni: **pozitii** si **viteze**.

| Index | Continut |
|-------|----------|
| 0 – 62 | Pozitie mana stanga (21 landmarks × 3 coords) |
| 63 – 125 | Pozitie mana dreapta (21 landmarks × 3 coords) |
| 126 – 188 | Viteza mana stanga (diferenta fata de cadrul anterior) |
| 189 – 251 | Viteza mana dreapta (diferenta fata de cadrul anterior) |

Daca o mana nu este detectata, pozitiile si vitezele corespunzatoare raman `0.0`. Pentru primul cadru al unei secvente, viteza este `0.0` (nu exista cadru anterior).

**Viteza (velocity features):** calculata ca `landmarks[t] - landmarks[t-1]` pentru fiecare cadru. Aceasta informatie este esentiala pentru recunoasterea gesturilor dinamice (J, Z) si ajuta la diferentierea semnelor statice similare vizual (ex. A, S, E).

### 3.3 Normalizarea Landmarks-urilor

Fiecare mana este normalizata independent:

1. **Centrare:** toate coordonatele sunt scazute din pozitia incheieturii (landmark 0 — wrist), rezultand coordonate relative
2. **Scalare:** vectorul centrat este impartit la distanta maxima fata de incheietura, obtinand valori in intervalul `[-1, 1]`

Aceasta normalizare face recunoasterea **invarianta la pozitia mainii in cadru si la marimea mainii**.

### 3.4 Augmentarea Spatiala

Se aplica **4 pipeline-uri de augmentare** pe cadrele video (transformari de culoare/luminozitate):

| Pipeline | Transformare |
|----------|-------------|
| 0 | Original — nicio modificare |
| 1 | Variatie de luminozitate/contrast ±7% (simuleaza iluminare diferita) |
| 2 | Variatie de nuanta/saturatie (simuleaza senzori de camera diferiti) |
| 3 | Luminozitate ±5% + saturatie ±8% combinate (cea mai realista variatie) |

**Transformari excluse intentionat:**
- Oglindire orizontala — ar schimba semnificatia semnelor (stanga/dreapta)
- Rotatie — schimba pozitia relativa a mainii fata de fata, informatie critica
- Redare inversa — gesturile inversate au semnificatie diferita

### 3.5 Augmentarea Temporala (Variatii de Viteza)

Fiecare videoclip este redat la **3 viteze diferite:**

| Viteza | Efect |
|--------|-------|
| 0.98x | Usor mai lent |
| 1.0x | Viteza normala |
| 1.02x | Usor mai rapid |

Variatia de ±2% simuleaza diferentele naturale de ritm intre persoane. Variatii mai mari au fost excluse deoarece ar distorsiona semnificativ gestul.

### 3.6 Crearea Secventelor

Din fiecare varianta augmentata se extrag secvente de **30 de cadre consecutive** (non-suprapuse). Lungimea de 30 cadre este aleasa pentru a captura gesturi complete, inclusiv dinamice (J, Z).

### 3.6.1 Augmentarea la Nivel de Landmarks

Dupa extragerea pozitiilor din fiecare varianta spatiala/temporala, se aplica suplimentar **3 tipuri de augmentare directa pe coordonatele landmarks-urilor** (nu pe pixelii video):

| Varianta | Descriere |
|----------|-----------|
| Original | Fara modificare |
| Noise | Zgomot Gaussian (std=0.015) adaugat pe coordonatele mainilor — simuleaza erori de detectie MediaPipe |
| Rot +7° | Rotatie 3D a landmarks-urilor cu +7 grade in jurul axei Z — simuleaza unghi usor diferit de camera |
| Rot -7° | Rotatie 3D a landmarks-urilor cu -7 grade in jurul axei Z |

Aceasta augmentare are avantajul ca actioneaza direct in spatiul caracteristicilor (nu pe imagine), creand variante mai realiste si mai diverse decat augmentarea video.

### 3.7 Multiplicatorul de Augmentare

```
4 augmentari spatiale × 3 viteze × 4 augmentari landmarks = 48 variante per videoclip
```

Fiecare videoclip genereaza aproximativ 48 fisiere `.npy`, salvate in structura:

```
data/alphabet_augmented/
└── A/
    └── a/
        ├── s0_sp0.98_r0.npy          (original)
        ├── s0_sp0.98_r0_noise.npy    (cu zgomot Gaussian)
        ├── s0_sp0.98_r0_rot7.npy     (rotatie +7°)
        ├── s0_sp0.98_r0_rotn7.npy    (rotatie -7°)
        ├── s0_sp1.0_r0.npy
        ├── ...
        └── metadata.json
```

Fiecare fisier `.npy` contine un array de forma `(N, 30, 252)` unde:
- `N` = numarul de secvente extrase din videoclip
- `30` = numarul de cadre per secventa
- `252` = numarul de caracteristici per cadru (126 pozitii + 126 viteze)

### 3.8 Procesare Paralela

Extragerea foloseste **4 procese paralele** (`multiprocessing.Pool`) pentru a accelera procesarea, fiecare proces gestionand un videoclip independent.

---

## 4. Etapa 2 — Pregatirea Dataset-ului

**Script:** `data_preparation/prepare_augmented_dataset.py`

### 4.1 Incarcarea Datelor

Toate fisierele `.npy` sunt incarcate si etichetate automat dupa numele directorului parinte (ex. directorul `A/` → eticheta `0`, `B/` → eticheta `1`, etc.).

### 4.2 Impartirea Datelor

Dataset-ul este impartit in doua seturi folosind `train_test_split` cu stratificare:

| Set | Proportie | Utilizare |
|-----|-----------|-----------|
| Antrenament | 80% | Antrenarea modelului |
| Validare | 20% | Evaluarea performantei in timpul antrenamentului si testarea finala |

Stratificarea asigura ca fiecare clasa este reprezentata proportional in ambele seturi.

### 4.3 Salvarea Dataset-ului

Dataset-ul este salvat ca fisier pickle (`dataset.pkl`) cu urmatoarea structura:

```python
{
    'X_train': np.array,  # forma: (N_train, 30, 252)
    'y_train': np.array,  # etichete intregi
    'X_val':   np.array,  # forma: (N_val, 30, 252)
    'y_val':   np.array,
    'X_test':  np.array,  # identic cu X_val (set mic de date)
    'y_test':  np.array,
    'info': {
        'sequence_length': 30,
        'landmarks_per_frame': 252,
        'num_classes': 30,
        'classes': ['A', 'B', ..., 'Ț'],
        'created_at': '...',
        'source': 'augmented_data'
    }
}
```

De asemenea, se salveaza `class_labels.json` cu maparea dintre indici si numele claselor.

---

## 5. Etapa 3 — Antrenarea Modelului

**Script:** `training/train_model.py`

### 5.1 Arhitectura Modelului (CNN-LSTM)

Modelul combina retele convolutionale (CNN) pentru extragerea caracteristicilor spatiale cu retele LSTM pentru modelarea secventelor temporale:

```
Input: (batch, 30, 252)
    ↓
Conv1D(252→64, kernel=3) + BatchNorm + ReLU + MaxPool
    ↓
Conv1D(64→128, kernel=3) + BatchNorm + ReLU + MaxPool
    ↓
LSTM(128→64, 2 straturi)
    ↓
Dropout(0.2)
    ↓
Linear(64→30)
    ↓
Output: probabilitati pentru 30 clase
```

### 5.2 Parametri de Antrenare

| Parametru | Valoare |
|-----------|---------|
| Epoci | 150 (cu early stopping la 25) |
| Batch size | 32 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Dropout | 0.2 |
| Hardware | NVIDIA RTX 4080 (CUDA) |

### 5.3 Rezultate

| Metric | Valoare |
|--------|---------|
| Acuratete antrenament | ~98.8% |
| Acuratete validare/test | ~98.8% |
| Timp antrenament | ~30 secunde (RTX 4080) |
| Secvente totale dataset | 816 (vs 204 anterior) |
| Secvente per clasa | ~27 (vs 4–8 anterior) |

Acuratetea usor mai scazuta fata de versiunea anterioara (99.86%) reflecta un model mai bine generalizat — dataset-ul este acum de 4x mai mare si mai divers prin augmentarea la nivel de landmarks, reducand suprafitarea.

---

## 6. Fluxul Complet al Pipeline-ului

```
Videoclipuri sursa (alphabet/)
         |
         | cv2.flip() — oglindire pentru consistenta cu inferenta
         |
         v
  MediaPipe Hands
  (detectie landmarks cu identificare mana stanga/dreapta)
         |
         | Normalizare relativa la incheietura
         |
         v
  Vector pozitii per cadru (126 valori)
  [0:63]   = mana stanga
  [63:126] = mana dreapta
         |
         | Augmentare landmarks: noise + rot+7° + rot-7°
         | Calcul viteze: velocity[t] = pos[t] - pos[t-1]
         | Concatenare: [pozitii(126) | viteze(126)] = 252
         |
         | Augmentare totala: 4 spatiale × 3 viteze × 4 landmarks = 48 variante
         |
         v
  Secvente .npy (N, 30, 252)
  salvate in data/alphabet_augmented/
         |
         | Incarcare + stratificare 80/20
         |
         v
  dataset.pkl
  salvat in data/alphabet_processed/
         |
         | Antrenare CNN-LSTM pe GPU
         |
         v
  best_model.pth
  salvat in models/alphabet/
```

---

## 7. Consideratii Tehnice

### 7.1 Consistenta Perspectivei
Videoclipurile de antrenament sunt oglindite inainte de procesare pentru a se alinia cu modul in care camera functioneaza in timp real (imagine in oglinda). Aceasta asigura ca `"Left"` din MediaPipe corespunde intotdeauna mainii stangi a persoanei, atat la antrenament cat si la inferenta.

### 7.2 Limitari ale Dataset-ului Actual
- Un singur videoclip per clasa genereaza ~27 secvente per clasa dupa augmentarea 48x — suficient pentru un prototip functional, insuficient pentru generalizare robusta
- Cu un singur autor al gesturilor, modelul poate fi sensibil la stilul specific de executie al altor persoane
- Se recomanda minimum 5–10 videoclipuri per clasa cu persoane si conditii diferite pentru productie

### 7.3 Extinderea Dataset-ului
Pentru a adauga date noi:
1. Adaugati videoclipuri in directorul corespunzator din `alphabet/{LITERA}/`
2. Rulati `python run_pipeline.py` pentru a re-extrage, re-pregati si re-antrena

---

## 8. Fisiere Generate

| Fisier | Locatie | Descriere |
|--------|---------|-----------|
| `*.npy` | `data/alphabet_augmented/` | Secvente de landmarks augmentate |
| `dataset_info.json` | `data/alphabet_augmented/` | Statistici despre extragere |
| `dataset.pkl` | `data/alphabet_processed/` | Dataset complet pentru antrenament |
| `class_labels.json` | `data/alphabet_processed/` | Maparea clase → indici |
| `best_model.pth` | `models/alphabet/` | Greutati model cu cea mai buna validare |
| `final_model.pth` | `models/alphabet/` | Greutati model dupa toate epocile |
| `training_results.json` | `models/alphabet/` | Metrici si configuratie antrenament |
| `confusion_matrix.png` | `models/alphabet/` | Matricea de confuzie pe setul de test |
| `training_history.png` | `models/alphabet/` | Grafic loss/acuratete per epoca |
