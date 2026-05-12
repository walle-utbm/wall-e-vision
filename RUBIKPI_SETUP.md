# wall-e-vision pour RubikPi 3 (Snapdragon 8 Gen1)

## Profil matériel

**RubikPi 3** est nettement plus puissant que Raspberry Pi 4:

| Aspect | Raspberry Pi 4 | RubikPi 3 |
|--------|---|---|
| **SoC** | Broadcom BCM2711 | Qualcomm Snapdragon 8 Gen1 |
| **CPU** | 4x Cortex-A72 @ 1.5 GHz | 4x A55 @ 2.0 GHz + 4x A78 @ 2.7 GHz |
| **Cores** | 4 | 8 (Big.Little) |
| **RAM** | 4-8 GB | 4-8 GB |
| **TDP** | ~5W | ~8W |
| **Backend** | NCNN recommandé | PyTorch suffisant |

## Installation sur RubikPi 3

### Prérequis

- Ubuntu 24.04 (ou plus récent) sur RubikPi
- Python 3.11+
- picamera2 pour la caméra IMX708

```bash
# Mise à jour système
sudo apt update && sudo apt upgrade -y

# Python + pip
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Dépendances OpenCV
sudo apt install -y libatlas-base-dev libjasper-dev libtiff5 libjasper1 libharfp libwebp6

# libcamera pour IMX708
sudo apt install -y libcamera-tools libcamera-apps

# picamera2 (préinstallé sur Ubuntu 24.04 pour RubikPi, sinon):
pip install picamera2
```

### Cloner le repository

```bash
git clone https://github.com/walle-utbm/wall-e-vision.git
cd wall-e-vision
git checkout rubikpi3  # Basculer vers la branche RubikPi3
```

### Configuration de l'environnement Python

```bash
# Environnement virtuel
python3.11 -m venv .venv
source .venv/bin/activate

# Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt
```

### Tester la détection du matériel

Le script détecte automatiquement si tu es sur RPi4 ou RubikPi:

```bash
# Test: infos matériel
python -c "
import os
from pathlib import Path
if Path('/proc/device-tree/model').exists():
    print('🔧 Plateforme:', Path('/proc/device-tree/model').read_text())
import platform
print('🔧 Architecture:', platform.machine())
import multiprocessing
print('🔧 Cores:', multiprocessing.cpu_count())
"
```

Sortie attendue pour RubikPi 3:
```
🔧 Plateforme: Thundercomm, Inc. RUBIK Pi 3
🔧 Architecture: aarch64
🔧 Cores: 8
```

## Différences de profils

### Raspberry Pi 4 (branche principale `main`)
```python
# Réglages conservateurs pour CPU limité
PiRuntimeConfig:
  - imgsz: 640           # Résolution training
  - fps: 30              # 30 fps smooth
  - infer_every: 1       # Inférence chaque frame
  - confirm_frames: 3    # Tracking stable
  - Backend: NCNN (fallback PyTorch)
```

### RubikPi 3 (branche `rubikpi3`)
```python
# Réglages agressifs pour 8 cores puissants
RubikPiRuntimeConfig:
  - imgsz: 768           # +20% résolution (meilleure précision)
  - fps: 60              # Double fréquence
  - infer_every: 1       # Inférence chaque frame (CPU tient)
  - confirm_frames: 2    # Confirmation plus rapide
  - Backend: PyTorch direct
```

## Lancement

```bash
python src/main.py
```

Le script détecte RubikPi3 automatiquement et charge `RubikPiRuntimeConfig`.

### Sortie attendue:
```
✅ Detected: RubikPi 3 (Snapdragon)
Loading PyTorch model: model/best.pt
📷 Warming up camera sensor (stabilizing white balance)...
[Inférence en cours...]
```

## Tuning selon ton cas d'usage

### Plus haute précision (petits objets)
```python
RubikPiRuntimeConfig:
  conf: float = 0.05    # Seuil plus permissif
  imgsz: int = 960      # Résolution plus grande
  max_det: int = 20     # Plus de détections
```

### Plus haute vitesse
```python
RubikPiRuntimeConfig:
  conf: float = 0.20    # Filtre plus strict
  imgsz: int = 640      # Résolution training
  fps: int = 30         # Moins de pression caméra
  infer_every: int = 2  # Analyser 1 frame sur 2
```