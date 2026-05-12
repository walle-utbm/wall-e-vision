# wall-e-vision pour RubikPi 3 (Snapdragon 8 Gen1)

## Profil matériel

**RubikPi 3** est nettement plus puissant que Raspberry Pi 4:

| Aspect | Raspberry Pi 4 | RubikPi 3 |
|--------|---|---|
| **SoC** | Broadcom BCM2711 | Qualcomm Snapdragon 8 Gen1 |
| **CPU** | 4x Cortex-A72 @ 1.5 GHz | 4x A55 @ 2.0 GHz + 4x A78 @ 2.7 GHz |
| **Cores** | 4 | 8 (Big.Little) |
| **RAM** | 8 GB | 8 GB |
| **TDP** | ~5W | ~8W |

## Installation sur RubikPi 3

### Prérequis

- Ubuntu 24.04 (ou plus récent) sur RubikPi
- Python 3.11+
- picamera2 pour la caméra IMX708

```bash
# Mise à jour système
sudo apt update && sudo apt upgrade -y

# Dépôts Ubuntu ARM complets (nécessaires si python3-venv/pip sont absents)
sudo tee /etc/apt/sources.list.d/ubuntu-ports.list >/dev/null <<'EOF'
deb http://ports.ubuntu.com/ubuntu-ports noble main restricted universe multiverse
deb http://ports.ubuntu.com/ubuntu-ports noble-updates main restricted universe multiverse
deb http://ports.ubuntu.com/ubuntu-ports noble-security main restricted universe multiverse
deb http://ports.ubuntu.com/ubuntu-ports noble-backports main restricted universe multiverse
EOF
sudo apt update

# Python + pip
sudo apt install -y python3 python3-venv python3-dev python3-pip

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
python -m venv .venv
source .venv/bin/activate

# Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt
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


## Tuning selon le cas d'usage

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
