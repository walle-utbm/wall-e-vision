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

- Ubuntu 24.04 ou plus recent sur RubikPi 3
- Python 3.11+
- camera IMX708 accessible via `picamera2` et `python3-libcamera`

## Installation

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

# picamera2 pour la camera IMX708
pip install picamera2
```

### Cloner le repository

```bash
git clone https://github.com/walle-utbm/wall-e-vision.git
cd wall-e-vision
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

## Lancement

```bash
python src/main.py
```

## Reglages

Les reglages du profil se trouvent dans [src/walle_vision/cli.py](src/walle_vision/cli.py) via `RubikPiRuntimeConfig`.

Tu peux ajuster notamment:
- `imgsz`
- `conf`
- `fps`
- `infer_every`
- `camera_test_mode` si tu veux generer des frames de verification

## Remarques

- Le modele par defaut est `model/best.pt`
- Les sorties sont ecrites dans `outputs/`
- Cette branche utilise uniquement le modele PyTorch `model/best.pt`
- Le profil par defaut utilise un pipeline GStreamer `qtiqmmfsrc` avec `appsink name=sink` pour la camera

## Depannage camera

Si `libcamera-hello` est introuvable, installe les outils systeme de diagnostic:

```bash
sudo apt install -y libcamera-tools
```

Si le script affiche `No camera was detected by libcamera`, alors la pile Python fonctionne mais aucun capteur n'est visible. Verifie alors:

- le branchement du cable CSI
- l'activation de la camera dans le firmware / l'OS
- la presence du paquet systeme `python3-libcamera`

Tu peux aussi tester la detection avec:

```bash
python - <<'PY'
from picamera2 import Picamera2
print(Picamera2.global_camera_info())
PY
```
