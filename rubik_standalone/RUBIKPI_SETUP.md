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
- camera IMX708 accessible via la pile GStreamer RubikPi (`qtiqmmfsrc` + `appsink`) et les bindings Python `gi`

`gi` est necessaire parce qu'il fournit les bindings Python de GObject/GStreamer. Le projet l'utilise pour ouvrir le pipeline camera directement depuis Python et recuperer les frames dans `appsink`.

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

# Python + pip + bindings GStreamer
sudo apt install -y python3 python3-venv python3-dev python3-pip python3-gi gir1.2-gstreamer-1.0 gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good

# Verifie que la pile camera RubikPi fournit bien qtiqmmfsrc
gst-inspect-1.0 qtiqmmfsrc
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

- Le modele par defaut est `model/best.onnx`, genere depuis `model/best.pt`
- Les sorties sont ecrites dans `outputs/`
- Cette branche utilise le modele PyTorch pour l'export initial, puis ONNX Runtime pour l'execution CPU
- Le profil par defaut utilise un pipeline GStreamer `qtiqmmfsrc` avec `appsink name=sink` pour la camera

## Performance

Le modele ONNX lance a `512` pixels d'entree offre un bien meilleur compromis vitesse / qualite que le chemin PyTorch de reference sur RubikPi 3.

## Fonctionnement

Le projet demarre un pipeline camera, alimente l'inference YOLO sur chaque frame, stabilise les detections sur quelques images, puis ecrit les resultats stables dans `outputs/detections.jsonl` et les images annotees dans `outputs/predict/`.

## Depannage camera

Si le pipeline ne demarre pas, verifie alors:

- le branchement du cable CSI
- l'activation de la camera dans le firmware / l'OS
- la presence de `qtiqmmfsrc` via `gst-inspect-1.0 qtiqmmfsrc`

Tu peux aussi tester la couche GStreamer avec:

```bash
gst-launch-1.0 qtiqmmfsrc camera=0 ! fakesink
```
