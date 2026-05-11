# wall-e-vision

Pipeline de vision temps reel pour detecter les dechets avec un modele YOLO fine-tune, puis les classer dans une categorie de tri:
- **yellow**: recyclable (bac jaune)
- **glass**: verre uniquement
- **other**: autre dechet / poubelle residuelle

Le code est volontairement decoupe en petits modules pour rester lisible et maintenable sur un systeme embarque (Raspberry Pi 4/5, 8 Go RAM + IMX708).

## Structure du projet

```text
wall-e-vision/
	model/
		best.pt
	outputs/
		frames/
		detections.jsonl
	src/
		main.py
		walle_vision/
			__init__.py
			camera.py
			cli.py
			detector.py
			labels.py
			pipeline.py
			sorting.py
			types.py
			visualization.py
	requirements.txt
	README.md
```

## Ce que le script produit

Pour chaque frame inferee:
- objet detecte (`class_name`)
- categorie de tri (`recycle_bin`: `yellow`, `glass`, `other`)
- boite englobante (`bbox_xyxy`: `x1,y1,x2,y2`)
- centre de la boite (`center_xy`)
- point de prise pour le robot (`pickup_xy`)
- score de confiance (`confidence`)
- ratio de surface de boite (`area_ratio`)
- ratio de surface masque (`mask_area_ratio`)
- indicateur de segmentation active (`segmentation_available`)
- indicateur de box coupee sur le bord image (`bbox_clipped`)
- identifiant de suivi si l'objet est confirme (`track_id`)
- nombre de frames confirmant le suivi (`track_hits`)
- nombre de frames ignorees avant suppression (`track_missed_frames`)

Le pipeline ne publie que les detections stables: un objet doit etre vu plusieurs fois de suite avant d'etre considere comme confirme. La confiance est aussi lissee sur plusieurs frames.

Si le modele est un modele de segmentation YOLO, `pickup_xy` est calcule sur le centroide du masque. Sinon, le code bascule automatiquement sur le centre de la bounding box.

Et en sortie:
- images annotees avec bounding boxes dans `outputs/predict/`
- journal structure JSONL dans `outputs/detections.jsonl`

Seules les frames contenant au moins une detection stable sont enregistrees sur disque (images et JSONL).
Le JSONL est volontairement compact (uniquement les champs utiles au robot).

## Mapping des classes vers tri

Le mapping des 14 classes est dans `src/walle_vision/labels.py`.

### Classes support ees:

| Classe | Tri |
|--------|-----|
| Plastic bottle | yellow |
| Glass bottle | glass |
| Cardboard | yellow |
| Cup | other |
| Paper bag | yellow |
| Soft plastic | yellow |
| Food Packet | other |
| Paper | yellow |
| Organic | other |
| Metal | yellow |
| Ramen Cup | other |
| Printing industry | yellow |
| Plastic bottle cap | yellow |
| Straw | other |

Tu peux facilement modifier les regles dans `src/walle_vision/labels.py` si tes donnees ou priorites changent.

## Installation (Windows, webcam PC)

```bash
# Option A: conda (recommande)
conda create -n walle-vision python=3.11 -y
conda activate walle-vision
pip install -r requirements.txt

# Option B: venv
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Raspberry Pi 4/5 (8GB RAM + IMX708)

**Prerequis: Camera IMX708 connectée et libcamera configurée**

#### Configuration IMX708 (ajouter dans /boot/config.txt)

```ini
# Décommenter/ajouter ces lignes:
camera_auto_detect=0
dtoverlay=imx708
```

Puis redémarrer:
```bash
sudo reboot
```

#### Installation

```bash
cd /home/walle
# Cloner et installer
git clone https://github.com/walle-utbm/wall-e-vision.git
cd wall-e-vision
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Si tu utilises la camera CSI IMX708, installer aussi picamera2 si besoin
pip install picamera2
# Sur Raspberry Pi OS, installer aussi les bindings systeme libcamera
sudo apt install python3-libcamera python3-picamera2 -y
```

Test caméra:
```bash
sudo apt install libcamera-apps -y
# Verifier que libcamera voit la camera
libcamera-hello --list-cameras
```

## Optimisation NCNN pour Raspberry Pi (Recommandé ⚡)

Pour **4x plus rapide** sur ARM, convertissez votre modèle en NCNN:

```bash
# Etape 1: Installer OpenVINO (requis une seule fois)
pip install openvino-dev

# Etape 2: Convertir .pt -> NCNN (.param + .bin)
python convert_to_ncnn.py --model model/best.pt --output model/

# Résultat:
# model/best.ncnn.param  (~100KB)
# model/best.ncnn.bin    (~30MB)
```

Le code détecte automatiquement et utilise NCNN s'il est disponible. Sur Raspberry Pi ARM64, l'inférence passe désormais par le runtime NCNN natif du fichier exporté `best_ncnn_model/`.

**Voir [NCNN_CONVERSION.md](NCNN_CONVERSION.md) pour détails complets**

## Lancement

Depuis la racine du projet:

```bash
python src/main.py
```

Cette commande lance un profil unique **optimisé pour Raspberry Pi 8GB + IMX708**.
Il n'y a plus de mode arguments pour simplifier l'utilisation.

### Configuration rapide

Pour modifier les reglages (fps, resolution, confiance, etc.), edite `PiRuntimeConfig` dans [src/walle_vision/cli.py](src/walle_vision/cli.py):

```python
@dataclass(slots=True)
class PiRuntimeConfig:
    model: str = "model/best.pt"          # chemin du modele
    source: str = "0"                      # 0 pour camera IMX708, path pour video
    conf: float = 0.30                     # seuil confiance (0.30 = equilibre bon)
    imgsz: int = 640                       # taille inference (640 = training resolution)
    infer_every: int = 1                   # inference a chaque frame (1 pour 8GB)
    width: int = 640                       # resolution camera (640 = training)
    height: int = 640
    fps: int = 30                          # frames par seconde (30 smooth real-time)
    show: bool = False                     # afficher debug (False on RPi headless)
    half: bool = True                      # FP16 pour GPU acceleration
```

### Sorties

Apres chaque run, tu obtiens:
- **outputs/predict/** - images annotees avec boxes detectees
- **outputs/detections.jsonl** - journal structure avec tous les detections stables

Format JSONL exemple:
```json
{"frame_index": 0, "timestamp": 123.456, "detections": [
  {"class_id": 0, "class_name": "Plastic bottle", "recycle_bin": "yellow", "confidence": 0.95, 
   "bbox_xyxy": [100, 200, 150, 250], "pickup_xy": [125, 225], "track_id": 1}
]}
```

## Role Des Fichiers

- [src/main.py](src/main.py) - point d'entree executable (lance cli.run())
- [src/walle_vision/cli.py](src/walle_vision/cli.py) - profil unique Raspberry Pi (sans args CLI)
- [src/walle_vision/camera.py](src/walle_vision/camera.py) - lecture flux camera (OpenCV + ArduCAM)
- [src/walle_vision/detector.py](src/walle_vision/detector.py) - inference YOLO + calcul pickup_xy
- [src/walle_vision/tracking.py](src/walle_vision/tracking.py) - stabilisation temporelle (IoU-based)
- [src/walle_vision/pipeline.py](src/walle_vision/pipeline.py) - orchestration capture -> infer -> export
- [src/walle_vision/visualization.py](src/walle_vision/visualization.py) - rendu boxes + point prise
- [src/walle_vision/types.py](src/walle_vision/types.py) - structures donnees shared
- [src/walle_vision/labels.py](src/walle_vision/labels.py) - 14 classes + mapping tri
- [src/walle_vision/sorting.py](src/walle_vision/sorting.py) - fonction map classe->bac

## Notes optimisation Raspberry Pi 4/5 (8GB + IMX708)

### Profil par defaut:
- Resolution: **640×640** (= training resolution) → meilleure precision
- FPS camera: **30** (smooth real-time) → moins de latence
- Inference: **chaque frame** (infer_every=1) → CPU peut tenir sur 8GB
- Confiance minimale: **0.30** → bon equilibre precision/recall
- Tracking confirmation: **3 frames** → plus stable
- **FP16 active** → 2x plus rapide si GPU disponible
- **workers=0** → evite crash DataLoader sur RPi/Windows

### IMX708 specifics:
- V4L2 backend pour stabilite libcamera
- Buffer limite a 1 frame → low-latency
- Autofocus desactive → detections plus coherentes
- Config /boot/config.txt: `camera_auto_detect=0` + `dtoverlay=imx708`

### Tuning selon tes besoins:

### Tuning selon tes besoins:

**Si tu veux PLUS de precision (meilleur recall):**
```python
conf: float = 0.25                    # detecte plus de petits objets
infer_every: int = 1                  # inference a chaque frame
track_window: int = 7                 # lissage plus long
imgsz: int = 416                      # augmenter taille (vs 640)
```

**Si tu veux PLUS de vitesse (moins de load CPU):**
```python
conf: float = 0.40                    # ignore petits/bruit detections  
infer_every: int = 2                  # saute frames
save_every: int = 10                  # moins d'I/O disque
```

## Dependances

- `ultralytics>=8.3.0` - YOLO inference
- `opencv-python>=4.10` - camera + visualisation
- `numpy>=2.0` - data structures

Optional:
- `torch` - CPU pre-built pour RPi si besoin acceleration

