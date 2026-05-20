wall-e-vision

Mettre en place une pipeline de vision temps reel pour RubikPi 3. Detecter les dechets avec YOLO, puis les classer dans une categorie de tri:
- yellow: recyclable
- glass: verre
- other: dechet residuel

Conserver une seule branche RubikPi.

## Installation
Branche client Raspberry Pi du projet wall-e-vision.
Le Raspberry Pi capture le flux camera, compresse les frames et les envoie au PC.
Le PC renverra ensuite les resultats d'inference sur la meme connexion TCP.

Cette branche ne fait plus l'inference localement sur le Raspberry Pi.

## Structure du projet

```text
wall-e-vision/
	model/
		best.pt
			remote_results.jsonl
		main.py
		walle_vision/
			__init__.py
			camera.py
			cli.py
			detector.py
				transport.py
				raspberry_pipeline.py
```


Pour chaque frame camera:
- capture sur le Raspberry Pi
- compression JPEG
- envoi au PC via TCP
- reception du resultat PC sous forme JSON
- journal local des resultats recus dans `outputs/remote_results.jsonl`

Le Raspberry Pi ne fait plus l'inference locale. Il reste uniquement le noeud d'acquisition et de transport.

## Installation minimale Raspberry Pi

```bash
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

git clone https://github.com/walle-utbm/wall-e-vision.git
cd wall-e-vision

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Si tu utilises la camera CSI IMX708, installer aussi picamera2 si besoin
pip install picamera2
# Sur Raspberry Pi OS, installer aussi les bindings systeme libcamera si necessaire
sudo apt install python3-libcamera python3-picamera2 -y
```

Test caméra:
```bash
sudo apt install libcamera-apps -y
# Verifier que libcamera voit la camera
libcamera-hello --list-cameras
```

## Lancement

Depuis la racine du projet, lancer:

```bash
python src/main.py
```

## Comment ca marche
Cette commande lance le profil Raspberry de streaming vers le PC.
Le PC doit exposer le serveur TCP d'inference sur l'adresse configuree dans `PiRuntimeConfig`.

Chainer le traitement ainsi:
1. Envoyer les images de la camera RubikPi via un pipeline GStreamer `qtiqmmfsrc` + `appsink`.
2. Lire les frames dans `src/walle_vision/camera.py` et les transmettre au pipeline.
3. Lancer l'inference YOLO dans `src/walle_vision/detector.py`.
4. Stabiliser les detections dans le temps avec `src/walle_vision/tracking.py`.
5. Ecrire les resultats dans `outputs/` et sauvegarder des images annotees depuis `src/walle_vision/pipeline.py`.

Utiliser par defaut:
- modele: `model/best.onnx` genere depuis `model/best.pt`
- resolution inference: `512`
- camera: pipeline GStreamer `qtiqmmfsrc` + `appsink name=sink` pour la capture Python
- sortie: `outputs/`

Terminer le pipeline par `appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true`.
Pour modifier les reglages reseau et camera, edite `PiRuntimeConfig` dans [src/walle_vision/cli.py](src/walle_vision/cli.py):

```python
@dataclass(slots=True)
class PiRuntimeConfig:
	source: str = "0"                      # 0 pour camera IMX708, path pour video
	output_dir: str = "outputs"            # logs reception locales
	width: int = 640                       # resolution camera
	height: int = 640
	fps: int = 30                          # frames par seconde
	show: bool = False                     # afficher la capture en local
	pc_host: str = "192.168.137.1"          # IP du PC
	pc_port: int = 5000                    # port TCP d'entree PC
	jpeg_quality: int = 80                 # compression JPEG avant envoi
```

## Configuration

Ouvrir [src/walle_vision/cli.py](src/walle_vision/cli.py) et modifier `RubikPiRuntimeConfig` pour ajuster:
- `imgsz`
- `fps`
- `conf`
- `infer_every`
- `show`
- `camera_test_mode` pour generer des frames de verification

Limiter le backend camera a `qtiqmmfsrc` + `appsink`.
Apres chaque run, tu obtiens:
- **outputs/remote_results.jsonl** - journal local des resultats recus du PC
- **outputs/camera_test/** - captures brutes si le mode test est active

Format JSONL exemple:
```json
{"frame_index": 0, "timestamp": 123.456, "detections": [{"class_id": 0, "class_name": "Plastic bottle", "recycle_bin": "yellow", "confidence": 0.95, "bbox_xyxy": [100, 200, 150, 250], "pickup_xy": [125, 225], "track_id": 1}]}
```

## Pourquoi `gi`

Utiliser `gi` comme pont Python vers GStreamer. Charger `Gst` directement depuis Python, demarrer le pipeline `qtiqmmfsrc`, et lire les images dans `appsink` sans passer par un autre moteur de capture.
- [src/main.py](src/main.py) - point d'entree executable
- [src/walle_vision/cli.py](src/walle_vision/cli.py) - profil Raspberry Pi de streaming
- [src/walle_vision/camera.py](src/walle_vision/camera.py) - lecture flux camera (OpenCV + Picamera2)
- [src/walle_vision/transport.py](src/walle_vision/transport.py) - protocole TCP frame/result
- [src/walle_vision/raspberry_pipeline.py](src/walle_vision/raspberry_pipeline.py) - capture + envoi + reception

Faire communiquer le code Python avec la pile GStreamer systeme de RubikPi via `gi`.

## Fichiers utiles
### Profil par defaut:
- Resolution: **640×640**
- FPS camera: **30**
- Compression JPEG: **80**
- Transport: **TCP full-duplex** vers le PC
- Resultats: **journalises localement** sur le Raspberry Pi

- [src/main.py](src/main.py) - point d entree
- [src/walle_vision/cli.py](src/walle_vision/cli.py) - profil RubikPi unique
- [src/walle_vision/camera.py](src/walle_vision/camera.py) - capture camera RubikPi via GStreamer
- [src/walle_vision/detector.py](src/walle_vision/detector.py) - inference YOLO PyTorch
- [src/walle_vision/pipeline.py](src/walle_vision/pipeline.py) - orchestration capture -> inference -> export
- [src/walle_vision/tracking.py](src/walle_vision/tracking.py) - suivi temporel
- [src/walle_vision/visualization.py](src/walle_vision/visualization.py) - annotation des images

## Sorties

- `outputs/detections.jsonl` - detections stables
- `outputs/predict/` - images annotees
- `outputs/camera_test/` - frames de verification si activees

## Notes
### Prochaine etape

La prochaine branche devra exposer un serveur TCP sur le PC pour recevoir les frames JPEG, executer le modele, puis renvoyer un JSON de resultats par frame.

## Dependances

- `opencv-python-headless` - camera + compression JPEG
- `numpy>=2.0` - data structures

Optional:
- `picamera2` - camera CSI Raspberry Pi

Centraliser le projet sur RubikPi 3 avec ONNX Runtime pour l'inference.
