# wall-e-vision

Mettre en place une pipeline de vision temps reel pour RubikPi 3. Detecter les dechets avec YOLO, puis les classer dans une categorie de tri:
- yellow: recyclable
- glass: verre
- other: dechet residuel

Conserver une seule branche RubikPi.

## Installation

```bash
cd /home/walle

git clone https://github.com/walle-utbm/wall-e-vision.git
cd wall-e-vision

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancement

Depuis la racine du projet, lancer:

```bash
python src/main.py
```

## Comment ca marche

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

## Configuration

Ouvrir [src/walle_vision/cli.py](src/walle_vision/cli.py) et modifier `RubikPiRuntimeConfig` pour ajuster:
- `imgsz`
- `fps`
- `conf`
- `infer_every`
- `show`
- `camera_test_mode` pour generer des frames de verification

Limiter le backend camera a `qtiqmmfsrc` + `appsink`.

## Pourquoi `gi`

Utiliser `gi` comme pont Python vers GStreamer. Charger `Gst` directement depuis Python, demarrer le pipeline `qtiqmmfsrc`, et lire les images dans `appsink` sans passer par un autre moteur de capture.

Faire communiquer le code Python avec la pile GStreamer systeme de RubikPi via `gi`.

## Fichiers utiles

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

Centraliser le projet sur RubikPi 3 avec ONNX Runtime pour l'inference.
