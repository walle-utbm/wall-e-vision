# wall-e-vision

Pipeline de vision temps reel pour RubikPi 3. Le modele YOLO detecte les dechets, puis les classe dans une categorie de tri:
- yellow: recyclable
- glass: verre
- other: dechet residuel

Cette branche est simplifiee pour un seul profil RubikPi.

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

Depuis la racine du projet:

```bash
python src/main.py
```

Le profil RubikPi 3 utilise par defaut:
- modele: `model/best.pt`
- resolution inference: `768`
- camera: pipeline GStreamer `qtiqmmfsrc` par defaut, avec `appsink name=sink` pour la capture Python
- sortie: `outputs/`

Si `gst-launch-1.0 qtiqmmfsrc ...` fonctionne sur ta machine, le pipeline Python utilise la meme pile par defaut. Le pipeline attendu termine par `appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true`.

## Configuration

Les reglages se trouvent dans [src/walle_vision/cli.py](src/walle_vision/cli.py). Modifie `RubikPiRuntimeConfig` si tu veux ajuster:
- `imgsz`
- `fps`
- `conf`
- `infer_every`
- `show`
- `camera_test_mode` si tu veux reactiver les frames de verification

## Fichiers utiles

- [src/main.py](src/main.py) - point d entree
- [src/walle_vision/cli.py](src/walle_vision/cli.py) - profil RubikPi unique
- [src/walle_vision/camera.py](src/walle_vision/camera.py) - capture video/camera RubikPi
- [src/walle_vision/detector.py](src/walle_vision/detector.py) - inference YOLO PyTorch
- [src/walle_vision/pipeline.py](src/walle_vision/pipeline.py) - orchestration capture -> inference -> export
- [src/walle_vision/tracking.py](src/walle_vision/tracking.py) - suivi temporel
- [src/walle_vision/visualization.py](src/walle_vision/visualization.py) - annotation des images

## Sorties

- `outputs/detections.jsonl` - detections stables
- `outputs/predict/` - images annotees
- `outputs/camera_test/` - frames de verification si activees

## Notes

Le projet est volontairement centre sur RubikPi 3 avec PyTorch.
