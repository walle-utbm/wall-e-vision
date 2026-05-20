# wall-e-vision

Branche PC du projet wall-e-vision.
Le Raspberry Pi envoie les frames camera en JPEG sur TCP, le PC execute le modele YOLO, puis renvoie les resultats structues au client.

## Structure du projet

```text
wall-e-vision/
	model/
		best.pt
	outputs/
		predict/
		detections.jsonl
	src/
		pc_main.py
		walle_vision/
			__init__.py
			pc/
				__init__.py
				detector.py
				labels.py
				sorting.py
				tracking.py
				types.py
				transport.py
				visualization.py
				server.py
	requirements-pc.txt
	README.md
```

## Ce que fait cette branche

Pour chaque frame recu:
- decodage JPEG
- inference YOLO
- stabilisation temporelle
- rendu optionnel des detections
- ecriture du journal local `outputs/detections.jsonl`
- renvoi d'un JSON de resultat sur la connexion TCP

## Lancement

Installe les dependances PC:

```bash
pip install -r requirements-pc.txt
```

Puis lance le serveur:

```bash
python src/pc_main.py
```

Par defaut, le serveur ecoute sur `0.0.0.0:5000`.

## Configuration rapide

Les principaux reglages sont dans [src/walle_vision/pc/server.py](src/walle_vision/pc/server.py) et [src/walle_vision/pc/detector.py](src/walle_vision/pc/detector.py).

```python
@dataclass(slots=True)
class PCRuntimeConfig:
    host: str = "0.0.0.0"
    port: int = 5000
    model: str = "model/best.pt"
    output_dir: str = "outputs"
    conf: float = 0.10
    iou: float = 0.45
	imgsz: tuple[int, int] = (1280, 720)
    max_det: int = 8
```

## Sorties

Apres chaque run, tu obtiens:
- **outputs/predict/** - images annotees avec boxes detectees
- **outputs/detections.jsonl** - journal structure avec les detections stables

## Dependances

- `ultralytics>=8.3.0` - YOLO inference
- `opencv-python>=4.10` - lecture/dessin d'images
- `numpy>=2.0` - data structures


## Branche PC

La branche PC utilise une entree dediee et ses propres dependances:

```bash
pip install -r requirements-pc.txt
python src/pc_main.py
```

Par defaut, le serveur ecoute sur `0.0.0.0:5000` et renvoie des resultats JSONL dans `outputs/detections.jsonl`.

Optional:
- `torch` - CPU pre-built pour RPi si besoin acceleration

