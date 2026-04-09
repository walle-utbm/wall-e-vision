# wall-e-vision

Pipeline de vision temps reel pour detecter les dechets avec un modele YOLO fine-tune, puis les classer dans une categorie de tri:
- yellow: recyclable (bac jaune)
- glass: verre
- other: autre dechet

Le code est volontairement decoupe en petits modules pour rester lisible et maintenable sur un systeme embarque (Raspberry Pi 4/5, 2 Go RAM).

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
- images annotees avec bounding boxes dans `outputs/frames/`
- journal structure JSONL dans `outputs/detections.jsonl`

Seules les frames contenant au moins une detection stable sont enregistrees sur disque (images et JSONL).
Le JSONL est volontairement compact (uniquement les champs utiles au robot).

## Mapping des classes vers tri

Le mapping des 28 classes est dans `src/walle_vision/labels.py`.
Tu peux le modifier facilement si tes regles de tri evoluent.

## Installation (Windows, webcam PC)

### Option A - conda (recommande)

```bash
conda create -n walle-vision python=3.11 -y
conda activate walle-vision
pip install -r requirements.txt
```

### Option B - venv

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Lancement

Depuis la racine du projet:

```bash
python src/main.py
```

Cette commande lance un profil unique adapte Raspberry Pi 2 Go.
Il n'y a plus de mode arguments pour simplifier l'utilisation.
Pour modifier les reglages, edite `PiRuntimeConfig` dans `src/walle_vision/cli.py`.

## Role Des Fichiers

- `src/main.py`: point d'entree executable
- `src/walle_vision/cli.py`: profil unique Raspberry Pi (sans args)
- `src/walle_vision/camera.py`: lecture flux camera/video
- `src/walle_vision/detector.py`: inference YOLO + calcul `pickup_xy`
- `src/walle_vision/tracking.py`: stabilisation temporelle des detections
- `src/walle_vision/pipeline.py`: orchestration globale capture -> infer -> export
- `src/walle_vision/visualization.py`: dessin des boxes et points de prise
- `src/walle_vision/types.py`: structures de donnees communes
- `src/walle_vision/labels.py`: classes du modele + mapping de tri
- `src/walle_vision/sorting.py`: fonction de mapping classe -> bac

## Notes optimisation Raspberry Pi 4/5 (2 Go)

- le profil par defaut est regle en `640x640` pour rester aligne avec l'entrainement
- l'inference est decimee (`infer_every=3`) pour contenir la charge CPU
- le JSON de sortie est compact et ecrit uniquement lors de detections stables
- le debug visuel est desactive par defaut (`show=False`)
- la camera applique un verrouillage best-effort de certains automatismes (focus/exposition/WB)

## Docker

Non ajoute volontairement pour garder le projet simple et leger.
Le script est directement utilisable via environnement Python/conda.
