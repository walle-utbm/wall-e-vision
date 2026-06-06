# wall-e-vision

Wall-e-vision est le dépôt de vision du robot Wall-E pour le **Rubik Pi 3**. Le projet expose un point d'entrée unique, [main.py](main.py), qui charge [config.yaml](config.yaml), construit le pipeline d'inférence locale et lance la boucle principale.

## Objectif du projet

Le dépôt regroupe la capture caméra, l'inférence d'objets et l'export des résultats dans une architecture unique, exécutée localement sur la carte embarquée Rubik Pi 3. Trois backends d'inférence sont disponibles selon le modèle utilisé :

- accélération NPU (HTP) via des modèles `.dlc` (Qualcomm SNPE / QAIRT) ;
- exécution CPU via des modèles `.onnx` (ONNX Runtime) ;
- exécution via le runner Edge Impulse HTTP local (`.eim`).

## Architecture

Le code est organisé en trois blocs sous [src/walle_vision/](src/walle_vision/).

- `core` gère l'acquisition vidéo.
- `ai` gère le chargement du modèle et l'inférence.
- `pipelines` assemble les briques pour le mode `edge_standalone`.

Le reste du dépôt contient les utilitaires d'affichage, le mapping des labels et les fichiers de configuration.

## Mode d'exécution

### `edge_standalone`

Le seul mode supporté : la capture et l'inférence tournent sur le Rubik Pi 3.

Comportement principal :

- la caméra est ouverte dans un backend dédié ;
- les frames sont récupérées dans un thread séparé pour ne pas bloquer la capture ;
- le détecteur est exécuté localement ;
- les détections sont écrites dans `outputs/detections.jsonl` seulement lorsqu'au moins un objet est détecté ;
- les images annotées peuvent être enregistrées dans `outputs/predict/` ;
- l'affichage OpenCV reste optionnel via `runtime.show`.

## Edge Impulse

Le dépôt prend en charge un backend `edge_impulse_http`.

Dans ce mode :

- [main.py](main.py) démarre automatiquement le runner Edge Impulse en HTTP local si l'URL configurée pointe vers `127.0.0.1`, `localhost` ou `0.0.0.0` ;
- le runner expose l'API sur `/api/info` et `/api/image` ;
- le code Python envoie les images au runner et récupère les prédictions ;
- la sortie du runner est volontairement silencieuse pour éviter les logs de debug dans la console.

Point important : si le fichier `.eim` correspond à un modèle de classification, la réponse Edge Impulse ne contient pas forcément de bounding boxes. Dans ce cas, le code conserve la meilleure classe et la transforme en détection pleine image. Pour obtenir de vraies bounding boxes, le modèle Edge Impulse doit être exporté en détection d'objets.

### Téléchargement du modèle
```
edge-impulse-linux-runner --clean --download /home/ubuntu/walle/wall-e-vision/model/model.eim
```

## Caméras

La création de caméra est centralisée dans [src/walle_vision/core/camera.py](src/walle_vision/core/camera.py) et dépend de `camera_type`.

### `picamera`

Backend destiné aux caméras CSI du Rubik Pi 3, via le pipeline GStreamer `qtiqmmfsrc`. `camera.warmup_frames` permet de stabiliser l'exposition avant la boucle principale.

### `usb`

Backend OpenCV classique.

- ouvre la source avec `cv2.VideoCapture` (V4L2 d'abord) ;
- applique largeur, hauteur, FPS et buffer minimal quand c'est possible ;
- convient aux webcams UVC, aux indices de périphériques et aux sources URL.

### `none`

Backend sans caméra réelle ; aucune frame n'est produite. Utile pour les tests.

## Détection

Le moteur d'inférence est défini dans [src/walle_vision/ai/detector.py](src/walle_vision/ai/detector.py).

### Modèles supportés

- `.dlc` via l'API SNPE (QAIRT) de Qualcomm pour exécution accélérée sur le NPU (HTP) ;
- `.onnx` via ONNX Runtime pour l'exécution CPU ;
- `.eim` via le backend Edge Impulse HTTP.

### Paramètres importants

- `detector.conf_threshold` contrôle le seuil de confiance ;
- `detector.iou_threshold` contrôle le filtrage des boîtes ;
- `detector.image_size` fixe la taille d'entrée du modèle ;
- `detector.max_detections` limite le nombre de résultats conservés ;
- `detector.backend` choisit entre `pysnpe`, `onnx` et `edge_impulse_http`.

### Logique d'exécution

- les modèles `.dlc` utilisent l'API Qualcomm SNPE. L'entrée subit un redimensionnement (letterbox) et une normalisation Float32 (0-1). La sortie est décodée directement : pour un export YOLO26 end-to-end (NMS-free), chaque ligne est déjà `[x1, y1, x2, y2, score, class]` ;
- les modèles `.onnx` passent par un décodage direct de la sortie (softmax sur les classes puis NMS) pour éviter les mauvaises interprétations d'un wrapper ;
- les réponses Edge Impulse sont décodées depuis JSON ; si le backend renvoie une classification pure, le résultat est converti en détection pleine image.

## Pipeline

### [pipeline_edge.py](src/walle_vision/pipelines/pipeline_edge.py)

Pipeline d'exécution locale.

Responsabilités :

- créer la caméra ;
- créer le détecteur ;
- récupérer les frames en continu ;
- exécuter l'inférence au rythme défini par la config ;
- écrire les résultats utiles dans `outputs/`.

Points pratiques :

- `runtime.infer_every_n_frames` réduit la charge CPU ;
- `runtime.save_every_n_frames` contrôle la fréquence des exports d'images annotées ;
- `runtime.camera_test_mode` permet de sauvegarder périodiquement des frames brutes pour diagnostic.

## Sorties générées

- `outputs/detections.jsonl` pour les détections.
- `outputs/predict/` pour les images annotées.
- `outputs/camera_test/` pour les frames brutes de diagnostic quand `runtime.camera_test_mode` est activé.

## Configuration

Le fichier [config.yaml](config.yaml) est la source de vérité.

### Clés principales

- `hardware`: `rubikpi3`.
- `camera_type`: `picamera`, `usb` ou `none`.
- `mode`: `edge_standalone`.

### Sections secondaires

- `paths`: emplacements des modèles et des sorties.
- `models`: nom du modèle à utiliser pour le Rubik Pi 3.
- `camera`: profil caméra du Rubik Pi 3.
- `detector`: backend, seuils et paramètres d'inférence.
- `runtime`: affichage, fréquence d'inférence, mode test caméra.

### Exemple

```yaml
hardware: rubikpi3
camera_type: picamera
mode: edge_standalone

models:
  rubikpi3: best_final_npu.dlc # best.onnx, best_final_npu.dlc or model.eim

detector:
  rubikpi3:
    backend: pysnpe # pysnpe, onnx or edge_impulse_http
```

## Installation

```bash
cd /home/ubuntu/walle

git clone https://github.com/walle-utbm/wall-e-vision.git
cd wall-e-vision

python -m venv .venv
source .venv/bin/activate
pip install -r requirements/edge.txt
export SNPE_ROOT=/home/ubuntu/qairt/qairt/2.46.0.260424
export SNPE_TARGET_ARCH=aarch64-oe-linux-gcc11.2
export PYTHONPATH=/usr/local/lib/python3/dist-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$SNPE_ROOT/lib/aarch64-oe-linux-gcc11.2:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH="$SNPE_ROOT/lib/hexagon-v68/unsigned;$SNPE_ROOT/lib/hexagon-v73/unsigned;/usr/lib/rfsa/adsp;/dsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp"

cd /home/ubuntu/walle/wall-e-vision/src/snpe_native
pip install -e .

cd /home/ubuntu/walle/wall-e-vision
python main.py
```

### Accélération matérielle (SNPE / QAIRT)

Pour utiliser le backend `pysnpe` et exploiter le NPU (HTP) avec des modèles `.dlc`, il faut installer le SDK Qualcomm AI Runtime (QAIRT) ainsi que le module natif `snpe_native` fourni dans [src/snpe_native/](src/snpe_native/).

**1. Bibliothèques système Qualcomm**
```bash
wget -qO- https://cdn.edgeimpulse.com/qc-ai-docs/device-setup/install_ai_runtime_sdk.sh | bash
source ~/.bash_profile
```

**2. Module natif `snpe_native`**
```bash
# Environnement virtuel actif
cd /home/ubuntu/walle/wall-e-vision/src/snpe_native
pip install -e .
```
*Note : le backend exige que la variable d'environnement `SNPE_ROOT` (ou `QAIRT_ROOT`) soit définie et que les dépendances système soient satisfaites pour accéder au DSP/HTP.*

## Lancement

Le lancement se fait depuis la racine du dépôt.

```bash
source .venv/bin/activate
python main.py
```

Le point d'entrée charge la configuration, démarre automatiquement le runner Edge Impulse si le backend configuré le demande, puis lance le pipeline.

## Dépendances

- [requirements/base.txt](requirements/base.txt) pour le socle commun (`numpy`, `opencv`, `PyYAML`).
- [requirements/edge.txt](requirements/edge.txt) pour l'embarqué (`onnx`, `onnxruntime`, `pybind11`).

Le backend NPU s'appuie en plus sur le module natif `snpe_native` et le SDK QAIRT.

## Points de débogage utiles

- Si la caméra ne démarre pas, vérifier `camera_type`, `source` et l'exposition du périphérique caméra.
- Si Edge Impulse ne répond pas, vérifier que le runner HTTP local est bien lancé sur l'URL configurée.
- Si aucune détection n'apparaît, vérifier le modèle chargé, le backend choisi et `detector.conf_threshold`.
- Si le modèle Edge Impulse est un classifieur, l'absence de bounding boxes est normale.

## Fichiers utiles

- [src/walle_vision/core/camera.py](src/walle_vision/core/camera.py)
- [src/walle_vision/ai/detector.py](src/walle_vision/ai/detector.py)
- [src/walle_vision/pipelines/pipeline_edge.py](src/walle_vision/pipelines/pipeline_edge.py)
- [src/walle_vision/utils/visualization.py](src/walle_vision/utils/visualization.py)
- [src/walle_vision/utils/labels.py](src/walle_vision/utils/labels.py)

## Documentation matérielle

https://www.thundercomm.com/rubik-pi-3/en/docs/about-rubikpi/
