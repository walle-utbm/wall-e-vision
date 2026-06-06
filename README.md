# wall-e-vision

Wall-e-vision est le dépôt de vision du robot Wall-E. Le projet expose un point d'entrée unique, [main.py](main.py), qui charge [config.yaml](config.yaml), choisit le pipeline adapté au matériel et au mode d'exécution, puis lance la boucle principale.

## Objectif du projet

Le dépôt regroupe la capture caméra, l'inférence d'objets, le transport réseau et l'export des résultats dans une architecture unique. L'objectif est de pouvoir exécuter le même code dans plusieurs contextes:

- inférence locale sur la carte embarquée;
- capture locale et inférence sur un PC distant;
- exécution Edge Impulse via un serveur HTTP local.

## Architecture

Le code est organisé en quatre blocs.

- `core` gère l'acquisition vidéo.
- `ai` gère le chargement du modèle et l'inférence.
- `network` gère le protocole TCP et le transport des frames JPEG.
- `pipelines` assemble les briques selon le mode choisi.

Le reste du dépôt contient les utilitaires d'affichage, le mapping des labels et les fichiers de configuration.

## Modes d'exécution

### `edge_standalone`

Ce mode fait tourner la capture et l'inférence sur la même machine.

Comportement principal:

- la caméra est ouverte dans un backend dédié;
- les frames sont récupérées dans un thread séparé pour ne pas bloquer la capture;
- le détecteur est exécuté localement;
- les détections sont écrites dans `outputs/detections.jsonl` seulement lorsqu'au moins un objet est détecté;
- les images annotées peuvent être enregistrées dans `outputs/predict/`;
- l'affichage OpenCV reste optionnel via `runtime.show`.

Ce mode est utilisé pour:

- Raspberry Pi 4 avec modèle local `.pt`;
- Rubik Pi 3 avec modèle local `.onnx` ou avec un backend Edge Impulse HTTP.

### `stream_client`

Ce mode capture localement et envoie les frames à un PC via TCP.

Comportement principal:

- la caméra est ouverte sur la machine cliente;
- chaque frame est compressée en JPEG;
- les frames sont envoyées au serveur PC;
- les réponses du serveur sont stockées dans `outputs/remote_results.jsonl`;
- le client limite le nombre de frames en vol avec `network.max_inflight_frames`.

### `stream_server`

Ce mode reçoit les frames sur un PC et exécute l'inférence côté serveur.

Comportement principal:

- le serveur écoute sur `network.host:network.port`;
- il reçoit des frames JPEG via un protocole TCP simple;
- il les décode en OpenCV;
- il exécute le détecteur;
- il renvoie les résultats au client;
- il archive les résultats dans `outputs/detections.jsonl` et les images annotées dans `outputs/predict/`.

## Edge Impulse

Le dépôt prend aussi en charge un backend `edge_impulse_http`.

Dans ce mode:

- [main.py](main.py) démarre automatiquement le runner Edge Impulse en HTTP local si l'URL configurée pointe vers `127.0.0.1`, `localhost` ou `0.0.0.0`;
- le runner expose l'API sur `/api/info` et `/api/image`;
- le code Python envoie les images au runner et récupère les prédictions;
- le seuil du runner est fixé à `0.1` au démarrage automatique actuel;
- la sortie du runner est volontairement silencieuse pour éviter les logs de debug dans la console.

Point important: si le fichier `.eim` correspond à un modèle de classification, la réponse Edge Impulse ne contient pas forcément de bounding boxes. Dans ce cas, le code conserve la meilleure classe et la transforme en détection pleine image. Pour obtenir de vraies bounding boxes, le modèle Edge Impulse doit être exporté en détection d'objets.

### Téléchargement du modèle
```
edge-impulse-linux-runner --clean --download /home/ubuntu/walle/wall-e-vision/model/model.eim
``` 

## Caméras

La création de caméra est centralisée dans [src/walle_vision/core/camera.py](src/walle_vision/core/camera.py).

### `picamera`

Backend destiné aux caméras CSI.

Comportement:

- tente d'abord `Picamera2` quand elle est disponible;
- sur Rubik Pi 3, le backend GStreamer peut être utilisé;
- `camera.warmup_frames` permet de stabiliser l'exposition avant la boucle principale.

### `usb`

Backend OpenCV classique.

Comportement:

- ouvre la source avec `cv2.VideoCapture`;
- applique largeur, hauteur, FPS et buffer minimal quand c'est possible;
- convient aux webcams UVC, aux indices de périphériques et aux sources URL.

### `none`

Backend sans caméra réelle.

Comportement:

- aucune frame n'est produite;
- ce backend sert aux cas où la caméra est gérée ailleurs ou aux tests réseau.

## Détection

Le moteur d'inférence est défini dans [src/walle_vision/ai/detector.py](src/walle_vision/ai/detector.py).

### Modèles supportés

- `.pt` via Ultralytics;
- `.onnx` via ONNX Runtime pour le chemin local;
- `.eim` via le backend Edge Impulse HTTP;
- `.dlc` via l'API PySNPE (QAIRT) de Qualcomm pour exécution accélérée sur le NPU (HTP) du Rubik Pi 3.

### Paramètres importants

- `detector.conf_threshold` contrôle le seuil de confiance;
- `detector.iou_threshold` contrôle le filtrage des boîtes;
- `detector.image_size` fixe la taille d'entrée du modèle;
- `detector.max_detections` limite le nombre de résultats conservés;
- `detector.backend` choisit entre `ultralytics`, `edge_impulse_http` et `pysnpe`.

### Logique d'exécution

- les modèles `.pt` passent par Ultralytics;
- les modèles `.onnx` passent par un décodage direct de la sortie pour éviter les mauvaises interprétations du wrapper;
- les modèles `.dlc` utilisent l'API Qualcomm PySNPE. L'entrée subit un redimensionnement et une normalisation Float32 (0-1). Après l'inférence sur le NPU, les tenseurs de sortie INT8 sont automatiquement déquantifiés (Float = (INT8 - offset) * scale) avant d'être passés au post-traitement agnostique;
- les réponses Edge Impulse sont décodées depuis JSON;
- si le backend Edge Impulse renvoie une classification pure, le résultat est converti en détection pleine image.

## Pipelines

### [pipeline_edge.py](src/walle_vision/pipelines/pipeline_edge.py)

Pipeline principal pour l'exécution locale.

Responsabilités:

- créer la caméra;
- créer le détecteur;
- récupérer les frames en continu;
- exécuter l'inférence au rythme défini par la config;
- écrire les résultats utiles dans `outputs/`.

Points pratiques:

- `runtime.infer_every_n_frames` réduit la charge CPU;
- `runtime.save_every_n_frames` contrôle la fréquence des exports d'images annotées;
- `runtime.camera_test_mode` permet de sauvegarder périodiquement des frames brutes pour diagnostic.

### [pipeline_client.py](src/walle_vision/pipelines/pipeline_client.py)

Pipeline client pour le mode streaming.

Responsabilités:

- capturer les frames;
- les compresser en JPEG;
- les envoyer au serveur PC;
- recevoir les réponses;
- suivre les métriques de latence et de débit.

### [pipeline_server.py](src/walle_vision/pipelines/pipeline_server.py)

Pipeline serveur pour le mode streaming.

Responsabilités:

- écouter les connexions entrantes;
- décoder les frames JPEG;
- exécuter l'inférence;
- renvoyer les résultats au client.

Le serveur traite les clients séquentiellement et s'arrête proprement avec les signaux système standard.

## Sorties générées

Les fichiers et dossiers principaux sont les suivants.

- `outputs/detections.jsonl` pour les détections locales ou serveur.
- `outputs/remote_results.jsonl` pour les résultats reçus par le client streaming.
- `outputs/predict/` pour les images annotées.
- `outputs/camera_test/` pour les frames brutes de diagnostic quand `runtime.camera_test_mode` est activé.

## Configuration

Le fichier [config.yaml](config.yaml) est la source de vérité.

### Clés principales

- `hardware`: `rpi4`, `rubikpi3` ou `pc`.
- `camera_type`: `picamera`, `usb` ou `none`.
- `mode`: `edge_standalone`, `stream_client` ou `stream_server`.

### Sections secondaires

- `paths`: emplacements des modèles et des sorties.
- `models`: nom du modèle à utiliser pour chaque matériel.
- `camera`: profils caméra par matériel.
- `detector`: backend, seuils et paramètres d'inférence.
- `runtime`: affichage, fréquence d'inférence, mode test caméra.
- `network`: hôte, ports, qualité JPEG et limites réseau.

### Exemple de profils

```yaml
# Raspberry Pi 4 autonome
hardware: rpi4
camera_type: picamera
mode: edge_standalone

# Rubik Pi 3 autonome avec Edge Impulse
hardware: rubikpi3
camera_type: picamera
mode: edge_standalone

# Client qui envoie les images au PC
hardware: rpi4
camera_type: picamera
mode: stream_client

# Serveur PC
hardware: pc
camera_type: none
mode: stream_server
```

## Installation

```bash
cd /home/walle

git clone https://github.com/walle-utbm/wall-e-vision.git
cd wall-e-vision

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Accélération matérielle Rubik Pi 3 (QAIRT / PySNPE)

Pour utiliser le backend `pysnpe` pour exploiter le NPU (HTP) du Rubik Pi 3 avec des modèles `.dlc`, il faut installer le SDK Qualcomm AI Runtime (QAIRT) et son wrapper Python, car `pysnpe` n'est pas disponible sur PyPI.

**1. Installation des bibliothèques système sur le Rubik Pi 3**
Exécuter ce script fourni par Qualcomm / Edge Impulse pour installer les bibliothèques C++ d'inférence (`libQnnHtp.so`, etc.) :
```bash
wget -qO- https://cdn.edgeimpulse.com/qc-ai-docs/device-setup/install_ai_runtime_sdk.sh | bash
source ~/.bash_profile
```

**2. Installation du wrapper pysnpe_utils**
Le module Python qui encapsule l'API C++ fait partie du SDK Qualcomm (ou du Qualcomm Innovators Development Kit - QIDK). Vous devez récupérer les sources et l'installer dans votre environnement virtuel :
```bash
# Clonez les outils QIDK (qui contiennent pysnpe_utils)
git clone https://github.com/qualcomm/qidk.git
cd qidk/Tools/pysnpe_utils

# Assurez-vous que l'environnement virtuel wall-e-vision est actif
# source /home/walle/wall-e-vision/.venv/bin/activate

# Installez le package
pip install .
```
*Note: Le backend exige que la variable d'environnement `SNPE_ROOT` (ou `QAIRT_ROOT`) soit définie et que les dépendances système soient satisfaites pour accéder au DSP/HTP.*

## Lancement

Le lancement se fait depuis la racine du dépôt.

```bash
source .venv/bin/activate
python main.py
```

Le point d'entrée charge la configuration, démarre automatiquement le runner Edge Impulse si le backend configuré le demande, puis lance le pipeline adapté.

## Dépendances

Les dépendances sont séparées par cible.

- [requirements/base.txt](requirements/base.txt) pour le socle commun.
- [requirements/edge.txt](requirements/edge.txt) pour les modes embarqués.
- [requirements/pc.txt](requirements/pc.txt) pour le serveur PC.

Le projet s'appuie principalement sur `ultralytics`, `opencv-python`, `numpy`, `PyYAML` et `onnxruntime` selon le chemin d'exécution.

## Points de débogage utiles

- Si la caméra ne démarre pas, vérifier `camera_type`, `source` et l'exposition du périphérique caméra.
- Si Edge Impulse ne répond pas, vérifier que le runner HTTP local est bien lancé sur l'URL configurée.
- Si aucune détection n'apparaît, vérifier le modèle chargé, le backend choisi et `detector.conf_threshold`.
- Si le modèle Edge Impulse est un classifieur, l'absence de bounding boxes est normale.

## Fichiers utiles

- [src/walle_vision/core/camera.py](src/walle_vision/core/camera.py)
- [src/walle_vision/ai/detector.py](src/walle_vision/ai/detector.py)
- [src/walle_vision/network/transport.py](src/walle_vision/network/transport.py)
- [src/walle_vision/pipelines/](src/walle_vision/pipelines/)
- [src/walle_vision/utils/visualization.py](src/walle_vision/utils/visualization.py)
- [src/walle_vision/utils/labels.py](src/walle_vision/utils/labels.py)

## Documentation matérielle

https://www.thundercomm.com/rubik-pi-3/en/docs/about-rubikpi/
