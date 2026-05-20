# wall-e-vision

Architecture unifiée pour la vision du robot Wall-E. Le dépôt expose un point d'entrée unique, [main.py](main.py), qui lit [config.yaml](config.yaml) puis instancie automatiquement le bon pipeline selon le matériel, le type de caméra et le mode d'exécution.

## Vue d'ensemble

Le refactor sépare le projet en quatre couches:

- `core` pour l'acquisition caméra.
- `ai` pour le chargement du modèle YOLO et l'inférence.
- `network` pour l'échange TCP des frames et des résultats.
- `pipelines` pour assembler les briques selon le mode choisi.

## Modes d'exécution

### `edge_standalone`

Mode d'inférence locale. La capture vidéo et l'inférence YOLO tournent sur la même machine.

Cas d'usage:

- Raspberry Pi 4 avec modèle `.pt`.
- Rubik Pi 3 avec modèle `.onnx`.

Comportement:

- la caméra est consommée dans un thread dédié;
- les frames sont traitées en continu par le détecteur local;
- les résultats peuvent être écrits dans `outputs/detections.jsonl`;
- les images annotées peuvent être enregistrées dans `outputs/predict/`;
- l'affichage OpenCV est optionnel via `runtime.show`.

### `stream_client`

Mode de capture et de streaming.
- le pc et le raspberry doivent être connectés au même réseau. 
- penser à changer l'ip du serveur donc du pc.

Cas d'usage:

- Raspberry Pi qui n'exécute pas l'IA localement;
- envoi des frames JPEG vers le PC via TCP.

Comportement:

- la caméra capture localement dans un thread dédié;
- les frames sont compressées en JPEG avant l'envoi;
- le client maintient une file d'in-flight limitée par `network.max_inflight_frames`;
- les résultats renvoyés par le PC sont stockés dans `outputs/remote_results.jsonl`;
- l'affichage peut superposer un résumé réseau et latence.

### `stream_server`

Mode serveur PC.

- le pc et le raspberry doivent être connectés au même réseau. 

Cas d'usage:

- PC de bureau recevant les frames depuis le Raspberry Pi client;
- inférence YOLO côté serveur sur les frames reçues.

Comportement:

- le serveur écoute sur `network.host:network.port`;
- il reçoit des frames JPEG encapsulées dans un protocole TCP simple;
- il décode, infère, tracke puis renvoie les résultats au client;
- les résultats sont archivés dans `outputs/detections.jsonl`;
- les images annotées sont sauvegardées dans `outputs/predict/`.

## Types de caméra

Le backend caméra est centralisé dans [src/walle_vision/core/camera.py](src/walle_vision/core/camera.py).

### `picamera`

Destiné aux caméras CSI.

Comportement:

- tente d'abord `Picamera2` quand elle est disponible;
- sur Rubik Pi 3, le backend GStreamer peut être utilisé;
- le paramètre `camera.warmup_frames` permet de stabiliser l'exposition avant la boucle principale.

### `usb`

Caméra USB classique via OpenCV.

Comportement:

- ouvre la source avec `cv2.VideoCapture`;
- applique `width`, `height`, `fps` et un buffer minimal;
- convient aux webcams, UVC et périphériques exposés par index ou par URL.

### `none`

Mode sans caméra réelle.

Usage:

- utile pour certains tests réseau ou pour des scénarios où la caméra est gérée ailleurs;
- la création de frames est volontairement interdite dans ce backend.

## Détection YOLO

Le moteur d'inférence est unifié dans [src/walle_vision/ai/detector.py](src/walle_vision/ai/detector.py).

Points importants:

- le modèle est choisi depuis `models` dans [config.yaml](config.yaml);
- `rpi4` et `pc` pointent vers un modèle `.pt`;
- `rubikpi3` pointe vers un modèle `.onnx`;
- si le fichier configuré n'existe pas, le détecteur tente l'extension alternative compatible;
- les paramètres `detector.conf_threshold`, `detector.iou_threshold`, `detector.image_size` et `detector.max_detections` pilotent l'inférence;
- le tracking temporel est appliqué après l'inférence pour stabiliser les résultats.

## Pipelines

### [pipeline_edge.py](src/walle_vision/pipelines/pipeline_edge.py)

Pipeline complet pour les modes standalone.

Responsabilités:

- créer la caméra;
- charger le détecteur local;
- exécuter le tracker;
- produire les résultats et les images annotées.

Spécificités:

- la capture est découplée de l'inférence via une file mémoire courte;
- `runtime.infer_every_n_frames` permet de réduire la charge CPU;
- `runtime.save_every_n_frames` contrôle la fréquence d'export des images annotées;
- `runtime.camera_test_mode` permet de sauvegarder périodiquement des frames brutes pour diagnostic.

### [pipeline_client.py](src/walle_vision/pipelines/pipeline_client.py)

Pipeline client du mode streaming.

Responsabilités:

- capturer les frames;
- les compresser en JPEG;
- les envoyer au PC;
- recevoir les résultats et mettre à jour les métriques réseau.

Spécificités:

- le client limite le nombre de frames en vol pour éviter la saturation mémoire;
- les reconnexions utilisent `network.reconnect_delay_sec`;
- les timings de capture, d'envoi et de retour sont affichés dans la sortie console;
- le fichier `outputs/remote_results.jsonl` centralise les réponses du serveur.

### [pipeline_server.py](src/walle_vision/pipelines/pipeline_server.py)

Pipeline serveur PC.

Responsabilités:

- écouter les connexions entrantes;
- décoder les frames JPEG;
- exécuter YOLO localement;
- renvoyer les résultats au client.

Spécificités:

- le serveur traite les clients séquentiellement;
- la fermeture propre passe par les signaux SIGINT et SIGTERM;
- les résultats sont serialisés dans un format JSON stable partagé avec le client.

## Arborescence utile

- [src/walle_vision/core/camera.py](src/walle_vision/core/camera.py) contient la factory caméra et les backends OpenCV, Picamera2 et GStreamer.
- [src/walle_vision/ai/detector.py](src/walle_vision/ai/detector.py) contient le chargement YOLO `.pt` et `.onnx`.
- [src/walle_vision/network/transport.py](src/walle_vision/network/transport.py) contient le protocole TCP et l'encodage JPEG.
- [src/walle_vision/pipelines/](src/walle_vision/pipelines/) contient les trois pipelines métiers.
- [src/walle_vision/utils/visualization.py](src/walle_vision/utils/visualization.py) gère l'affichage des détections.
- [src/walle_vision/utils/labels.py](src/walle_vision/utils/labels.py) gère le mapping métier des classes.

## Configuration

Le fichier [config.yaml](config.yaml) est la source de vérité.

### Clés principales

- `hardware`: `rpi4`, `rubikpi3` ou `pc`.
- `camera_type`: `picamera`, `usb` ou `none`.
- `mode`: `edge_standalone`, `stream_client` ou `stream_server`.

### Sections secondaires

- `paths`: emplacements des modèles et des sorties.
- `models`: nom du modèle à utiliser par matériel.
- `camera`: peut définir un profil commun et des surcharges par matériel (`rubikpi3`, `rpi4`, `pc`).
- `detector`: idem pour les seuils, `image_size` et les options YOLO.
- `runtime`: affichage, fréquence d'inférence et mode test caméra.
- `network`: adresse serveur, client, qualité JPEG et limites d'in-flight.

### Exemple de choix par profil

```yaml
# Raspberry Pi 4 autonome
hardware: rpi4
camera_type: picamera
mode: edge_standalone

# Rubik Pi 3 autonome
hardware: rubikpi3
camera_type: picamera
mode: edge_standalone

# Client Raspberry Pi qui stream vers le PC
hardware: rpi4
camera_type: picamera
mode: stream_client

# Serveur PC
hardware: pc
camera_type: none
mode: stream_server
```

## Sorties générées

Selon le mode, le projet écrit principalement:

- `outputs/detections.jsonl` pour les résultats locaux ou serveur;
- `outputs/remote_results.jsonl` pour les réponses reçues par le client;
- `outputs/predict/` pour les images annotées;
- `outputs/camera_test/` quand `runtime.camera_test_mode` est activé.

## Dépendances

Les dépendances sont séparées par cible:

- [requirements/base.txt](requirements/base.txt) pour le socle commun;
- [requirements/edge.txt](requirements/edge.txt) pour les modes embarqués;
- [requirements/pc.txt](requirements/pc.txt) pour le serveur PC.

Le projet s'appuie notamment sur `ultralytics`, `opencv-python`, `numpy` et `PyYAML`. Selon la plateforme, `picamera2` et le support GStreamer peuvent aussi être nécessaires.

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

Le lancement se fait toujours depuis la racine:

```bash
source .venv/bin/activate
python main.py
```

Le code charge [config.yaml](config.yaml), construit automatiquement le pipeline approprié puis lance sa méthode `run()`.

## Notes d'architecture

- Le point d'entrée ne contient pas de logique métier, uniquement le chargement de configuration et la sélection du pipeline.
- La caméra et l'inférence sont découplées afin de ne pas bloquer la capture pendant le traitement.
- Le streaming réseau conserve un format commun entre client et serveur pour faciliter le debug et l'archivage.
- Le `tracking` est appliqué après l'inférence pour rendre les sorties plus stables d'une frame à l'autre.