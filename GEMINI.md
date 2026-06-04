# Documentation Architecturale : Wall-E Vision

*Rédigé par un développeur système embarqué.*

Ce document (`gemini.md`) détaille l'architecture logicielle, les choix techniques et la structure du dépôt `wall-e-vision`. Conçu pour des environnements contraints (Raspberry Pi 4, Rubik Pi 3), ce projet se distingue par sa forte modularité, sa gestion efficace des ressources (threads, buffers) et son agnosticisme matériel.

---

## 1. Philosophie et Architecture Globale

Le projet repose sur une **architecture orientée pipeline** et pilotée par la configuration (`config.yaml`). Le point d'entrée (`main.py`) est dépourvu de logique métier : il charge la configuration et instancie l'un des trois pipelines disponibles.

La base de code est segmentée en **4 couches isolées (SoC - Separation of Concerns)** :
1. **`core`** : Acquisition matérielle (Capteurs CSI/USB).
2. **`ai`** : Traitement neuronal (Inférence YOLO, segmentation).
3. **`network`** : Communication bas niveau (Protocole TCP sur-mesure).
4. **`pipelines`** : Orchestration des flux de données (Standalone, Client, Serveur).

---

## 2. Abstraction Matérielle (Hardware Targets)

Le système est conçu pour s'adapter dynamiquement à la cible matérielle, résolvant la fragmentation classique des projets robotiques.

| Plateforme | Caméra Backend | Accélération / Modèle | Spécificités Embarquées |
| :--- | :--- | :--- | :--- |
| **Rubik Pi 3** | GStreamer (`qtiqmmfsrc`) | ONNX (`best.onnx`) | Traitement NV12 optimisé, pipeline matériel. |
| **Raspberry Pi 4** | Picamera2 / V4L2 | PyTorch (`best.pt`) | Tuning des threads CPU via `torch.set_num_threads()`. |
| **PC (Serveur)** | None (Réseau) | PyTorch (`best.pt`) | Mode "Headless" caméra, écoute TCP. |

---

## 3. Analyse Détaillée des Composants

### 3.1 Couche Core : `camera.py` (L'Acquisition)
La capture vidéo en milieu embarqué est critique. Le fichier implémente le pattern **Factory** (`CameraStream.create`) et masque la complexité de l'API sous-jacente.
* **GStreamer (`_GStreamerCamera`)** : Utilisé pour le Rubik Pi 3. Instancie un pipeline optimisé avec `appsink`. Utilisation de `leaky=downstream` et `max-buffers=1` pour éviter la latence (drop des vieilles frames).
* **Picamera2 (`_Picamera2Camera`)** : Allocation directe de buffers RGB/BGR. Prise en charge des frames de chauffe (`warmup_frames`) pour la stabilisation de l'Auto-Exposure / Auto-White-Balance.
* **OpenCV (`_OpenCVCamera`)** : Fallback V4L2 classique, avec forçage de `CAP_PROP_BUFFERSIZE` à 1 pour limiter le buffer bloat.

### 3.2 Couche IA : `detector.py` (L'Inférence)
Encapsulation robuste du modèle YOLO.
* **Gestion du CPU** : `_configure_torch_runtime()` limite intelligemment le nombre de threads inter/intra-op pour éviter le context-switching excessif sur ARM.
* **Fallback Automatique** : Si le `.pt` manque, cherche le `.onnx` (et inversement).
* **Traitement Spatial** : Calcul natif du *pickup_point* via le centre de gravité (moments de l'image `cv2.moments`) si un masque de segmentation est disponible. Clipping strict des Bounding Boxes aux dimensions de la frame pour éviter les segfaults dans le traitement aval.

### 3.3 Couche Réseau : `transport.py` (Le Streaming TCP)
Évite les lourdeurs de ROS ou d'un broker externe (MQTT/ZeroMQ) au profit d'un protocole binaire TCP ultra-léger et déterministe.
* **Framing** : `[Taille Header (4 octets)] + [Header JSON] + [Taille Body (4 octets)] + [Body Binaire (JPEG)]`.
* **Endienness** : Utilisation de `struct.pack("!I")` (Big-Endian) pour la compatibilité inter-architectures (x86_64 vs ARM64).
* **Thread-Safety** : Utilisation de `threading.Lock()` sur les sockets lors de l'envoi (`sendall`).

---

## 4. Les Pipelines d'Exécution

Le composant central de l'exécution (ex: `pipeline_edge.py`) met en œuvre les bonnes pratiques du temps-réel souple (soft real-time) :

1.  **Découplage Producteur-Consommateur** :
    * Un `capture_thread` (Daemon) lit le capteur à la fréquence max.
    * Il pousse dans une `queue.Queue(maxsize=2)`.
    * **Gestion de la Contre-Pression (Backpressure)** : Si la queue est pleine (l'inférence est trop lente), la frame la plus ancienne est *dropped* (`get_nowait()`). Cela garantit que l'IA analyse toujours l'image la plus récente.
2.  **Throttling CPU** : L'option `infer_every_n_frames` permet de sauter des frames pour l'inférence, économisant la batterie et réduisant la chauffe thermique (Thermal Throttling très fréquent sur RPi4).
3.  **Télémétrie Intégrée** : Affichage périodique du FPS glissant (toutes les 5 secondes).
4.  **Logging Structuré** : Les détections sont sérialisées en `.jsonl` (JSON Lines), format idéal pour un ajout séquentiel (`append`) résistant aux coupures d'alimentation impromptues (SD card corruption).

---

## 5. Synthèse des Bonnes Pratiques Observées

En tant que développeur embarqué, voici pourquoi cette architecture est saine :
* **Zéro couplage** entre la source vidéo et le modèle d'IA.
* **Gestion agressive de la latence** (`buffer=1`, `leaky queues`, `TCP_NODELAY` sur les sockets).
* **Tolérance aux pannes** : Auto-fallback des modèles, gestion propre des signaux de fermeture (`threading.Event()`, `try/finally` pour relâcher le capteur `camera.close()`).
* **Configuration centralisée** : Le fichier `config.yaml` permet de changer le comportement du robot sans recompiler ni modifier le code, facilitant les tests sur le terrain (field-testing).
