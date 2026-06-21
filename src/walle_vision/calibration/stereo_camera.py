"""
stereo_camera.py
Capture synchronisee de deux flux camera (qtiqmmfsrc) via GStreamer + OpenCV.
 
Chaque camera est lue en continu dans un thread dedie pour qu'une camera
lente ne bloque pas l'autre. get_synced_pair() renvoie la derniere paire
de frames disponible, avec l'ecart de timestamp entre les deux.
 
Limite importante : la synchronisation est "best effort", basee sur
l'horloge systeme au moment ou OpenCV recoit chaque frame (pas sur un
trigger hardware). Pour des objets statiques ou lents, c'est suffisant.
Pour des objets rapides, il faudra explorer une synchro materielle ou
reduire max_skew au prix de davantage de frames rejetees.
"""
 
import threading
import time
 
import cv2
 
 
def build_pipeline(camera_id, width=1280, height=720, framerate=30):
    """Construit la chaine GStreamer qtiqmmfsrc -> BGR pour OpenCV.
 
    Reprend le format confirme fonctionnel (NV12, 1280x720, 30fps) et
    ajoute la conversion vers BGR + un appsink adapte a la lecture par
    cv2.VideoCapture.
    """
    return (
        f"qtiqmmfsrc camera={camera_id} ! "
        f"video/x-raw,format=NV12,width={width},height={height},framerate={framerate}/1 ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )
 
 
class CameraStream:
    """Lit en continu une camera dans un thread separe."""
 
    def __init__(self, camera_id, width=1280, height=720, framerate=30):
        self.pipeline = build_pipeline(camera_id, width, height, framerate)
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Impossible d'ouvrir la camera {camera_id}.\n"
                f"Pipeline utilise :\n  {self.pipeline}\n"
                "Testez d'abord ce pipeline avec gst-launch-1.0 (en remplacant "
                "appsink par un sink visuel) pour verifier qu'il fonctionne seul."
            )
 
        self.frame = None
        self.timestamp = 0.0
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
 
    def _update(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
                    self.timestamp = time.time()
 
    def read(self):
        with self.lock:
            if self.frame is None:
                return None, None
            return self.frame.copy(), self.timestamp
 
    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()
 
 
class StereoCamera:
    """Regroupe les deux cameras et fournit des paires synchronisees."""
 
    def __init__(self, left_id=0, right_id=1, width=1280, height=720, framerate=30):
        self.left = CameraStream(left_id, width, height, framerate)
        self.right = CameraStream(right_id, width, height, framerate)
        time.sleep(0.5)  # laisser les threads remplir le premier frame
 
    def get_synced_pair(self, max_skew=0.05):
        """
        Renvoie (frame_gauche, frame_droite, ecart_secondes).
        Renvoie (None, None, ecart) si l'ecart depasse max_skew.
        """
        fl, tl = self.left.read()
        fr, tr = self.right.read()
        if fl is None or fr is None:
            return None, None, None
        skew = abs(tl - tr)
        if skew > max_skew:
            return None, None, skew
        return fl, fr, skew
 
    def stop(self):
        self.left.stop()
        self.right.stop()
 
 
def capture_single_frame(camera_id, width=1280, height=720, framerate=30, warmup_frames=5):
    """
    Ouvre UNE SEULE camera, recupere une frame, puis referme la camera.
 
    Utile sur ce materiel ou deux flux qtiqmmfsrc simultanes (chemin
    memoire systeme NV12) ne sont pas supportes : on ouvre les cameras
    une a la fois plutot qu'en parallele. warmup_frames jette les
    premieres frames (l'exposition automatique doit se stabiliser).
    """
    pipeline = build_pipeline(camera_id, width, height, framerate)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(
            f"Impossible d'ouvrir la camera {camera_id}.\n"
            f"Pipeline utilise :\n  {pipeline}"
        )
 
    frame = None
    try:
        for _ in range(warmup_frames):
            ok, f = cap.read()
            if ok:
                frame = f
    finally:
        cap.release()
 
    if frame is None:
        raise RuntimeError(f"Aucune frame recue de la camera {camera_id}")
    return frame
