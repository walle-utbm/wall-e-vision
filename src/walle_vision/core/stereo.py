from __future__ import annotations
 
"""Estimation de profondeur stereo, calculee ponctuellement (pas en flux continu).
 
Contrainte materielle (etablie experimentalement sur RUBIK Pi 3 / QCM6490) :
qtiqmmfsrc ne supporte pas deux flux camera CSI simultanes -- la seconde
camera reste bloquee indefiniment si la premiere est deja active. La
camera de detection (camera.rubikpi3, "gauche" dans la convention de
calibration) tourne en continu pendant tout le pipeline ; ce module ne
gere PAS sa mise en pause (c'est la responsabilite de l'appelant, cf.
PipelineEdge._run_stereo_capture). Il s'occupe uniquement de :
  - ouvrir brievement la seconde camera ("droite"), prendre une frame, la
    refermer ;
  - rectifier la paire d'images avec la calibration ;
  - calculer une carte de disparite ;
  - transformer des points d'interet (coins de bbox, centre, point de
    ramassage), exprimes dans l'image ORIGINALE de la camera de
    detection, en position 3D (X, Y, Z en mm, repere camera gauche).
 
Les points sont transformes individuellement via cv2.undistortPoints
(avec R1/P1 issus de la calibration) plutot que de remapper toute
l'image -- on n'a besoin que de quelques points, pas d'une carte dense.
"""
 
from dataclasses import dataclass
from pathlib import Path
 
import cv2
import numpy as np
 
from ..config import CameraSettings
from .camera import CameraStream
 
Point2D = tuple[float, float]
Point3D = tuple[float, float, float]
 
 
@dataclass(slots=True)
class StereoCalibration:
    mtx_left: np.ndarray
    dist_left: np.ndarray
    R1: np.ndarray
    P1: np.ndarray
    Q: np.ndarray
    map1x: np.ndarray
    map1y: np.ndarray
    map2x: np.ndarray
    map2y: np.ndarray
    baseline_mm: float
 
    @classmethod
    def load(cls, path: Path) -> "StereoCalibration":
        if not path.exists():
            raise FileNotFoundError(
                f"Fichier de calibration stereo introuvable : {path}. "
                "Verifiez stereo.calib_path dans config.yaml."
            )
        data = np.load(str(path))
        required = {"mtx_L", "dist_L", "R1", "P1", "Q", "map1x", "map1y", "map2x", "map2y", "T"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(
                f"{path} ne contient pas les champs {sorted(missing)}. "
                "Relancez calibrate_stereo.py avec la version a jour "
                "(qui sauvegarde R1/P1/R2/P2)."
            )
        return cls(
            mtx_left=data["mtx_L"],
            dist_left=data["dist_L"],
            R1=data["R1"],
            P1=data["P1"],
            Q=data["Q"],
            map1x=data["map1x"],
            map1y=data["map1y"],
            map2x=data["map2x"],
            map2y=data["map2y"],
            baseline_mm=float(np.linalg.norm(data["T"])),
        )
 
 
def build_stereo_matcher() -> cv2.StereoSGBM:
    """Memes parametres que valides dans le projet de calibration
    (calib/live_stereo.py) -- mode SGBM complet car on n'a plus de
    contrainte de temps reel ici (une mesure ponctuelle, pas un flux)."""
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=256,
        blockSize=7,
        P1=8 * 3 * 7 ** 2,
        P2=32 * 3 * 7 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=150,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM,
    )
 
 
def _to_rectified_points(points_xy: list[Point2D], calib: StereoCalibration) -> np.ndarray:
    """Transforme des points de l'image ORIGINALE (camera de detection,
    "gauche") en coordonnees dans l'image RECTIFIEE -- sans remapper
    toute l'image, juste les points qui nous interessent."""
    src = np.array(points_xy, dtype=np.float64).reshape(-1, 1, 2)
    rectified = cv2.undistortPoints(src, calib.mtx_left, calib.dist_left, R=calib.R1, P=calib.P1)
    return rectified.reshape(-1, 2)
 
 
def _sample_disparity(disparity: np.ndarray, x: float, y: float, window: int) -> float | None:
    """Mediane des disparites VALIDES dans une petite fenetre autour de
    (x, y) -- plus robuste qu'un pixel unique sur une carte de disparite
    bruitee (cf. surfaces peu texturees)."""
    h, w = disparity.shape
    xi, yi = int(round(x)), int(round(y))
    half = window // 2
    x0, x1 = max(0, xi - half), min(w, xi + half + 1)
    y0, y1 = max(0, yi - half), min(h, yi + half + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    patch = disparity[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid))
 
 
def _disparity_to_3d(x: float, y: float, disparity: float, Q: np.ndarray) -> Point3D:
    homog = Q @ np.array([x, y, disparity, 1.0])
    homog /= homog[3]
    return float(homog[0]), float(homog[1]), float(homog[2])
 
 
class StereoEstimator:
    def __init__(
        self,
        hardware: str,
        camera_type: str,
        camera2_settings: CameraSettings,
        calib_path: Path,
        sample_window: int = 5,
    ) -> None:
        self.hardware = hardware
        self.camera_type = camera_type
        self.camera2_settings = camera2_settings
        self.calib = StereoCalibration.load(calib_path)
        self.matcher = build_stereo_matcher()
        self.sample_window = sample_window
 
    def capture_second_frame(self) -> np.ndarray:
        """Ouvre la seconde camera, recupere UNE frame, la referme.
 
        A appeler UNIQUEMENT quand la camera de detection est deja
        arretee (contrainte materielle : pas de flux CSI simultane)."""
        cam2 = CameraStream.create(self.hardware, self.camera_type, self.camera2_settings)
        try:
            _, _, frame2 = next(cam2.frames())
        finally:
            cam2.close()
        return frame2
 
    def estimate_points(
        self,
        frame_left: np.ndarray,
        frame_right: np.ndarray,
        points_xy: list[Point2D],
    ) -> list[Point3D | None]:
        """Position 3D (X, Y, Z en mm, repere camera gauche) pour chaque
        point de `points_xy` (coordonnees dans l'image ORIGINALE, non
        rectifiee, de la camera de detection). None si la profondeur n'a
        pas pu etre estimee a cet endroit (zone sans disparite valide)."""
        rect_left = cv2.remap(frame_left, self.calib.map1x, self.calib.map1y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(frame_right, self.calib.map2x, self.calib.map2y, cv2.INTER_LINEAR)
 
        gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
 
        disparity = self.matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
 
        rectified_points = _to_rectified_points(points_xy, self.calib)
 
        results: list[Point3D | None] = []
        for x_rect, y_rect in rectified_points:
            d = _sample_disparity(disparity, x_rect, y_rect, self.sample_window)
            if d is None or d <= 0:
                results.append(None)
                continue
            results.append(_disparity_to_3d(x_rect, y_rect, d, self.calib.Q))
        return results
