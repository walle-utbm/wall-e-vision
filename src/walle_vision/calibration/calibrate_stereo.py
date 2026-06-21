"""
calibrate_stereo.py
Calibration intrinseque (par camera) puis extrinseque (stereo) a partir
des paires d'images du damier capturees par capture_calib_images.py.

Resultat : stereo_calib.npz, contenant tout ce qu'il faut pour rectifier
et trianguler en temps reel (utilise par live_stereo.py).

Ce script ne touche pas aux cameras : il travaille uniquement sur les
images deja sauvegardees sur disque.
"""

import glob
import os

import cv2
import numpy as np

# --- A ADAPTER selon votre damier ---
CHESSBOARD_SIZE = (9, 7)   # nombre de coins INTERNES (colonnes, lignes)
SQUARE_SIZE_MM = 19.0      # taille reelle d'une case, mesuree avec une regle
IMAGE_DIR = "calib_images"

# Indices de poses a exclure (identifies via diagnose_pairs.py -- damier
# qui a bouge entre la prise gauche et la prise droite, ou autre probleme
# specifique a cette pose). Exemple : {15, 16, 17}
EXCLUDE_INDICES = {15, 16, 17}


def _index_from_path(path):
    """Extrait l'index numerique 'NNN' depuis un nom de fichier img_NNN.png."""
    stem = os.path.splitext(os.path.basename(path))[0]  # "img_007"
    return int(stem.split("_")[-1])


def find_corners(image_paths, pattern_size):
    """Detecte les coins du damier sur une liste d'images.

    Renvoie un dict {index_fichier: (objp, corners)} -- l'index permet
    de ne reassembler que les paires gauche/droite qui correspondent
    REELLEMENT a la meme prise de vue (cf. bug corrige : avant, on
    associait juste les N premieres reussites de chaque cote, ce qui
    desynchronise tout des qu'une image differente echoue a gauche et
    a droite).
    """
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    results = {}
    image_size = None

    for path in image_paths:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if found:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            results[_index_from_path(path)] = (objp, corners)
        else:
            print(f"  Damier non detecte : {path}")

    return results, image_size


def main():
    left_paths = sorted(glob.glob(f"{IMAGE_DIR}/left/*.png"))
    right_paths = sorted(glob.glob(f"{IMAGE_DIR}/right/*.png"))
    assert len(left_paths) == len(right_paths), "Nombre d'images gauche/droite different"
    print(f"{len(left_paths)} paires trouvees, recherche des coins du damier...")

    print("\n-- Camera gauche --")
    results_L, size_L = find_corners(left_paths, CHESSBOARD_SIZE)
    print("\n-- Camera droite --")
    results_R, size_R = find_corners(right_paths, CHESSBOARD_SIZE)

    # Ne garder que les indices ou le damier a ete detecte des DEUX cotes
    # (cf. correction du bug : avant on prenait juste les N premieres
    # reussites de chaque liste, sans verifier que c'etait la meme pose).
    common_idx_all = sorted(set(results_L) & set(results_R))
    common_idx = [i for i in common_idx_all if i not in EXCLUDE_INDICES]
    excluded_present = sorted(set(common_idx_all) & EXCLUDE_INDICES)
    n_pairs = len(common_idx)

    print(f"\n{n_pairs} paires valides (damier detecte des deux cotes, "
          f"sur la MEME prise de vue).")
    if excluded_present:
        print(f"  ({len(excluded_present)} pose(s) exclue(s) manuellement "
              f"via EXCLUDE_INDICES : {excluded_present})")
    if n_pairs < 10:
        print("ATTENTION : trop peu de paires valides. Reprenez des photos "
              "avec un meilleur eclairage ou un damier plus net.")
        return

    objp_L = [results_L[i][0] for i in common_idx]
    imgp_L = [results_L[i][1] for i in common_idx]
    objp_R = [results_R[i][0] for i in common_idx]
    imgp_R = [results_R[i][1] for i in common_idx]

    image_size = size_L

    print("\nCalibration intrinseque camera gauche...")
    err_L, mtx_L, dist_L, _, _ = cv2.calibrateCamera(objp_L, imgp_L, image_size, None, None)
    print(f"  Erreur de reprojection : {err_L:.3f} px (viser < 0.5 px)")

    print("Calibration intrinseque camera droite...")
    err_R, mtx_R, dist_R, _, _ = cv2.calibrateCamera(objp_R, imgp_R, image_size, None, None)
    print(f"  Erreur de reprojection : {err_R:.3f} px (viser < 0.5 px)")

    print("\nCalibration stereo (extrinseque)...")
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    ret, mtx_L, dist_L, mtx_R, dist_R, R, T, E, F = cv2.stereoCalibrate(
        objp_L, imgp_L, imgp_R, mtx_L, dist_L, mtx_R, dist_R,
        image_size, criteria=criteria, flags=flags
    )
    print(f"  Erreur de reprojection stereo : {ret:.3f} px (viser < 1 px)")
    print(f"  Distance entre cameras mesuree (norme de T) : {np.linalg.norm(T):.1f} mm")
    print("  -> Comparez cette valeur a la distance que vous avez mesuree "
          "physiquement sur la regle. Un grand ecart indique un probleme "
          "de calibration ou de detection du damier.")

    print("\nRectification...")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_L, dist_L, mtx_R, dist_R, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    map1x, map1y = cv2.initUndistortRectifyMap(mtx_L, dist_L, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(mtx_R, dist_R, R2, P2, image_size, cv2.CV_32FC1)

    np.savez(
        "stereo_calib.npz",
        mtx_L=mtx_L, dist_L=dist_L, mtx_R=mtx_R, dist_R=dist_R,
        R=R, T=T, Q=Q, image_size=image_size,
        map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y,
    )
    print("\nSauvegarde : stereo_calib.npz")
    print("Vous pouvez maintenant lancer live_stereo.py")


if __name__ == "__main__":
    main()
