"""
live_stereo.py (version sequentielle)
Mesure de profondeur stereo SANS flux video simultane : sur ce materiel,
qtiqmmfsrc ne supporte pas deux flux camera en parallele (confirme par
plusieurs tests : deux process separes, un seul process a deux sources,
pipeline GBM/UBWC, reglage multiCameraLogicalXMLFile -- rien n'a
fonctionne). On capture donc chaque scene en deux temps (gauche, puis
droite), exactement comme pour la calibration.

IMPORTANT - consequences pratiques :
  - Latence de 1 a 3 secondes entre les deux prises (ouverture/fermeture
    de chaque camera a chaque fois). Convient pour une scene statique ou
    qui bouge lentement, PAS pour suivre un objet rapide en temps reel.
  - Aucun affichage graphique disponible en SSH -> ce script sauvegarde
    les images (image gauche rectifiee avec le point mesure marque,
    image droite rectifiee, carte de disparite coloree) sur disque a
    chaque mesure. Recuperez-les avec scp pour les regarder. La mesure
    de profondeur elle-meme est affichee directement dans le terminal.

Utilisation :
  - Entree seule       : capturer une scene, mesurer le CENTRE de l'image
  - "x,y" puis Entree  : capturer une scene, mesurer le pixel (x,y)
                         (coordonnees dans l'image rectifiee, ex: 640,360)
  - q puis Entree      : quitter
"""

import os

import cv2
import numpy as np

from stereo_camera import capture_single_frame

LEFT_ID = 0
RIGHT_ID = 1
WIDTH, HEIGHT, FPS = 1280, 720, 30
OUTPUT_DIR = "live_output"


def build_stereo_matcher():
    # Memes parametres que precedemment -- a ajuster selon votre scene
    # (cf. README : numDisparities pour les objets proches, blockSize
    # pour le lissage).
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=288,       # multiple de 16 -- augmentez encore si les
                                   # objets proches restent "non fiables"
        blockSize=7,
        P1=8 * 3 * 7 ** 2,
        P2=32 * 3 * 7 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    calib = np.load("stereo_calib.npz")
    map1x, map1y = calib["map1x"], calib["map1y"]
    map2x, map2y = calib["map2x"], calib["map2y"]
    Q = calib["Q"]

    stereo = build_stereo_matcher()

    print("Capture sequentielle -- la scene doit rester immobile entre les deux prises.")
    print("Entree = mesurer le centre | 'x,y' = mesurer un pixel precis | q = quitter\n")

    while True:
        cmd = input("> ").strip().lower()
        if cmd == 'q':
            break

        target = None
        if cmd:
            try:
                x_str, y_str = cmd.split(',')
                target = (int(x_str.strip()), int(y_str.strip()))
            except ValueError:
                print("  Format invalide, attendu 'x,y' (ex: 640,360). Ignore.\n")
                continue

        print("  Ouverture camera gauche...")
        frame_l = capture_single_frame(LEFT_ID, WIDTH, HEIGHT, FPS)
        print("  Camera gauche refermee. Ouverture camera droite...")
        frame_r = capture_single_frame(RIGHT_ID, WIDTH, HEIGHT, FPS)
        print("  Camera droite refermee. Calcul de la disparite...")

        rect_l = cv2.remap(frame_l, map1x, map1y, cv2.INTER_LINEAR)
        rect_r = cv2.remap(frame_r, map2x, map2y, cv2.INTER_LINEAR)

        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

        disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
        points_3d = cv2.reprojectImageTo3D(disparity, Q)

        h, w = rect_l.shape[:2]
        if target is None:
            target = (w // 2, h // 2)
        x, y = target
        if not (0 <= x < w and 0 <= y < h):
            print(f"  Pixel ({x},{y}) hors de l'image ({w}x{h}). Ignore.\n")
            continue

        X, Y, Z = points_3d[y, x]

        marked = rect_l.copy()
        cv2.circle(marked, (x, y), 6, (0, 0, 255), 2)

        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_vis = cv2.applyColorMap(disp_vis.astype(np.uint8), cv2.COLORMAP_JET)

        cv2.imwrite(os.path.join(OUTPUT_DIR, "rect_left.png"), marked)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "rect_right.png"), rect_r)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "disparity.png"), disp_vis)

        if np.isfinite(Z) and Z > 0:
            print(f"  Pixel ({x},{y}) -> X={X:.0f} mm, Y={Y:.0f} mm, Z={Z:.0f} mm")
        else:
            print(f"  Pixel ({x},{y}) -> profondeur non fiable (pas de disparite valide ici)")
        print(f"  Images sauvegardees dans {OUTPUT_DIR}/ (rect_left.png marque le point mesure)\n")

    print("Termine.")


if __name__ == "__main__":
    main()
