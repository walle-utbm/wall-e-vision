"""
capture_calib_images.py (version sequentielle)
Capture des paires d'images du damier pour la calibration stereo.

IMPORTANT : sur ce materiel (RUBIK Pi 3 / QCM6490), deux flux qtiqmmfsrc
simultanes en memoire systeme (le format qu'on utilise) ne sont PAS
supportes -- la deuxieme camera reste bloquee indefiniment si la
premiere est encore ouverte. Ce script n'ouvre donc JAMAIS les deux
cameras en meme temps : il capture la gauche, referme la camera, capture
la droite, referme la camera.

Consequence pratique : le damier doit rester IMMOBILE pendant les
quelques secondes qui separent les deux prises. C'est sans probleme pour
de la calibration (contrairement a de la stereo en temps reel).

Avant de lancer ce script :
  - Imprimez un damier (ex: 10x7 cases = 9x6 coins internes), collez-le
    sur un support RIGIDE et bien plat (carton epais, planche...).
  - Mesurez precisement la taille reelle d'une case (en mm) avec une
    regle -> a reporter dans calibrate_stereo.py (SQUARE_SIZE_MM).

Utilisation :
  - Entree : capturer une paire (apres avoir positionne le damier)
  - q puis Entree : quitter

Conseils de prise de vue :
  - Variez l'angle et l'orientation du damier (incline, tourne...).
  - Variez la distance (proche / loin dans la plage de travail prevue).
  - Couvrez les bords et les coins de l'image, pas seulement le centre.
  - Visez 20 a 30 paires valides au minimum.
"""

import os

import cv2

from stereo_camera import capture_single_frame

OUTPUT_DIR = "calib_images"
LEFT_ID = 0
RIGHT_ID = 1
WIDTH, HEIGHT, FPS = 1280, 720, 30


def main():
    os.makedirs(os.path.join(OUTPUT_DIR, "left"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "right"), exist_ok=True)

    count = 0
    print("Positionnez le damier, gardez-le IMMOBILE, puis appuyez sur Entree.")
    print("(tapez 'q' puis Entree pour quitter)\n")

    while True:
        cmd = input(f"[{count}] Entree = capturer | q = quitter > ").strip().lower()
        if cmd == 'q':
            break

        print("  Ouverture camera gauche...")
        frame_l = capture_single_frame(LEFT_ID, WIDTH, HEIGHT, FPS)
        print("  Camera gauche refermee. Ouverture camera droite...")
        frame_r = capture_single_frame(RIGHT_ID, WIDTH, HEIGHT, FPS)
        print("  Camera droite refermee.")

        left_path = os.path.join(OUTPUT_DIR, "left", f"img_{count:03d}.png")
        right_path = os.path.join(OUTPUT_DIR, "right", f"img_{count:03d}.png")
        cv2.imwrite(left_path, frame_l)
        cv2.imwrite(right_path, frame_r)

        print(f"  Paire {count} sauvegardee : {left_path} / {right_path}\n")
        count += 1

    print(f"\nTermine. {count} paires sauvegardees dans {OUTPUT_DIR}/")
    if count < 20:
        print("Conseil : visez au moins 20-30 paires pour une bonne calibration.")


if __name__ == "__main__":
    main()
