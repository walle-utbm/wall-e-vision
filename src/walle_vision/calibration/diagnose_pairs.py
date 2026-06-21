"""
diagnose_pairs.py
Diagnostic : detecte les poses ou le damier semble avoir bouge entre la
prise gauche et la prise droite (cause probable d'une calibration stereo
qui explose alors que chaque camera individuellement est bonne).

Logique : les deux cameras partagent a peu pres la meme orientation et
une base horizontale. Pour une MEME pose immobile, on attend donc :
  - une difference verticale (Y) limitee entre coins correspondants
    (quelques pixels a quelques dizaines de pixels, pas plus)
  - une taille apparente du damier (diagonale du quadrillage detecte)
    quasi identique entre gauche et droite

Si le damier a bouge (translate, tourne, rapproche/eloigne) entre les
deux prises, ces deux indicateurs sortent largement de la plage normale
PRECISEMENT sur les poses concernees -- ce qui permet de les identifier.

Ne necessite aucune nouvelle photo : travaille sur calib_images/ existant.
"""

import numpy as np

from calibrate_stereo import find_corners, CHESSBOARD_SIZE, IMAGE_DIR
import glob


def chirality_sign(corners, cols):
    """
    Signe du produit vectoriel (coin[0,1]-coin[0,0]) x (coin[1,0]-coin[0,0]).
    Indique le sens de rotation "premiere ligne -> premiere colonne" dans
    l'image. Doit etre LE MEME pour gauche et droite si les deux cameras
    voient le monde de la meme maniere (pas d'effet miroir). Un signe
    systematiquement oppose = une des deux images est inversee gauche-droite.
    """
    u = corners[1] - corners[0]
    v = corners[cols] - corners[0]
    cross = u[0] * v[1] - u[1] * v[0]
    return np.sign(cross)


def main():
    left_paths = sorted(glob.glob(f"{IMAGE_DIR}/left/*.png"))
    right_paths = sorted(glob.glob(f"{IMAGE_DIR}/right/*.png"))

    results_L, _ = find_corners(left_paths, CHESSBOARD_SIZE)
    results_R, _ = find_corners(right_paths, CHESSBOARD_SIZE)
    common_idx = sorted(set(results_L) & set(results_R))
    cols = CHESSBOARD_SIZE[0]

    print(f"\n{'Pose':>5} | {'Ecart Y moyen (px)':>20} | {'Ratio taille L/R':>18} | {'Chiralite L':>12} | {'Chiralite R':>12}")
    print("-" * 90)

    rows = []
    n_opposite = 0
    for idx in common_idx:
        corners_L = results_L[idx][1].reshape(-1, 2)
        corners_R = results_R[idx][1].reshape(-1, 2)

        y_diff = np.mean(np.abs(corners_L[:, 1] - corners_R[:, 1]))

        size_L = np.linalg.norm(corners_L[0] - corners_L[-1])
        size_R = np.linalg.norm(corners_R[0] - corners_R[-1])
        size_ratio = size_L / size_R

        chir_L = chirality_sign(corners_L, cols)
        chir_R = chirality_sign(corners_R, cols)
        if chir_L != chir_R:
            n_opposite += 1

        rows.append((idx, y_diff, size_ratio, chir_L, chir_R))

    # Trie par ecart Y decroissant -- les pires en premier
    rows.sort(key=lambda r: r[1], reverse=True)

    for idx, y_diff, size_ratio, chir_L, chir_R in rows:
        flag = ""
        if chir_L != chir_R:
            flag = "  <-- CHIRALITE OPPOSEE (effet miroir ?)"
        elif y_diff > 20 or abs(size_ratio - 1.0) > 0.1:
            flag = "  <-- suspect (damier probablement bouge)"
        print(f"{idx:>5} | {y_diff:>20.1f} | {size_ratio:>18.3f} | {chir_L:>12.0f} | {chir_R:>12.0f}{flag}")

    print(f"\n{n_opposite} / {len(rows)} poses ont une chiralite OPPOSEE entre gauche et droite.")
    if n_opposite > len(rows) * 0.7:
        print(">>> Quasi-systematique : une des deux cameras produit tres "
              "probablement une image en miroir par rapport a l'autre. "
              "C'est la cause la plus probable de l'echec de la calibration "
              "stereo, pas le mouvement du damier.")

    y_diffs = [r[1] for r in rows]
    print(f"\nEcart Y moyen sur l'ensemble : {np.mean(y_diffs):.1f} px "
          f"(mediane : {np.median(y_diffs):.1f} px)")
    print("Si quelques poses seulement sont 'suspectes' : retirez-les et "
          "relancez calibrate_stereo.py sur le reste.")
    print("Si la QUASI-TOTALITE des poses sont suspectes ou que l'ecart Y "
          "moyen est tres eleve partout : le probleme est probablement "
          "le protocole de capture (damier qui bouge systematiquement) "
          "plutot que quelques poses isolees -- il faudra refixer le "
          "damier sur un support rigide et tout reprendre.")


if __name__ == "__main__":
    main()
