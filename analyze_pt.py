from ultralytics import YOLO
import sys

def analyze_yolo_pt(model_path):
    print(f"=== Analyse du modèle PyTorch : {model_path} ===")
    try:
        # Chargement du modèle
        model = YOLO(model_path)

        # 1. Structure globale (GFLOPs, Paramètres, Couches)
        print("\n--- 1. Architecture & Paramètres ---")
        model.info()

        # 2. Configuration d'entrée
        print("\n--- 2. Configuration attendue ---")
        # Récupération de la taille d'image attendue si elle est sauvegardée
        if hasattr(model, 'model') and hasattr(model.model, 'args'):
            args = model.model.args
            imgsz = getattr(args, 'imgsz', 'Non définie')
            print(f"Taille d'image optimale (imgsz) : {imgsz}")
        else:
            print("Taille d'image optimale : Non trouvée dans les métadonnées.")

        # 3. Dictionnaire des classes
        print("\n--- 3. Classes détectées ---")
        classes = model.names
        print(f"Nombre total de classes : {len(classes)}")
        for class_id, class_name in classes.items():
            print(f"  [{class_id}] -> {class_name}")

    except Exception as e:
        print(f"Erreur lors de l'analyse : {e}")

if __name__ == "__main__":
    # Par défaut, analyse 'best.pt' ou prend le fichier passé en argument
    fichier = sys.argv[1] if len(sys.argv) > 1 else "model/best.pt"
    analyze_yolo_pt(fichier)