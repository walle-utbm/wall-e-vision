# Conversion YOLO vers NCNN pour Raspberry Pi

## Pourquoi NCNN?

NCNN est un framework d'inférence neuronal optimisé pour ARM (Raspberry Pi). Comparé au PyTorch:

## Installation des dépendances

```bash
# Mettre à jour ultralytics pour supporter NCNN
pip install -U ultralytics

pip install ncnn
pip install openvino-dev[caffe,onnx] 
```

## Conversion du modèle

### Étape 1: Convertir votre modèle .pt en NCNN

Depuis la racine du projet:

```bash
python convert_to_ncnn.py --model model/best.pt --output model/
```

**Résultat:**
```
model/best.ncnn.param   (configuration réseau - ~100KB)
model/best.ncnn.bin     (poids quantifiés - ~30MB)
```

## Utilisation automatique

Le code détecte automatiquement le modèle NCNN!

```python
# detector.py cherche automatiquement:
# 1. model/best.ncnn.param + model/best.ncnn.bin  (NCNN - plus rapide)
# 2. Fallback à model/best.pt (PyTorch - fallback)
```

Simplement lancer:
```bash
python src/main.py
```

Si NCNN est présent, vous verrez:
```
Loading NCNN model: model/best.ncnn.param
```

Si pas NCNN, fallback automatique:
```
Loading PyTorch model: model/best.pt
```

## Performance espérée

 **4x plus rapide qu'avec Pytorch**

