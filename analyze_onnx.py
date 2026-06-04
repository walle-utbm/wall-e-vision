import onnx

# Charger le modèle
model_path = "best.onnx"
model = onnx.load(model_path)

# Vérifier la validité du modèle
onnx.checker.check_model(model)
print("Le modèle ONNX est valide.\n")

# Afficher les informations générales
print(f"Version IR ONNX : {model.ir_version}")
print(f"Opset utilisé : {model.opset_import[0].version}\n")

# Analyser les entrées
print("=== ENTRÉES ===")
for input in model.graph.input:
    print(f"Nom : {input.name}")
    tensor_type = input.type.tensor_type
    shape = [dim.dim_value for dim in tensor_type.shape.dim]
    print(f"Shape : {shape}\n")

# Analyser les sorties
print("=== SORTIES ===")
for output in model.graph.output:
    print(f"Nom : {output.name}")
    tensor_type = output.type.tensor_type
    shape = [dim.dim_value for dim in tensor_type.shape.dim]
    print(f"Shape : {shape}\n")

# Lister les opérateurs utilisés (crucial pour la compatibilité QNN/SNPE)
print("=== OPÉRATEURS UTILISÉS ===")
ops = set([node.op_type for node in model.graph.node])
print(", ".join(ops))