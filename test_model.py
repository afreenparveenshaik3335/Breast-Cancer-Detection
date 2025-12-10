import pickle

model_path = "Weight files/adaboost_model_with_smote_on_original_data.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

print("Loaded object type:", type(model))
