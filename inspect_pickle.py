import pickle
import numpy as np

path = "Weight files/adaboost_model_with_smote_on_original_data.pkl"

with open(path, "rb") as f:
    obj = pickle.load(f)

print("Loaded object type:", type(obj))

# If inside it's a numpy array, show what's inside
if isinstance(obj, np.ndarray):
    print("ndarray dtype:", obj.dtype)
    print("ndarray shape:", obj.shape)
    print("\nInspecting first 20 elements:")
    for i, el in enumerate(obj.flat):
        if i >= 20:
            break
        print(f"[{i}] type =", type(el))
        try:
            print("   repr =", repr(el)[:200])
        except:
            print("   repr could not be printed")
else:
    print("\nObject content:")
    print(repr(obj))
