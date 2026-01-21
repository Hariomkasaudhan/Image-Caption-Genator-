import pickle
import numpy as np

def verify_cache():
    file_path = "all_features.pkl"
    
    try:
        print(f"[INFO] Opening {file_path}...")
        with open(file_path, "rb") as f:
            features = pickle.load(f)
        
        total_images = len(features)
        print(f"[SUCCESS] File loaded correctly.")
        print(f"[INFO] Total images in cache: {total_images}")
        
        # Check a random sample to verify shape
        first_key = list(features.keys())[0]
        sample_feature = np.array(features[first_key])
        
        print(f"[INFO] Sample Image ID: {first_key}")
        print(f"[INFO] Feature Shape: {sample_feature.shape}")
        
        # Flattened shape check
        if sample_feature.flatten().shape[0] == 4096:
            print("[SUCCESS] Feature dimensions are correct (4096).")
        else:
            print(f"[ERROR] Dimension mismatch! Expected 4096, got {sample_feature.flatten().shape[0]}")

    except FileNotFoundError:
        print("[ERROR] all_features.pkl not found. Run train_both.py first.")
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")

if __name__ == "__main__":
    verify_cache()