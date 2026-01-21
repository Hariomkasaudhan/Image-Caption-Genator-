import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from features import extract_features
from preprocess import (
    load_captions,
    load_coco_captions,
    clean_caption,
    create_tokenizer
)
from model import define_model

# --------------------------------------------------
# 1. LOAD DATASETS & FEATURES (WITH CACHING)
# --------------------------------------------------
FEATURE_PICKLE = "all_features.pkl"

if os.path.exists(FEATURE_PICKLE):
    print("[INFO] Loading cached features from disk...")
    with open(FEATURE_PICKLE, "rb") as f:
        features = pickle.load(f)
    # Still need captions to build tokenizer
    flickr_captions = load_captions("dataset/captions.txt")
    coco_captions = load_coco_captions("dataset/coco2017/annotations/captions_train2017.json")
    captions = {**flickr_captions, **coco_captions}
else:
    print("[INFO] No cache found. Starting extraction (this will take time)...")
    flickr_captions = load_captions("dataset/captions.txt")
    flickr_features = extract_features("dataset/Images", limit=None)

    coco_captions = load_coco_captions("dataset/coco2017/annotations/captions_train2017.json")
    coco_features = extract_features("dataset/coco2017/train2017", limit=12000)

    captions = {**flickr_captions, **coco_captions}
    features = {**flickr_features, **coco_features}

    print("[INFO] Saving features to cache...")
    with open(FEATURE_PICKLE, "wb") as f:
        pickle.dump(features, f)

# --------------------------------------------------
# 2. TOKENIZER & PARAMETERS
# --------------------------------------------------
print("[INFO] Creating tokenizer...")
tokenizer = create_tokenizer(captions)
vocab_size = len(tokenizer.word_index) + 1

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Find max length based on clean captions
max_length = 51 # Based on your previous output
print(f"[INFO] Vocabulary Size: {vocab_size}")
print(f"[INFO] Max Caption Length: {max_length}")

# --------------------------------------------------
# 3. DATA GENERATOR (Standard Python)
# --------------------------------------------------
def data_generator_func():
    while True:
        for key, cap_list in captions.items():
            if key not in features:
                continue
            
            # Extract feature and ensure it is flat
            feat = np.array(features[key]).flatten()
            
            for cap in cap_list:
                cleaned = clean_caption(cap)
                seq = tokenizer.texts_to_sequences([cleaned])[0]
                
                for i in range(1, len(seq)):
                    in_seq = seq[:i]
                    out_seq = seq[i]
                    
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    # Yield as a nested tuple that matches the signature
                    yield (feat, in_seq), out_seq

# --------------------------------------------------
# 4. TF DATASET WRAPPER (Fixes output_signature error)
# --------------------------------------------------
output_signature = (
    (
        tf.TensorSpec(shape=(4096,), dtype=tf.float32), 
        tf.TensorSpec(shape=(max_length,), dtype=tf.int32)
    ),
    tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32)
)

train_dataset = tf.data.Dataset.from_generator(
    data_generator_func,
    output_signature=output_signature
)

# Batch the dataset
batch_size = 32
train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

# --------------------------------------------------
# 5. TRAINING
# --------------------------------------------------
print("[INFO] Building model...")
model = define_model(vocab_size, max_length)

# Calculate steps based on total captions
steps = (len(captions) * 5) // batch_size # Approx 5 captions per image

print("[INFO] Starting training...")
model.fit(
    train_dataset,
    steps_per_epoch=steps,
    epochs=20,
    verbose=1
)

model.save("image_caption_model.keras")
print("[INFO] Model saved successfully!")