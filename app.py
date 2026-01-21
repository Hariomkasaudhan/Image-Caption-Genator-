import os
import numpy as np
import pickle
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input # type: ignore
from tensorflow.keras.models import Model # type: ignore
from PIL import Image
from werkzeug.utils import secure_filename

# --------------------------------------------------
# FLASK APP SETUP
# --------------------------------------------------
app = Flask(__name__)
BASE_DIR = app.root_path
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --------------------------------------------------
# LOAD MODEL & TOKENIZER
# --------------------------------------------------
# Ensure the filename matches your saved model from train_both.py
model = load_model("image_caption_model.keras", compile=False)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Set this to the value printed during training (e.g., 51 or 35)
max_length = 51 

# --------------------------------------------------
# VGG16 FEATURE EXTRACTION
# --------------------------------------------------
base_model = VGG16(weights="imagenet")
cnn_model = Model(
    inputs=base_model.inputs,
    outputs=base_model.layers[-2].output
)

def extract_image_feature(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # Extract features and reshape to (1, 4096)
    feature = cnn_model.predict(image, verbose=0)
    return feature.reshape(1, 4096)

# --------------------------------------------------
# IMPROVED BEAM SEARCH CAPTION GENERATION
# --------------------------------------------------
def generate_caption(photo, beam_width=5):
    start = tokenizer.word_index.get("startseq")
    end = tokenizer.word_index.get("endseq")
    
    # Each element: [list_of_indices, total_log_probability]
    sequences = [[ [start], 0.0 ]]
    
    for _ in range(max_length):
        all_candidates = []
        
        for seq, score in sequences:
            if seq[-1] == end:
                all_candidates.append([seq, score])
                continue
            
            padded = pad_sequences([seq], maxlen=max_length)
            preds = model.predict([photo, padded], verbose=0)[0]
            
            # Get top k probabilities
            top_k = np.argsort(preds)[-beam_width:]
            
            for idx in top_k:
                # Use negative log probability for scoring
                candidate_score = score - np.log(preds[idx] + 1e-9)
                
                # Penalty for immediate word repetition
                if len(seq) > 1 and idx == seq[-1]:
                    candidate_score += 1.5 
                
                candidate = seq + [idx]
                all_candidates.append([candidate, candidate_score])
        
        # Sort by lowest score (highest probability)
        sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]
        
        # Stop if all candidates hit 'endseq'
        if all(s[0][-1] == end for s in sequences):
            break

    final_seq = sequences[0][0]
    
    # Map indices to words
    words = []
    for idx in final_seq:
        word = tokenizer.index_word.get(idx)
        if word is None or word == "startseq" or word == "endseq":
            continue
        words.append(word)
        
    return " ".join(words)

# --------------------------------------------------
# FLASK ROUTES
# --------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename:
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            # URL for HTML display
            image_path = "static/uploads/" + filename

            # Generate Caption
            photo = extract_image_feature(save_path)
            caption = generate_caption(photo)

    return render_template("index.html", caption=caption, image=image_path)

# --------------------------------------------------
# RUN APP
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)