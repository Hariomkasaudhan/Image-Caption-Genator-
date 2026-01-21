import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input # type: ignore
from tensorflow.keras.models import Model # type: ignore
import pickle

# --------------------------------------------------
# LOAD TOKENIZER
# --------------------------------------------------
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------
model = load_model("image_caption_model_flickr_coco.h5")

# --------------------------------------------------
# LOAD VGG16 FOR FEATURE EXTRACTION
# --------------------------------------------------
base_model = VGG16(weights="imagenet")
cnn_model = Model(
    inputs=base_model.inputs,
    outputs=base_model.layers[-2].output
)

max_length = 51   # same as training time

# --------------------------------------------------
# IMAGE FEATURE EXTRACTION
# --------------------------------------------------
def extract_image_feature(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image)

    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    feature = cnn_model.predict(image, verbose=0)
    feature = feature.reshape(1, 4096)
    return feature

# --------------------------------------------------
# CAPTION GENERATION
# --------------------------------------------------
def generate_caption(photo_feature):
    caption = "startseq"

    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([caption])[0]
        seq = pad_sequences([seq], maxlen=max_length)

        yhat = model.predict([photo_feature, seq], verbose=0)
        yhat = np.argmax(yhat)

        word = tokenizer.index_word.get(yhat)
        if word is None:
            break

        caption += " " + word

        if word == "endseq":
            break

    return caption

# --------------------------------------------------
# TEST IMAGE
# --------------------------------------------------
image_path = "test.jpg"   # 👈 yahan apni image ka path do
photo = extract_image_feature(image_path)
result = generate_caption(photo)

print("\nGenerated Caption:")
print(result.replace("startseq", "").replace("endseq", ""))
