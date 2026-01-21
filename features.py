import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input # type: ignore
from tensorflow.keras.models import Model # type: ignore

# --------------------------------------------------
# FEATURE EXTRACTION FUNCTION
# --------------------------------------------------
def extract_features(image_dir, limit=None):
    """
    Extract CNN features using VGG16.

    Parameters:
    - image_dir : path to image folder
    - limit     : number of images to process

    Returns:
    - features  : dictionary {image_id: feature_vector}
    """

    # ✅ Load model ONLY when function is called
    base_model = VGG16(weights="imagenet")
    model = Model(
        inputs=base_model.inputs,
        outputs=base_model.layers[-2].output
    )

    features = {}
    images = os.listdir(image_dir)

    if limit is not None:
        images = images[:limit]

    for img in tqdm(images, desc="Extracting features"):
        try:
            path = os.path.join(image_dir, img)

            image = Image.open(path).convert("RGB")
            image = image.resize((224, 224))
            image = np.array(image)

            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)

            feature = model.predict(image, verbose=0)

            image_id = img.split('.')[0]
            features[image_id] = feature

        except Exception as e:
            print(f"[WARNING] Skipping image {img}: {e}")

    return features
