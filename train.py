import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from keras.utils import to_categorical

from features import extract_features
from preprocess import (
    load_captions,
    load_coco_captions,
    create_tokenizer
)
from model import define_model

# ---------- FLICKR ----------
flickr_images = "dataset/Images"
flickr_captions_file = "dataset/captions.txt"

flickr_captions = load_captions(flickr_captions_file)
flickr_features = extract_features(flickr_images)

# ---------- COCO ----------
coco_images = "dataset/coco2017/train2017"
coco_captions_file = "dataset/coco2017/annotations/captions_train2017.json"

coco_captions = load_coco_captions(coco_captions_file)
coco_features = extract_features(coco_images, limit=5000)

# ---------- MERGE ----------
captions = {**flickr_captions, **coco_captions}
features = {**flickr_features, **coco_features}

tokenizer = create_tokenizer(captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(c.split())
                 for caps in captions.values() for c in caps)

def data_generator(captions, features):
    while True:
        for key, caps in captions.items():
            if key not in features:
                continue
            feature = features[key][0]
            for cap in caps:
                seq = tokenizer.texts_to_sequences([cap])[0]
                for i in range(1, len(seq)):
                    in_seq = pad_sequences([seq[:i]], maxlen=max_length)[0]
                    out_seq = to_categorical(seq[i], vocab_size)
                    yield ([feature, in_seq], out_seq)

model = define_model(vocab_size, max_length)

model.fit(
    data_generator(captions, features),
    steps_per_epoch=1500,
    epochs=5
)

model.save("image_caption_model_flickr_coco.h5")
