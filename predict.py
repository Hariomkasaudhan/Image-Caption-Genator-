import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from keras.utils import to_categorical

from features import extract_features
from preprocess import load_captions, clean_caption, create_tokenizer

model = load_model("image_caption_model.h5")

captions = load_captions("dataset/captions.txt")
tokenizer = create_tokenizer(captions)

max_length = max(len(clean_caption(c).split())
                 for caps in captions.values() for c in caps)

def generate_caption(photo):
    text = 'startseq'
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([text])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = model.predict([photo, seq], verbose=0)
        word = tokenizer.index_word[np.argmax(yhat)]
        text += ' ' + word
        if word == 'endseq':
            break
    return text

features = extract_features("dataset/Images")
image_id = list(features.keys())[0]
caption = generate_caption(features[image_id])
print(caption)
