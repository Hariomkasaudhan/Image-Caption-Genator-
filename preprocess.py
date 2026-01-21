import string
import json
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore

# --------------------------------------------------
# FLICKR8k CAPTION LOADER (captions.txt)
# --------------------------------------------------
def load_captions(file):
    """
    Loads Flickr-style captions from captions.txt
    Format:
    image_name.jpg,caption text
    """
    captions = {}
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            image_id, caption = line.strip().split(',', 1)
            image_id = image_id.split('.')[0]
            captions.setdefault(image_id, []).append(caption)
    return captions


# --------------------------------------------------
# COCO 2017 CAPTION LOADER (JSON)
# --------------------------------------------------
def load_coco_captions(json_file):
    """
    Loads COCO 2017 captions from captions_train2017.json
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    captions = {}
    for ann in data['annotations']:
        image_id = str(ann['image_id']).zfill(12)
        caption = ann['caption']
        captions.setdefault(image_id, []).append(caption)

    return captions


# --------------------------------------------------
# CAPTION CLEANING
# --------------------------------------------------
def clean_caption(caption):
    caption = caption.lower()
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    caption = caption.replace('\n', '').replace('\r', '')
    caption = caption.strip()
    return "startseq " + caption + " endseq"


# --------------------------------------------------
# TOKENIZER CREATION (WORKS FOR BOTH DATASETS)
# --------------------------------------------------
def create_tokenizer(captions):
    """
    Creates and fits tokenizer on all captions
    """
    all_caps = []
    for caps in captions.values():
        for cap in caps:
            all_caps.append(clean_caption(cap))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_caps)
    return tokenizer
