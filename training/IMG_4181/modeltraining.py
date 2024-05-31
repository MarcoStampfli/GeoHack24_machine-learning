import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

### Daten laden und vorverarbeiten
# CSV-Datei laden
#data = pd.read_csv('outputIMG_4181.csv', delimiter=",")

# # Beispiel: Annahme, dass die CSV-Datei zwei Spalten hat: 'image_path' und 'text'
# image_paths = data['image_path'].values
# texts = data['text'].values

# # Daten in Trainings- und Testdaten aufteilen
# image_paths_train, image_paths_test, texts_train, texts_test = train_test_split(image_paths, texts, test_size=0.2, train_size=0.2 , random_state=42)


# ### Datengeneratoren erstellen


from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

# load image from the IAM database
url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
image = Image.open(requests.get(url=url, stream=True).raw).convert("RGB")

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(generated_ids, generated_text)
