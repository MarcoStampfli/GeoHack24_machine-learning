from easyocr import Reader

train_data = [
    {"image_path": "image1.jpg", "text": "Dies ist ein Beispieltext."},
    {"image_path": "image2.jpg", "text": "Ein anderer Beispieltext hier."},
    {"image_path": "image3.jpg", "text": "Noch ein Beispieltext."},
    # Weitere Daten...
]

# Initialisiere den EasyOCR-Reader
reader = Reader(['en'])

# Trainiere das Modell mit den Trainingsdaten
reader.train(train_data, validation_data)

# Bewerte die Leistung des Modells auf dem Testset
results = reader.test(test_data)
print(results)