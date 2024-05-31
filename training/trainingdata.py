from easyocr import Reader

input_file = "IMG_4181/outputIMG_4181.txt"

data = []
with open(input_file, "r") as file:
    data = file.readlines()

# Entferne Newline-Zeichen am Ende jeder Zeile (optional)
data = [line.strip() for line in data]

# Ausgabe der Liste
print(data)
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