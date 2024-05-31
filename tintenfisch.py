import easyocr
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])
picture= "test"
ending= ".png"
# Load the image
image_path = f'pages/{picture}{ending}'
image = cv2.imread(image_path)
new_directory = f"training/{picture}"
os.makedirs(new_directory, exist_ok=True)

# Perform OCR on the image
results = reader.readtext(image_path)
samples = []
outputdata = []

print(len(results))


# Display the image and draw bounding boxes around recognized text
for i, (bbox, text, prob) in enumerate(results):
    print(i, text, bbox)
     # Extrahiere die Koordinaten der Bounding Box
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox

    # Konvertiere die Koordinaten in Ganzzahlen
    x1, y1, x2, y2, x3, y3, x4, y4 = map(int, [x1, y1, x2, y2, x3, y3, x4, y4])

    # Zeichne die Bounding Box auf das Bild
    cv2.rectangle(image, (x1, y1), (x3, y3), (0, 255, 0), 2)

    # Füge den erkannten Text zur Bounding Box hinzu
    cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Extrahiere den Bildausschnitt (ROI) basierend auf den Bounding Box-Koordinaten
    sample = image[y1:y3, x1:x3]




    try:
         # Füge den Bildausschnitt zur Liste hinzu
        samples.append(sample)
        cv2.imwrite(f"training/{picture}/sample{picture}_{i}.jpg", sample)
        outputdata.append({"image_path": f"training/{picture}/sample_{i}.jpg", "text": f"{text}"})
    
    except:
        pass# Weitere Daten...
   


# Speichere die Bilder mit den Bounding Boxen und erkannten Texten
#for i, sample in enumerate(samples):
  #  cv2.imwrite(f"sample_{i}.jpg", sample)

print(outputdata)

# Dateipfad für die Ausgabedatei
output_file = f"training/{picture}/output{picture}.txt"

# Öffne die Ausgabedatei im Schreibmodus
with open(output_file, "w") as file:
    # Schreibe jedes Element der Liste in eine Zeile der Datei
    for item in outputdata:
        file.write(f"{item}\n")


# Show the image with annotations
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Print the recognized text
for (bbox, text, prob) in results:
    print(f"Detected text: {text} (Confidence: {prob:.2f})")
