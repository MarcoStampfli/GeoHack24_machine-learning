import easyocr
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Load the image
image_path = 'pages/IMG_4181.jpg'
image = cv2.imread(image_path)

# Perform OCR on the image
results = reader.readtext(image_path)

# Display the image and draw bounding boxes around recognized text
for (bbox, text, prob) in results:
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))
    
    # Draw the bounding box
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    # Display the detected text
    cv2.putText(image, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Show the image with annotations
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Print the recognized text
for (bbox, text, prob) in results:
    print(f"Detected text: {text} (Confidence: {prob:.2f})")
