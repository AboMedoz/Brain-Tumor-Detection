import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model

img_size = 128
data = []
img_names = []
true_labels = []
label_map = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}

total = 0
correct = 0
false_data = []

for category in label_map.keys():
    folder = os.path.join('Dataset', 'Testing', category)
    for img_name in os.listdir(folder):
        try:
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))
            img = img / 255.0
            img = img.reshape(1, img_size, img_size, 1)
            data.append(img)
            img_names.append(img_name)
            true_labels.append(category)
        except:
            pass

model = load_model('brain_tumor_detection.keras')

for i, image in enumerate(data):
    pred = model.predict(image)
    class_index = int(np.argmax(pred))
    predicted_class = [k for k, v in label_map.items() if v == class_index][0]
    actual_class = true_labels[i]

    is_correct = (predicted_class == actual_class)
    if is_correct == 0:
        false_data.append(img_names[i])
    correct += is_correct
    total += 1
    print(f"{img_names[i]:30s} | Actual: {actual_class:10s} | Predicted: {predicted_class:10s} | {'✅' if is_correct else '❌'}")

accuracy = (correct / total) * 100
print(f'Accuracy: {accuracy:.2f} in Total: {total}')
print(f'False Predictions: {false_data}')

