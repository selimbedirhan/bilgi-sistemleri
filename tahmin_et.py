import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os
import random

# Modelin ve verinin yolu
MODEL_PATH = 'alzheimer_resnet50_model_final.h5'
TEST_DIR = 'Alzheimers_Detection_dataset/test/'

# Sınıf isimleri (eğitimdeki sırayla aynı olmalı)
CLASS_NAMES = ['MD', 'MoD', 'ND', 'VMD'] 
IMAGE_SIZE = 224

# 1. Modeli Yükle
try:
    model = load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' başarıyla yüklendi.")
except Exception as e:
    print(f"Hata: Model yüklenemedi. {e}")
    print(f"'{MODEL_PATH}' dosyasının bu script ile aynı klasörde olduğundan emin olun.")
    exit()

# 2. Rastgele bir test görüntüsü seç
try:
    random_image_name = random.choice(os.listdir(TEST_DIR))
    image_path = os.path.join(TEST_DIR, random_image_name)
    print(f"Rastgele test görüntüsü seçildi: {random_image_name}")
except FileNotFoundError:
    print(f"Hata: Test klasörü bulunamadı: {TEST_DIR}")
    print("Lütfen 'Alzheimers_Detection_dataset' klasörünün doğru yerde olduğundan emin olun.")
    exit()
    

# 3. Görüntüyü yükle ve modele hazırla
img = image.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
img_array = image.img_to_array(img)
img_array_expanded = np.expand_dims(img_array, axis=0) # Batch boyutu ekle (1, 224, 224, 3)
img_processed = preprocess_input(img_array_expanded)

# 4. Tahmin Yap
prediction = model.predict(img_processed)

# 5. Sonucu Yorumla
predicted_class_index = np.argmax(prediction)
predicted_class_name = CLASS_NAMES[predicted_class_index]
confidence_score = np.max(prediction) * 100

print("\n--- TAHMİN SONUCU ---")
print(f"Görüntü: {random_image_name}")
print(f"Modelin Tahmini: {predicted_class_name}")
print(f"Emin Olma Skoru: {confidence_score:.2f} %")

# İsteğe bağlı: Görüntüyü göstermek için
# import matplotlib.pyplot as plt
# plt.imshow(img)
# plt.title(f"Tahmin: {predicted_class_name}")
# plt.axis('off')
# plt.show()