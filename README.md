# yolo
# 1. Install library YOLO dari ultralytics
!pip install --upgrade ultralytics opencv-python-headless matplotlib --quiet

# 2. Import library
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from google.colab import files

# 3. Unggah gambar
print("Silakan unggah gambar yang akan digunakan:")
uploaded = files.upload()

# Ambil nama file gambar yang diunggah
image_path = list(uploaded.keys())[0]

# 4. Load model YOLO
model = YOLO("yolov5s.pt")  # YOLOv5s pretrained model

# Kamus label ke bahasa Indonesia
label_dict = {
    'person': 'Orang',
    'bicycle': 'Sepeda',
    'car': 'Mobil',
    'motorbike': 'Motor',
    'airplane': 'Pesawat/Helicopter',
    'helicopter': 'Helicopter',
    'bus': 'Bus',
    'train': 'Kereta',
    'truck': 'Tank',
    'boat': 'Kapal',
    'dog': 'Anjing',
    'cat': 'Kucing',
    'horse': 'Kuda',
    'sheep': 'Domba',
    'cow': 'Sapi',
    'elephant': 'Gajah',
    'bear': 'Beruang',
    'zebra': 'Zebra',
    'giraffe': 'Jerapah'

}

# 5. Fungsi untuk deteksi objek
def detect_objects(image_path):
    # Baca gambar
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Deteksi objek
    results = model(image_path)
    
    # Annotate gambar dengan label dalam bahasa Indonesia
    for result in results[0].boxes:
        # Dapatkan label dalam bahasa Inggris
        label = model.names[int(result.cls)]
        # Ubah label ke bahasa Indonesia menggunakan kamus
        label_indonesia = label_dict.get(label, label)  # Jika tidak ditemukan, tetap gunakan label asli
        
        # Dapatkan koordinat dan confidence
        coordinates = result.xyxy.cpu().numpy()[0]  # Koordinat bounding box
        confidence = result.conf.item()  # Confidence score
        
        # Gambar bounding box dan label di atas gambar
        cv2.rectangle(img, (int(coordinates[0]), int(coordinates[1])), 
                      (int(coordinates[2]), int(coordinates[3])), (255, 0, 0), 2)  # Menggambar kotak
        cv2.putText(img, f"{label_indonesia} ({confidence:.2f})", 
                    (int(coordinates[0]), int(coordinates[1]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)  # Menambahkan teks label

    # Plot hasil deteksi
    annotated_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.imshow(annotated_img)
    plt.axis('off')
    plt.show()

    # Print hasil deteksi ke terminal
    print("Hasil Deteksi Objek:")
    for result in results[0].boxes:
        # Dapatkan label dalam bahasa Inggris
        label = model.names[int(result.cls)]
        # Ubah label ke bahasa Indonesia
        label_indonesia = label_dict.get(label, label)
        confidence = result.conf.item()  # Confidence score
        coordinates = result.xyxy.cpu().numpy()  # Koordinat bounding box
        print(f"Label: {label_indonesia}, Kepercayaan: {confidence:.2f}, Koordinat: {coordinates}")

# 6. Jalankan deteksi objek
detect_objects(image_path)


![Screenshot 2024-12-23 115027](https://github.com/user-attachments/assets/22ee86c9-0e82-4b6e-a0c5-354e3799f198)

