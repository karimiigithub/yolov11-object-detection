import cv2
from ultralytics import YOLO
from window_capture import WindowCapture  # Senin kodun window_capture.py dosyasında olmalı

# Eğitimli model dosyanı buraya yaz
model = YOLO("yolov11_custom.pt")

# Pencere adını bulmak için önce şu kodu çalıştır:
# WindowCapture.list_window_names(WindowCapture)  → sonra buraya pencere adını yaz
wincap = WindowCapture("1-105 M2-Hero:Maceranın Sınırlarını Zorla! - WWARRIOR")  # Örnek: "Euro Truck Simulator 2"

while True:
    # Oyun penceresinden ekran görüntüsü al
    frame = wincap.get_screenshot()

    # YOLO modeliyle tespit yap
    results = model(frame)

    # Tahmin sonuçlarını çiz
    annotated_frame = results[0].plot()

    # Görüntüyü göster
    cv2.imshow("YOLO Detection", annotated_frame)

    # ESC tuşuyla çık
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
