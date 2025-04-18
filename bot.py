import cv2
from ultralytics import YOLO
from window_capture import WindowCapture 


model = YOLO("yolov11_custom.pt")


wincap = WindowCapture("1-105 M2-Hero:Maceranın Sınırlarını Zorla! - WWARRIOR")  

while True:
    
    frame = wincap.get_screenshot()

   
    results = model(frame)

   
    annotated_frame = results[0].plot()

   
    cv2.imshow("YOLO Detection", annotated_frame)

   
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
