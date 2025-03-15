import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("store_video2_high.mp4")
heatmap = solutions.Heatmap(colormap=cv2.COLORMAP_PARULA, show=True, model="yolo11n.pt")

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    results = heatmap(im0)
cap.release()
cv2.destroyAllWindows()