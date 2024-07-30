import cv2
import numpy as np
import torch

def lane_detection(frame):
    """Gelişmiş şerit tespiti yap."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    height, width = frame.shape[:2]
    roi_vertices = np.array([[
        (0, height),
        (width, height),
        (width * 0.9, height * 0.6),
        (width * 0.1, height * 0.6)
    ]], np.int32)
    
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 20, minLineLength=20, maxLineGap=300)
    
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    
        if left_fit and right_fit:
            left_fit_average = np.average(left_fit, axis=0)
            right_fit_average = np.average(right_fit, axis=0)
            
            y1 = height
            y2 = int(height * 0.6)
            left_x1 = int((y1 - left_fit_average[1]) / left_fit_average[0])
            left_x2 = int((y2 - left_fit_average[1]) / left_fit_average[0])
            right_x1 = int((y1 - right_fit_average[1]) / right_fit_average[0])
            right_x2 = int((y2 - right_fit_average[1]) / right_fit_average[0])
            
            cv2.line(frame, (left_x1, y1), (left_x2, y2), (0, 255, 0), 2)
            cv2.line(frame, (right_x1, y1), (right_x2, y2), (0, 255, 0), 2)
    
    return frame

def collision_warning(frame, model):
    """Araç tespiti yap ve çarpışma uyarısı ver."""
    results = model(frame)
    detections = results.xyxy[0]

    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f'Confidence: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if conf > 0.5:
            cv2.putText(frame, "Dikkat! Çarpışma Riski", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame

def process_video(video_path):
    """Video dosyasını işleyin."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Video açılamadı: {video_path}")
        return

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video sonlandı.")
            break

        # Şerit tespiti
        frame = lane_detection(frame)
        # Araç tespiti
        frame = collision_warning(frame, model)

        # Sonucu göster
        cv2.imshow('ADAS', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = r'C:\projeler\footages\traffic.mp4'  # Kendi video dosyanızın yolunu yazın
    process_video(video_path)