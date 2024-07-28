import torch
import cv2
import os
import numpy as np

# YOLOv5 modelini yükle (önceden eğitilmiş model)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Video dosyalarının yollarını belirleyin
video_root = 'C:/projeler/footages'
video_paths = [
    f'{video_root}/4586141-hd_1920_1080_30fps.mp4',  # İlk video dosyasını fotoğraf ile değiştirin
    f'{video_root}/testvideo.mp4',
    f'{video_root}/4644521-uhd_2562_1440_30fps.mp4',
    f'{video_root}/4644508-uhd_2562_1440_30fps.mp4',
    f'{video_root}/vecteezy_abstract-blurry-busy-traffic-with-chaotic-vehicles_35196149.mp4',
    # Daha fazla video yolu ekleyin
]

# Önceki karedeki nesnelerin pozisyonlarını saklamak için bir sözlük
previous_positions = {}

# FPS (kare hızı) bilgisini saklamak için bir değişken
fps = 30

# Her bir video veya fotoğraf dosyasını işleyin
for video_path in video_paths:
    if video_path.endswith(('.png', '.jpg', '.jpeg')):
        # Eğer dosya bir fotoğraf ise
        img = cv2.imread(video_path)

        # YOLOv5 modelini kullanarak nesne algılama
        results = model(img)
        detections = results.xyxy[0].numpy()  # Tespit edilen nesnelerin koordinatları

        for *box, conf, cls in detections:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{model.names[int(cls)]}: {conf:.2f}'
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Sonuçları göster
        cv2.imshow('ADAS - Coyote', img)

        # İşlenmiş fotoğrafı kaydet
        output_path = os.path.join('C:/Users/kids/source/repos/Coyote/output/', os.path.basename(video_path))
        cv2.imwrite(output_path, img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        # Eğer dosya bir video ise
        cap = cv2.VideoCapture(video_path)

        # FPS (kare hızı) değerini al
        fps = cap.get(cv2.CAP_PROP_FPS)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # YOLOv5 modelini kullanarak nesne algılama
            results = model(frame)
            detections = results.xyxy[0].numpy()  # Tespit edilen nesnelerin koordinatları

            for *box, conf, cls in detections:
                x1, y1, x2, y2 = map(int, box)
                object_id = f'{x1}{y1}{x2}{y2}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{model.names[int(cls)]}: {conf:.2f}'

                # Hız tahmini
                if object_id in previous_positions:
                    prev_x1, prev_y1, prev_x2, prev_y2 = previous_positions[object_id]
                    dx = x1 - prev_x1
                    dy = y1 - prev_y1
                    distance = np.sqrt(dx**2 + dy**2)
                    speed = distance * fps * 0.036  # Hızı km/h cinsinden tahmin et
                    label += f' Speed: {speed:.2f} km/h'
                
                previous_positions[object_id] = (x1, y1, x2, y2)

                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('ADAS - Coyote', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
