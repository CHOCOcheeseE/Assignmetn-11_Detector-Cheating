import cv2
import time
import os
from datetime import datetime

# Setup direktori dan file
os.makedirs("captured_images", exist_ok=True)

# Inisialisasi komponen
detector = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)

# Variabel tracking
stats = {'total': 0, 'multi': 0, 'missing': 0}
timers = {'last_alert': 0, 'absence_start': None}
ALERT_COOLDOWN = 3
ABSENCE_THRESHOLD = 3

print("=== MONITORING UJIAN ONLINE ===")
print("Tekan 'q' untuk keluar")

while True:
    ret, frame = camera.read()
    if not ret:
        break
    
    current_time = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    face_count = len(detections)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Handle multiple faces
    if face_count > 1 and (current_time - timers['last_alert'] > ALERT_COOLDOWN):
        alert_msg = f"[{timestamp}] ALERT: {face_count} wajah terdeteksi!"
        print(alert_msg)
        with open("cheating_log.txt", "a", encoding='utf-8') as f:
            f.write(alert_msg + "\n")
        cv2.imwrite(f"captured_images/multiple_{datetime.now().strftime('%H%M%S')}.jpg", frame)
        stats['total'] += 1
        stats['multi'] += 1
        timers['last_alert'] = current_time
    
    # Handle missing face
    if face_count == 0:
        if timers['absence_start'] is None:
            timers['absence_start'] = time.time()
        elif current_time - timers['absence_start'] > ABSENCE_THRESHOLD:
            duration = current_time - timers['absence_start']
            alert_msg = f"[{timestamp}] ALERT: Wajah hilang selama {duration:.1f} detik!"
            print(alert_msg)
            with open("cheating_log.txt", "a", encoding='utf-8') as f:
                f.write(alert_msg + "\n")
            cv2.imwrite(f"captured_images/absent_{datetime.now().strftime('%H%M%S')}.jpg", frame)
            stats['total'] += 1
            stats['missing'] += 1
            timers['absence_start'] = None
    else:
        timers['absence_start'] = None
    
    # Draw face boxes
    for (x, y, w, h) in detections:
        color = (0, 255, 0) if face_count == 1 else (0, 0, 255)
        status = "OK" if face_count == 1 else "ALERT"
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Info overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (380, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    info = [
        f"Waktu: {datetime.now().strftime('%H:%M:%S')}",
        f"Wajah: {face_count} | Pelanggaran: {stats['total']}"
    ]
    
    if timers['absence_start']:
        info.append(f"Hilang: {current_time - timers['absence_start']:.1f}s")
    
    for i, text in enumerate(info):
        cv2.putText(frame, text, (15, 30 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    cv2.imshow("Monitor Ujian Online", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

print(f"\n=== RINGKASAN ===")
print(f"Total Pelanggaran: {stats['total']}")
print(f"Multiple Faces: {stats['multi']}")
print(f"Missing Face: {stats['missing']}")