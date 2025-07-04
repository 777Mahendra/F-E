import cv2
from google.colab.patches import cv2_imshow

# Load cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

def detect_and_display(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, 1.1, 2, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(30, 30))

    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        frame = cv2.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 2)

        faceROI = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(faceROI, 1.1, 2, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(30, 30))

        for (ex, ey, ew, eh) in eyes:
            eye_center = (x + ex + ew // 2, y + ey + eh // 2)
            radius = int(round((ew + eh) * 0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 2)

    cv2_imshow(frame)  # <- Colab-specific image display

# Open uploaded video
cap = cv2.VideoCapture('uploaded_video.mp4')

frame_count = 0
max_frames = 30  # To prevent too many outputs in Colab

while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    detect_and_display(frame)
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
