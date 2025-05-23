import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 5
dataset_size = 100

# ابحث عن الفهارس المتاحة للكاميرا
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Using camera index {i}")
        break
    cap.release()

if not cap.isOpened():
    print("Error: Could not open any camera.")
    exit()

def reset_camera():
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Space" to start! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord(' '):  # استخدام زر المسطرة (Space bar)
            break

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    reset_camera()

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        cv2.putText(frame, 'Collecting images... ({}/{})'.format(counter + 1, dataset_size), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1

    print('Finished collecting data for class {}'.format(j))

cap.release()
cv2.destroyAllWindows()
