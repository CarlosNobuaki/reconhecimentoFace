import cv2
# Procura a câmera pela sua indexação, no caso o número index da câmera é 0
def find_camera_index(camera_number=10):
    for index in range(camera_number):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Camera encontrada no index: {index}")
            cap.release()

find_camera_index()
