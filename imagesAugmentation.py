import tensorflow as tf
import os
import uuid
import cv2
import numpy as np

# Método de augmentation
def data_aug(img):
    data = []
    for i in range(9):
        img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1,2))
        img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1,3))
        img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_saturation(img, lower=0.9,upper=1, seed=(np.random.randint(100),np.random.randint(100)))
            
        data.append(img)
    
    return data

# Função para aplicar augmentation em um diretório
def apply_augmentation_to_directory(directory_path):
    for name_file in os.listdir(directory_path):
        img_path = os.path.join(directory_path, name_file)
        img = cv2.imread(img_path)
        augmented_images = data_aug(img)

        for image in augmented_images:
            cv2.imwrite(os.path.join(directory_path, '{}.jpg'.format(uuid.uuid1())), image.numpy())

# Escolha do diretório para aplicar o augmentation
choice = input("Deseja aplicar o augmentation em ANC_PATH (a) ou POS_PATH (p)? Digite 'a' ou 'p': ")

if choice == 'a':
    ANC_PATH = '/home/carlos/Documentos/faceRecognition/FaceRecognition/data/anchor'
    apply_augmentation_to_directory(ANC_PATH)
elif choice == 'p':
    POS_PATH = '/home/carlos/Documentos/faceRecognition/FaceRecognition/data/positive'
    apply_augmentation_to_directory(POS_PATH)
else:
    print("Escolha inválida. Por favor, digite 'a' para ANC_PATH ou 'p' para POS_PATH.")
