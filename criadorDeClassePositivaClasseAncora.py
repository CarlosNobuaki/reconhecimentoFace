import uuid
import os
import cv2

ANC_PATH = '/home/carlos/Documentos/faceRecognition/FaceRecognition/data/anchor'
# Aplicando identificação única nos arquivos âncora
os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))


#Estabelecendo conexão com a câmera
ANC_PATH = '/home/carlos/Documentos/faceRecognition/FaceRecognition/data/anchor'
POS_PATH = '/home/carlos/Documentos/faceRecognition/FaceRecognition/data/positive'
# Abre a câmera e faz a cptura de vídeo usando a indexação (0)
# Se precisar encontrar o index da câmera, rodar o código: 'procuraCameraIndex.py'
cap = cv2.VideoCapture(0)
while cap.isOpened(): 
    ret, frame = cap.read()
   
    # Cortando o frame para 250x250px
    frame = frame[120:120+250,200:200+250, :]
    
    # Coletando imagens âncoras - Clicar a letra 'a'  
    if cv2.waitKey(1) & 0XFF == ord('a'):
        # Create the unique file path 
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        # Write out anchor image
        cv2.imwrite(imgname, frame)
    
    # Coletando as imagens positivas  - Clicar a letra 'p'
    if cv2.waitKey(1) & 0XFF == ord('p'):
        # Create the unique file path 
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        # Write out positive image
        cv2.imwrite(imgname, frame)
    
    # Show image back to screen
    cv2.imshow('Image Collection', frame)
    
    # Finalizar processo - Clicar a letra 'q'
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
        
# Release the webcam
cap.release()
# Close the image show frame
cv2.destroyAllWindows()