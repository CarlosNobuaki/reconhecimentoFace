import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ANC_PATH = '/home/carlos/Documentos/faceRecognition/FaceRecognition/data/anchor'
POS_PATH = '/home/carlos/Documentos/faceRecognition/FaceRecognition/data/positive'
NEG_PATH = '/home/carlos/Documentos/faceRecognition/FaceRecognition/data/negative'

# Use barras invertidas simples para formar o caminho
anchor = tf.data.Dataset.list_files(ANC_PATH + '/*.jpg').take(3000)
positive = tf.data.Dataset.list_files(POS_PATH + '/*.jpg').take(3000)
negative = tf.data.Dataset.list_files(NEG_PATH + '/*.jpg').take(3000)
#Pré-processamento
# Redimencionando tamanho das imagens para treinamento. O modelo de rede neural aceita imagens até 150x150. Nesse caso estamos redimensinando para 100x100
def preprocess(file_path):
    
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100,100))
    # Scale image to be between 0 and 1 
    img = img / 255.0

    # Return image
    return img
#Pré-processamento
# Criando datasets rotulados:  positive = 1 e negative = 0. Se a imagem for reconhecida como negativa,ou seja, não é uma face reconhecida, a resposta será 0.
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)


samples = data.as_numpy_iterator()
exampple = samples.next()
print (exampple)



#Pré-processamento
# Particionando dados para treino e teste
def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)
res = preprocess_twin(*exampple)
plt.imshow(res[1])
#Captura uma imagem e mostra o label 
print(res[2])

# Dataloader Iterator - Para carregar as imagens e conferir se a classificação esta correta
# Build dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=10000)

samples = data.as_numpy_iterator()
len(samples.next())
sam = samples.next()
plt.imshow(sam[1])
#Captura uma imagem e mostra o label 
print(sam[2])

# Separando partição de treinamento
# Treinamnento esta em 70% dos dados disponibilizados
train_data = data.take(round(len(data)*.7))
# Batch esta em 16
train_data = train_data.batch(16)
# Carrega 8 quadros em memória antes de rodar o batch novamente
train_data = train_data.prefetch(8)

# Separando partição de teste
# O teste esta em 30% dos dados disponibilizados
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
# Batch esta em 16
test_data = test_data.batch(16)
# Carrega 8 quadros em memória antes de rodar o batch novamente
test_data = test_data.prefetch(8)

#Construindo camada completa 
def make_embedding(): 
    inp = Input(shape=(100,100,3), name='input_image')    
    # First block
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)    
    # Second block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)    
    # Third block 
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)    
    # Final embedding block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)    
    return Model(inputs=[inp], outputs=[d1], name='embedding')

#Chamando o metodo
embedding = make_embedding()
embedding.summary()

# Construindo camada de distância
# Classe de distância siamesa L1
class L1Dist(Layer):
    
    # Iniciando o método - herança
    def __init__(self, **kwargs):
        super().__init__()
       
    # Calculando similaridades
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

l1 = L1Dist()

# Montando método de modelo siamês
def make_siamese_model(): 
    
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100,100,3))
    
    # Validation image in the network 
    validation_image = Input(name='validation_img', shape=(100,100,3))
    
    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

siamese_model = make_siamese_model()
siamese_model.summary()

print("O código foi pausado para  verificação dos summary. Digite '1' para continuar:")
user_input = input()

if user_input == '1':
    print("Continuando o código...")

    #Configuração de perda(loss) e optimizador
    binary_cross_loss = tf.losses.BinaryCrossentropy()
    opt = tf.keras.optimizers.Adam(1e-4) # 0.0001
    #Estabelecendo Checkpoints de treinamento
    checkpoint_dir = '/home/carlos/Documentos/faceRecognition/FaceRecognition/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

    #Construindo a função de treinamento
    @tf.function
    def train_step(batch):
        
        # Record all of our operations 
        with tf.GradientTape() as tape:     
            # Get anchor and positive/negative image
            X = batch[:2]
            # Get label
            y = batch[2]
            
            # Forward pass
            yhat = siamese_model(X, training=True)
            # Calculate loss
            loss = binary_cross_loss(y, yhat)
        print(loss)
            
        # Calculate gradients
        grad = tape.gradient(loss, siamese_model.trainable_variables)
        
        # Calculate updated weights and apply to siamese model
        opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
            
        # Return loss
        return loss
    

    def train(data, EPOCHS):
        # Loop through epochs
        for epoch in range(1, EPOCHS+1):
            print('\n Epoch {}/{}'.format(epoch, EPOCHS))
            progbar = tf.keras.utils.Progbar(len(data))
            
            # Creating a metric object 
            r = Recall()
            p = Precision()
            
            # Loop through each batch
            for idx, batch in enumerate(data):
                # Run train step here
                loss = train_step(batch)
                yhat = siamese_model.predict(batch[:2])
                r.update_state(batch[2], yhat)
                p.update_state(batch[2], yhat) 
                progbar.update(idx+1)
            print(loss.numpy(), r.result().numpy(), p.result().numpy())
            
            # Save checkpoints
            if epoch % 10 == 0: 
                checkpoint.save(file_prefix=checkpoint_prefix)

    # Treinando o modelo
    print("Digite a quantidade de épocas para treinamento: ")
    epoch_input = input()

    try:
        # Converta a entrada do usuário em um número inteiro
        epochs = int(epoch_input)
        
        # Verifique se o número de épocas é válido (maior que zero)
        if epochs > 0:
            train(train_data, epochs)
        else:
            print("O número de épocas deve ser maior que zero.")
    except ValueError:
        print("Entrada inválida. Por favor, insira um número inteiro válido para as épocas.")

else:
    print("Comando inválido. O código será encerrado.")


