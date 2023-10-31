import tensorflow as tf

# Listar as GPUs disponíveis
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        print("Nome da GPU:", gpu.name)
else:
    print("Nenhuma GPU disponível.")