# Cria as parastas necessárias para armazenar as imagens positivas, negativas e as âncoras

import os

# configurando os Paths
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# Criando os diretórios
os.makedirs(POS_PATH)
os.makedirs(NEG_PATH)
os.makedirs(ANC_PATH)

# base de dados de faces humanas para treinamento 
lfw_directory = '/home/carlos/Documentos/reconhecimentoFace/lfw/'
# Atribuindo a pasta 'negative' à variável negative_directory
negative_directory = '/home/carlos/Documentos/reconhecimentoFace/FaceRecognition/data/negative'
# Percorrendo os arquivos na pasta lfw e adicionando na pasta negative_directory
for directory in os.listdir(lfw_directory):
    directory_path = os.path.join(lfw_directory, directory)
    if os.path.isdir(directory_path):
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            new_path = os.path.join(negative_directory, file)
            os.replace(file_path, new_path)
            print(new_path) 



import subprocess

# Comando Docker que você deseja executar
docker_command = "docker exec -it FaceRecognition python3 criadorDePastas.py"

# Execute o comando Docker
try:
    subprocess.run(docker_command, shell=True, check=True)
    print("Comando Docker executado com sucesso.")
except subprocess.CalledProcessError as e:
    print(f"Erro ao executar o comando Docker: {e}")

            