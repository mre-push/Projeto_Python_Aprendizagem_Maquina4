import cv2                               			# OpenCV
import numpy as np                       			# NumPy

from keras.datasets import mnist         			# Importando o dataset usado no treino
from keras.models import Sequential      			# Modelo de rede neural
from keras.layers import Dense           			# Layer do tipo densamente conectado
from keras.utils import np_utils         			# Usaremos dela o metodo 'to_categorical()'

# Carregando o Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)              # reshape(linhas, colunas)
x_test = x_test.reshape(10000, 784)

x_train = x_train/255.0    
x_test = x_test/255.0	

  # Reshape training data _ RETIRADO DE CONSULTA mateus transformando lista em matriz
mnist_data = MNIST(os.path.abspath(path))
new_varnew_var = X, Y = mnist_data.load_training()
X_train = np.reshape(np.asarray(X, dtype=np.uint8), (60000, 28, 28, 1))
Y_train = np.reshape(np.asarray(Y, dtype=np.uint8), (60000,))
  # Normalize data for better results
X_train = X_train.astype('float32')/255
# Convert labels to categoricals (0 -> [1 0 0 0 0 0 0 0 0 0])
Y_train = keras.utils.to_categorical(Y_train, num_classes=10)
  # Reshape test data _ RETIRADO DE CONSULTA mateus 
X, Y = mnist_data.load_testing()
X_test = np.reshape(np.asarray(X, dtype=np.uint8), (10000, 28, 28, 1))
Y_test = np.reshape(np.asarray(Y, dtype=np.uint8), (10000,))
  # Normalize data for better results
X_test = X_test.astype('float32')/255
  # Convert labels to categoricals (5 -> [0 0 0 0 0 1 0 0 0 0])
Y_test = keras.utils.to_categorical(Y_test, num_classes=10)
def new_func(X_train, Y_train, X_test, Y_test):
 #return (X_train, Y_train), (X_test, Y_test)
    

 #return new_func(X_train, Y_train, X_test, Y_test)
 y_train = np_utils.to_categorical(y_train)  # Transformando lista em matriz VERIFICAR A NECESSIDADE DISSO
y_test =np_utils.to_categorical(y_test)

 # Criando o modelo do tipo Sequencial
model = Sequential() 

#CRIANDO LISTA DE PARAMETROS DE TESTE _ RETIRADO DE CONSULTA mateus
def get_test_models():
  # Create list of parameters to test
  models_params = [
    {
      'name': 'padrao-4-6-500',
      'parameters': {
        'first_layer': {'filters': 4, 'kernel_size': 5},
        'second_layer': {'filters': 6, 'kernel_size': 5},
        'mlp_neurons': 500
      }
    },
    {
      'name': 'mudar_filtros-6-4-500',
      'parameters': {
        'first_layer': {'filters': 6, 'kernel_size': 5},
        'second_layer': {'filters': 4, 'kernel_size': 5},
        'mlp_neurons': 500
      }
    },
    {
      'name': 'mudar_mlp-4-6-700',
      'parameters': {
        'first_layer': {'filters': 4, 'kernel_size': 5},
        'second_layer': {'filters': 6, 'kernel_size': 5},
        'mlp_neurons': 700
      }
    },
    {
      'name': 'mudar_mlp-4-6-300',
      'parameters': {
        'first_layer': {'filters': 4, 'kernel_size': 5},
        'second_layer': {'filters': 6, 'kernel_size': 5},
        'mlp_neurons': 300
      }
    },
    {
      'name': 'mudar_kernel7-4-6-500',
      'parameters': {
        'first_layer': {'filters': 4, 'kernel_size': 7},
        'second_layer': {'filters': 6, 'kernel_size': 7},
        'mlp_neurons': 500
      }
    },
    {
      'name': 'mudar_kernel3-4-6-500',
      'parameters': {
        'first_layer': {'filters': 4, 'kernel_size': 3},
        'second_layer': {'filters': 6, 'kernel_size': 3},
        'mlp_neurons': 500
      }
    },
  ]
  return models_params

#REMOVENDO MARKS _ RETIRADO DE CONSULTA mateus
def remove_marks(ax):
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax.get_xaxis().set_ticks([])
  ax.get_yaxis().set_ticks([])
  ax.get_xaxis().set_ticklabels([])
  ax.get_yaxis().set_ticklabels([])
# MODELOS de TREINO _ RETIRADO DE CONSULTA mateus
def train_models(models_params, X_train, Y_train, X_test, Y_test):
  data = {}
  for model_params in models_params:
    name = model_params['name']
    params = model_params['parameters']

    # Create the Model for the mnist
    model = MnistModel(name)
    model.generate_model(**params)
    # Train the model and save the model
    model.train(X_train, Y_train, epochs=20)
    model.save_model()
    print("Trained model {}.".format(name))

    # Load metrics
    history = model.history.copy()
    metrics = model.calculate_metrics(X_test, Y_test)
    metrics_names = model.get_metrics_names()

    # Extend metrics to all epochs
    for metric, values in history.items():
      for i in range(len(values), 20):
        values.append(values[len(values)-1])

    # Save the metrics
    data[name] = (history, dict(zip(metrics_names, metrics)))
  return data
#CARREGANDO MODELOS PARAMETROS _ RETIRADO DE CONSULTA mateus
def load_models(models_params, X_test, Y_test):
  data = {}
  for model_params in models_params:
    name = model_params['name']
    params = model_params['parameters']

    # Load the Model for the mnist
    model = MnistModel(name)
    model.load_model()
    print("Loaded model {}.".format(name))

    # Load metrics
    history = model.history.copy()
    metrics = model.calculate_metrics(X_test, Y_test)
    metrics_names = model.get_metrics_names()

    # Extend metrics to all epochs
    for metric, values in history.items():
      for i in range(len(values), 20):
        values.append(values[len(values)-1])

    # Save the metrics
    data[name] = (history, dict(zip(metrics_names, metrics)))
  return data




# Camada Oculta
model.add(Dense(256, input_dim=784, activation='relu'))       # Adicionando a camada densa. Dense(qtde_de_neurônios, input_dim = qtde_de_entradas, Activation='tipo_de_ativação')
# 784 = qtde_pixel
# 256 = valor arbitrário, altere por valores de potência de 2. Ex: 2, 4, 8, 16, 32, 64...

# Camada de Saída
model.add(Dense(10, activation='softmax'))                    # Adicionando a camada de saída Dense(qtde_de_neuronios, activation='tipo_de_ativação')
# 10 = qtde_classes

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])

model.fit(x_train, y_train, epochs=5)

evaluate = model.evaluate(x_test, y_test)

print('\nloss:{:3.2f}, accuracy:{:2.2f}'.format(evaluate[0], evaluate[1]))

##### Usando na predição uma imagem do computador, DESCOMENTE SE FOR USÁ-LA
# OBS: a imagem deve ter o fundo preto com o numero em branco!
# img = cv2.imread('numero.png')
# img = cv2.resize(img, (28, 28))
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = img.reshape(1, 28*28)
# img_do_pc = img/255.0

##### Usando uma imagem do dataset de testes 
img_do_dataset = x_test[0].reshape(1, 28*28)

##### Prevendo o valor
# Use img_do_pc ou img_do_dataset
resultado = model.predict(img_do_dataset)

print('Valor previsto: ',resultado.argmax())
print('Precisão: {:4.2f}%'.format(resultado.max()* 100))
