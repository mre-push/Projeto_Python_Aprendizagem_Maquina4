import cv2                               			# OpenCV
import numpy as np                       			# NumPy

from keras.datasets import mnist         			# Importando o dataset usado no treino
from keras.models import Sequential      			# Modelo de rede neural
from keras.layers import Dense           			# Layer do tipo densamente conectado
from keras.utils import np_utils         			# Usaremos dela o metodo 'to_categorical()'

# Carregando o Dataset
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train = x_train.reshape(60000, 784)              # reshape(linhas, colunas)
#x_test = x_test.reshape(10000, 784)
#x_train = x_train/255.0    
#x_test = x_test/255.0	

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
  # return (X_train, Y_train), (X_test, Y_test)
    

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
#MODELOS de TREINO _ RETIRADO DE CONSULTA mateus
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

def present_metrics(data):
  # Crie as figuras e eixos para traçar os dados
  loss = plt.figure("Loss").gca()
  accuracy = plt.figure("Accuracy").gca()
  precision = plt.figure("Precision").gca()

  for name, (history, metrics) in data.items():
    # Métricas de plotagem
    loss.plot(history['val_loss'], label=name)
    accuracy.plot(history['val_acc'], label=name)
    precision.plot(history['val_precision'], label=name)

    # Imprimir resultados de testes
    print(name)
    for name, metric in metrics.items():
      print("{}: {}".format(name, metric))
    print()

  # Embeleze os gráficos
  loss.set_xlabel("Epochs")
  loss.set_ylabel("Loss (Categorical Crossentropy)")
  loss.set_title("Loss Evolution")
  loss.set_ylim(0, 0.15)
  loss.set_xticks(list(range(20)))
  loss.legend()

  accuracy.set_xlabel("Epochs")
  accuracy.set_ylabel("Accuracy")
  accuracy.set_title("Accuracy Evolution")
  accuracy.set_ylim(0.95, 1)
  accuracy.set_xticks(list(range(20)))
  accuracy.legend()

  precision.set_xlabel("Epochs")
  precision.set_ylabel("Precision")
  precision.set_title("Precision Evolution")
  precision.set_ylim(0.95, 1)
  precision.set_xticks(list(range(20)))
  precision.legend()
  return loss, accuracy, precision

def save_all_figs():
  for label in plt.get_figlabels():
    fig = plt.figure(label)
    fig.savefig('img/'+label+'.png')

def visualize_filters(model):
  # Carregar filtros para exibição
  layers = model.get_filters()

  # Mostrar todos os filtros de neurônios da primeira camada em um único gráfico
  layer = 'first_conv'
  fig = plt.figure(layer+'-filters-all_neurons')
  axes = fig.subplots(2,2)
  axes = list(axes.flatten())
  filt = layers[layer][0]
  ax_num = 0
  for i in range(filt.shape[2]):
    for o in range(filt.shape[3]):
      axes[ax_num].matshow(filt[:,:,i:i+1,o:o+1].squeeze(), cmap='gray')
      axes[ax_num].set_title('{}-{}-{}'.format(layer, i, o))
      remove_marks(axes[ax_num])
      ax_num += 1

  # Mostre todos os 4 canais de cada neurônio da segunda camada em um único gráfico
  layer = 'second_conv'
  filt = layers[layer][0]
  for o in range(filt.shape[3]):
    fig = plt.figure(layer+'-filters-neuron_'+str(o))
    axes = fig.subplots(2,2)
    axes = list(axes.flatten())
    ax_num = 0
    for i in range(filt.shape[2]):
      axes[ax_num].matshow(filt[:,:,i:i+1,o:o+1].squeeze(), cmap='gray')
      axes[ax_num].set_title('{}-{}-{}'.format(layer, i, o))
      remove_marks(axes[ax_num])
      ax_num += 1

def visualize_activations(model, X):
  layers = {
    'first_conv': ((2,2), {}),
    'second_conv': ((2,3), {'fontsize':10.5}),
  }
  
  # Apresente as ativações como exemplo
  for layer, (subplots, title_properties) in layers.items():
    # Obtenha a ativação para a entrada especificada
    activation = model.get_activation(X, layer_name=layer)
    fig = plt.figure(layer+'-neurons_activation')
    axes = fig.subplots(*subplots)
    axes = list(axes.flatten())
    for neuron in range(activation.shape[3]):
      axes[neuron].matshow(activation[0:1,:,:,neuron:neuron+1].squeeze(), cmap='gray')
      axes[neuron].set_title('{}-neuron_{}'.format(layer, neuron), **title_properties)
      remove_marks(axes[neuron])

def visualize_wrong(model, X_test, Y_test, n_samples=5):
  prediction = np.round(model.predict(X_test))
  cond = (prediction != Y_test).any(axis=1)
  X_wrong = X_test[cond]
  Y_wrong = Y_test[cond]
  pred_wrong = prediction[cond]

  for i in range(n_samples):
    X = X_wrong[i:i+1,:,:,:].squeeze()
    Y = Y_wrong[i:i+1,:].squeeze().argmax()
    pred = pred_wrong[i:i+1,:].squeeze().argmax()

    fig = plt.figure('wrong_prediction_{}'.format(i))
    fig.gca().matshow(X, cmap='gray')
    remove_marks(fig.gca())

    fig.text(0.68,0.75+0.11-0.05,'Esperado:',fontsize=28)
    fig.text(0.68,0.5+0.11-0.05, str(Y),fontsize=100)
    fig.text(0.68,0.25+0.11,'Predito:',fontsize=28)
    fig.text(0.68,0.11, str(pred),fontsize=100)
    fig.subplots_adjust(left=0.04, right=0.68)


# In[ ]:

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
