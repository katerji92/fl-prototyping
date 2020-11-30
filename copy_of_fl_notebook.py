import tensorflow_federated as tff
import numpy as np
import tensorflow as tf
import collections
import attr
import functools
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

from matplotlib import pyplot as plt
import nest_asyncio

np.random.seed(0)
nest_asyncio.apply()
tf.compat.v1.enable_eager_execution()

# Constants
NUMBER_OF_CLIENTS = 25
BATCH_SIZE = 20
NUMBER_OF_EPOCHS = 25
NUMBER_OF_CLUSTERS = 10
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10

# Get the dataset
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
# Pre-processing the data (Converting (28,28)pixel shape to 784pixel)

def preprocess(dataset):
  def batch_format_fn(element):
    return collections.OrderedDict(
        x=tf.reshape(element['pixels'], [-1, 784]),
        y=tf.reshape(element['label'], [-1, 1]))

  return dataset.repeat(NUMBER_OF_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
      BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

def preprocess_val(dataset):
  def batch_format_fn(element):
    return (tf.reshape(element['pixels'],[-1,784]),
            tf.reshape(element['label'],[-1,1]))
  return dataset.batch(BATCH_SIZE).map(batch_format_fn)

def get_dataset_for_client(client_id, dataset):
  return preprocess(dataset.create_tf_dataset_for_client(client_id))


# Get device ids
device_ids = emnist_train.client_ids
np.random.shuffle(device_ids)
device_ids = device_ids[:NUMBER_OF_CLIENTS]

# Get the device data
train_device_datasets = [get_dataset_for_client(device_id, emnist_train) for device_id in device_ids]
test_device_datasets = [get_dataset_for_client(device_id, emnist_test) for device_id in device_ids]
central_emnist_test = emnist_test.create_tf_dataset_from_all_clients().take(1000)
central_emnist_test = preprocess_val(central_emnist_test)

# Define the model
def keras_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10, kernel_initializer='zeros'),
    tf.keras.layers.Softmax(),
  ])

# Define TFF wrapper
def wrap_model_with_tff(model, input_spec):
  return tff.learning.from_keras_model(
    model, input_spec = input_spec,
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])

input_spec = train_device_datasets[0].element_spec

# FL Training
@tf.function
def server_update(model, mean_clients_weights):
  model_weights = model.weights.trainable

  tf.nest.map_structure(lambda x,y: x.assign(y),
                        model_weights,mean_clients_weights)
  return model_weights

@tff.tf_computation
def server_init():
  tff_model = wrap_model_with_tff(keras_model(), input_spec)
  return tff_model.weights.trainable

model_weights_type = server_init.type_signature.result
tf_dataset_type = tff.SequenceType(input_spec)

@tff.tf_computation(model_weights_type)
def server_update_fn(mean_client_weights):
  tff_model = wrap_model_with_tff(keras_model(), input_spec)
  return server_update(tff_model, mean_client_weights)

@tf.function
def client_update(model, dataset, server_weights, client_optimizer):
  #Initialize the client weights with the server weights
  client_weights = model.weights.trainable

  #Assign the server weights to the client model
  tf.nest.map_structure(lambda x,y: x.assign(y),
                        client_weights,server_weights)
  #Use the client optimizer to update the local model.
  for batch in dataset:
    with tf.GradientTape() as tape:
      #compute a forward pass on the batch of data
      outputs = model.forward_pass(batch)
    #compute the corresponding gradient
    grads = tape.gradient(outputs.loss, client_weights)
    grads_and_vars = zip(grads, client_weights)
    
    #apply the gradient using client optimizer
    client_optimizer.apply_gradients(grads_and_vars)

  return client_weights

@tff.tf_computation(tf_dataset_type, model_weights_type)
def client_update_fn(tf_dataset, server_weights):
  tff_model = wrap_model_with_tff(keras_model(), input_spec)
  client_optimizer = tf.keras.optimizers.Adam()
  return client_update(tff_model, tf_dataset, server_weights, client_optimizer)

federated_server_type = tff.FederatedType(model_weights_type, tff.SERVER)
federated_dataset_data = tff.FederatedType(tf_dataset_type, tff.CLIENTS)

@tff.federated_computation(federated_server_type, federated_dataset_data)
def next_fn(server_weights, federated_dataset):
  # Send server weights to clients
  server_weights_to_clients = tff.federated_broadcast(server_weights)

  # Each client computes their updated weights
  client_weights = tff.federated_map(client_update_fn, (federated_dataset,server_weights_to_clients))

  # Client mean
  mean_client_weights = tff.federated_mean(client_weights)

  # Server averages all the client weights
  mean_client_weights = tff.federated_mean(client_weights)

  # The server updates it model 
  server_weights = tff.federated_map(server_update_fn, mean_client_weights)

  return (server_weights, client_weights)

@tff.tf_computation(model_weights_type)
def client_work(model_weights):
  return model_weights

@tff.federated_computation(tff.FederatedType(model_weights_type, tff.CLIENTS))
def run_one_round(weights):
  tff_model = wrap_model_with_tff(keras_model(), input_spec)
  return tff.federated_map(client_work, weights)

@tff.federated_computation
def initialize_fn():
  return tff.federated_value(server_init(), tff.SERVER)

federated_algorithm = tff.templates.IterativeProcess(
    initialize_fn = initialize_fn, next_fn = next_fn
)

# Phase 1
def evaluate(server_state, dataset = central_emnist_test):
  model = keras_model()
  model.compile(
      loss = tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
  )
  model.set_weights(server_state)
  return model.evaluate(dataset)

# Train as a traditional FL model
server_state = federated_algorithm.initialize()
updated_server_state_phase_1 = server_state
updated_client_weights = []
for i in tqdm(range(0, NUMBER_OF_EPOCHS)):
  result = federated_algorithm.next(updated_server_state_phase_1, train_device_datasets)
  updated_server_state_phase_1 = result[0]
  updated_client_weights = result[1]
  evaluate(updated_server_state_phase_1)

# Phase 2
# Cluster the weights from Phase 1
client_weights_flat = [client_weight[0].numpy().reshape(-1) for client_weight in updated_client_weights]
number_of_clusters = NUMBER_OF_CLUSTERS
kmeans = KMeans(n_clusters=number_of_clusters)
kmeans.fit(client_weights_flat)
clusterd_weights_indexes = kmeans.predict(client_weights_flat)

# Calculate the each of the cluster's weights
cluster_device_map = {}
cluster_device_weight_map = {}
cluster_device_datasets = {}
for i in range(0, len(clusterd_weights_indexes)):
  cluster_key = str(clusterd_weights_indexes[i])
  if cluster_key not in cluster_device_map:
    cluster_device_map[cluster_key] = []
    cluster_device_weight_map[cluster_key] = []
    cluster_device_datasets[cluster_key] = []
  cluster_device_map[cluster_key].append(i)
  cluster_device_weight_map[cluster_key].append(updated_client_weights[i])
  cluster_device_datasets[cluster_key].append(train_device_datasets[int(cluster_key)])

# Calculate the averages of the clusters
cluster_device_average_weights = {}
for cluster_key in cluster_device_weight_map.keys():
  cluster_weights = np.array(cluster_device_weight_map[cluster_key])
  cluster_size = len(cluster_weights)
  cluster_sums = []
  for i in range(0, len(cluster_weights[0])):
    cluster_sums.append(np.add.reduce(np.array(cluster_weights)[:, i])/cluster_size)
  cluster_device_average_weights[cluster_key] = cluster_sums

# Clustered Training
# ToDo: How to set weights of indiviudal clients?
cluster_weights_after_training = {}
cluster_accuracies = {}
for cluster_key in cluster_device_average_weights.keys():
  server_state = federated_algorithm.initialize()
  server_update_fn(cluster_device_average_weights[cluster_key])
  updated_server_state = cluster_device_average_weights[cluster_key]
  run_one_round(updated_client_weights)
  updated_client_weights_curr = []
  cluster_eval_results = []
  for i in tqdm(range(0, NUMBER_OF_EPOCHS)):
    result = federated_algorithm.next(updated_server_state, cluster_device_datasets[cluster_key])
    updated_server_state = result[0]
    updated_client_weights_curr = result[1]
    cluster_eval_results = evaluate(updated_server_state)
  # cluster_weights_after_training[cluster_key] = (updated_server_state, updated_client_weights_curr)
  cluster_weights_after_training[cluster_key] = updated_server_state
  cluster_accuracies[cluster_key] = cluster_eval_results[1]

cluster_weights_sums = None
cluster_weights_sums_weighted = None
max_accuracy = np.max(np.array(list(cluster_accuracies.values())))
for cluster_key in cluster_weights_after_training.keys():
  curr_cluster_weights = np.array(cluster_weights_after_training[cluster_key])
  curr_accuracy_factor = cluster_accuracies[cluster_key] / max_accuracy
  if cluster_weights_sums is None:
    cluster_weights_sums = curr_cluster_weights
    cluster_weights_sums_weighted = curr_cluster_weights * curr_accuracy_factor
  else:
    for i in range(len(cluster_weights_sums)):
      cluster_weights_sums[i] = cluster_weights_sums[i] + curr_cluster_weights[i]
      cluster_weights_sums_weighted[i] = cluster_weights_sums_weighted[i] + curr_cluster_weights[i] * curr_accuracy_factor
cluster_weights_average = cluster_weights_sums / len(list(cluster_device_map.keys()))
cluster_weights_weighted_average = cluster_weights_sums_weighted / len(list(cluster_device_map.keys()))

results = {
  "Phase 1 eval": evaluate(updated_server_state_phase_1),
  "Phase 3 avg eval": evaluate(cluster_weights_average),
  "Phase 3 acc combination avg eval": evaluate(cluster_weights_weighted_average),
}

for cluster_key in cluster_device_map:
  for device_id in cluster_device_map[cluster_key]:
    results[f"Cluster {cluster_key} with device {device_id}:"] = evaluate((
        cluster_weights_after_training[cluster_key] + cluster_weights_weighted_average) / 2, 
        dataset = preprocess_val(emnist_test.create_tf_dataset_for_client(device_ids[device_id])
      ))

for result in results:
  print(f"{result}: {results[result]}")