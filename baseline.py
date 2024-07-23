import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

zip_file = keras.utils.get_file(
    fname="cora.tgz",
    origin="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
    extract=True,
)

data_dir = os.path.join(os.path.dirname(zip_file), "cora")
print(data_dir)

# Citation Data
citations = pd.read_csv(
    os.path.join(data_dir, "cora.cites"),
    sep="\t",
    header=None,
    names=["target", "source"],
)

# print(citations.head(n=100))

# Papers Data
column_names = (
    ["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"]
)
papers = pd.read_csv(
    os.path.join(data_dir, "cora.content"),
    sep="\t",
    header=None,
    names=column_names,
)
# print(citations.head(n=100))
# print(papers.sample(5).T)

# Create zero-based id value for the data
class_values = sorted(papers["subject"].unique())
paper_values = sorted(papers["paper_id"].unique())
class_idx = {name: idx for idx, name in enumerate(class_values)}
paper_idx = {name: idx for idx, name in enumerate(paper_values)}

papers["paper_id"] = papers["paper_id"].apply(lambda name: paper_idx[name])
citations["source"] = citations["source"].apply(lambda name: paper_idx[name])
citations["target"] = citations["target"].apply(lambda name: paper_idx[name])
papers["subject"] = papers["subject"].apply(lambda value: class_idx[value])

# print(citations.head(n=100))
# print(papers.sample(5))


# Visualize the citation graph data
# print(citations.sample(n=1500))
plt.figure(figsize=(10, 10))
cora_graph = nx.from_pandas_edgelist(citations.sample(n=1500))
subjects = list(
    papers[papers["paper_id"].isin(list(cora_graph.nodes))]["subject"]
)
nx.draw_spring(cora_graph, node_size=15, node_color=subjects)

# plt.savefig("graph_data.png")

train_data, test_data = [], []

# Get the papers from each class (subject)
for _, group_data in papers.groupby("subject"):
    random_selection = np.random.rand(len(group_data.index)) <= 0.5
    train_data.append(group_data[random_selection])
    test_data.append(group_data[~random_selection])

# Shuffle the data
train_data = pd.concat(train_data).sample(frac=1)
test_data = pd.concat(test_data).sample(frac=1)

hidden_units = [32, 32]
learning_rate = 0.01
dropout_rate = 0.5
num_epochs = 300
batch_size = 256


def run_experiment(model, x_train, y_train):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=50, restore_best_weights=True
    )
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early_stopping],
    )
    return history


def display_learning_curves(history):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history["loss"])
    ax1.plot(history.history["val_loss"])
    ax1.legend(["train", "test"], loc="upper right")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2.plot(history.history["acc"])
    ax2.plot(history.history["val_acc"])
    ax2.legend(["train", "test"], loc="upper right")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")


def create_ffn(hidden_units, dropout_rate, name=None):
    ffn_layers = []
    for units in hidden_units:
        ffn_layers.append(keras.layers.BatchNormalization())
        ffn_layers.append(keras.layers.Dropout(dropout_rate))
        ffn_layers.append(keras.layers.Dense(units, activation=tf.nn.gelu))

    return keras.Sequential(ffn_layers, name=name)


feature_names = list(set(papers.columns) - {"paper_id", "subject"})
num_features = len(feature_names)
num_classes = len(class_idx)


x_train = train_data[feature_names].to_numpy()
x_test = test_data[feature_names].to_numpy()

y_train = train_data["subject"]
y_test = test_data["subject"]


def create_baseline_model(hidden_units, num_classes, dropout_rate=0.2):
    inputs = keras.layers.Input(shape=(num_features,))
    x = create_ffn(hidden_units, dropout_rate)(inputs)
    for _ in range(4):
        x1 = create_ffn(hidden_units, dropout_rate)(x)
        x = keras.layers.Add()([x, x1])

    logits = keras.layers.Dense(num_classes)(x)
    return keras.models.Model(inputs=inputs, outputs=logits)


model = create_baseline_model(hidden_units, num_classes, dropout_rate)
history = run_experiment(model, x_train, y_train)

display_learning_curves(history)

_, test_accuracy = model.evaluate(x=x_test, y=y_test, verbose=0)
print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")
