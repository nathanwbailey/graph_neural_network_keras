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

feature_names = list(set(papers.columns) - {"paper_id", "subject"})
num_features = len(feature_names)
num_classes = len(class_idx)

x_train = train_data[feature_names].to_numpy()
x_test = test_data[feature_names].to_numpy()

y_train = train_data["subject"]
y_test = test_data["subject"]


# Source paper cites target papers
# Edges are laid out like:
# [0, 1, 1, 0, 0]
# [5, 1, 8, 9, 7]
# Neighbours of 0 are 5, 9, 7
# Neighbours of 1 are 1, 8
edges = citations[["source", "target"]].to_numpy().T
# Set to ones, as no weights needed here
edge_weights = tf.ones(shape=edges.shape[1])
# Node features contains data associated with each node
node_features = tf.cast(
    papers.sort_values("paper_id")[feature_names].to_numpy(),
    dtype=tf.dtypes.float32,
)

# Collate graph info
graph_info = (node_features, edges, edge_weights)

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

    plt.savefig("training_graphs.png")


def create_ffn(hidden_units, dropout_rate, name=None):
    ffn_layers = []
    for units in hidden_units:
        ffn_layers.append(keras.layers.BatchNormalization())
        ffn_layers.append(keras.layers.Dropout(dropout_rate))
        ffn_layers.append(keras.layers.Dense(units, activation=tf.nn.gelu))

    return keras.Sequential(ffn_layers, name=name)


class GraphConvLayer(keras.layers.Layer):
    def __init__(
        self,
        hidden_units,
        dropout_rate=0.2,
        aggregration_type="mean",
        combination_type="concat",
        normalize=False,
    ):
        super().__init__()

        self.aggregation_type = aggregration_type
        self.combination_type = combination_type
        self.normalize = normalize

        self.ffn_prepare = create_ffn(hidden_units, dropout_rate)
        self.update_fn = create_ffn(hidden_units, dropout_rate)

    def prepare(self, node_representations, weights=None):
        # Node features will be passed through a NN to produce messages
        messages = self.ffn_prepare(node_representations)
        if weights is not None:
            messages = messages * tf.expand_dims(weights, -1)
        return messages

    def aggregate(
        self, node_indices, neighbour_messages, node_representations
    ):
        num_nodes = node_representations.shape[0]

        # Aggregate the messages corresponding to the node neighbours
        # Messages matching the node index will be summed
        # E.g. neighbour_messages = [5, 1, 8, 9, 7]
        # Node Indices = [0, 1, 1, 0, 0]
        # Result is [21, 9]
        if self.aggregation_type == "sum":
            aggregated_message = tf.math.unsorted_segment_sum(
                neighbour_messages, node_indices, num_segments=num_nodes
            )

        elif self.aggregation_type == "mean":
            aggregated_message = tf.math.unsorted_segment_mean(
                neighbour_messages, node_indices, num_segments=num_nodes
            )

        elif self.aggregation_type == "max":
            aggregated_message = tf.math.unsorted_segment_max(
                neighbour_messages, node_indices, num_segments=num_nodes
            )

        return aggregated_message

    def update(self, node_representations, aggregated_messages):
        if self.combination_type == "concat":
            h = tf.concat([node_representations, aggregated_messages], axis=1)
        elif self.combination_type == "add":
            h = node_representations + aggregated_messages

        node_embeddings = self.update_fn(h)
        return node_embeddings

    def call(self, inputs):
        node_representations, edges, edge_weights = inputs
        node_indices, neighbour_indices = edges[0], edges[1]
        # Expand the representations so we create a copy of the neighbour for each node that links to it
        # We pass each neighbour representation through the same weight matrix, so same result for same representation
        # Allows us to share the weight for the layer
        neighbour_representations = tf.gather(
            node_representations, neighbour_indices
        )
        neighbour_messages = self.prepare(
            neighbour_representations, edge_weights
        )
        aggregated_messages = self.aggregate(
            node_indices, neighbour_messages, node_representations
        )
        # Perform the update
        return self.update(node_representations, aggregated_messages)


class GNNNodeClassifier(keras.models.Model):
    def __init__(
        self,
        graph_info,
        num_classes,
        hidden_units,
        aggregation_type="sum",
        combination_type="concat",
        dropout_rate=0.2,
    ):
        super().__init__()

        node_features, edges, edge_weights = graph_info
        self.node_features = node_features
        self.edges = edges
        self.edge_weights = edge_weights

        if self.edge_weights is None:
            self.edge_weights = tf.ones(shape=edges.shape[1])

        self.edge_weights = self.edge_weights / tf.math.reduce_sum(
            self.edge_weights
        )

        self.preprocess = create_ffn(hidden_units, dropout_rate)

        self.conv1 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
        )
        self.conv2 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
        )

        self.postprocess = create_ffn(hidden_units, dropout_rate)
        self.compute_logits = keras.layers.Dense(units=num_classes)

    def call(self, input_node_indices):
        x = self.preprocess(self.node_features)
        x1 = self.conv1((x, self.edges, self.edge_weights))
        x = x1 + x
        x2 = self.conv2((x, self.edges, self.edge_weights))
        x = x2 + x
        x = self.postprocess(x)
        # Get the final node embeddings for the batch of  node IDs we passed in
        node_embeddings = tf.gather(x, input_node_indices)
        # Compute the logits for classification
        return self.compute_logits(node_embeddings)


gnn_model = GNNNodeClassifier(
    graph_info=graph_info,
    num_classes=num_classes,
    hidden_units=hidden_units,
    dropout_rate=dropout_rate,
)


print("GNN output shape:", gnn_model([1, 10, 100]))

gnn_model.summary()

# For each paper ID,
x_train = train_data.paper_id.to_numpy()
history = run_experiment(gnn_model, x_train, y_train)
display_learning_curves(history)

x_test = test_data.paper_id.to_numpy()
_, test_accuracy = gnn_model.evaluate(x=x_test, y=y_test, verbose=0)
print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")
