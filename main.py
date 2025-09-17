import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers

data_frame = pd.read_csv(
    "./data/conn.log.labeled",
    #"./data/test", 
    sep="\t", 
    comment="#", 
    low_memory=False
    )

label_mapping = {
    "Benign":      0,
    "Malicious":   1
    }

target_column = "label"

data_frame[target_column] = data_frame[target_column].map(label_mapping)

data_frame.columns = data_frame.columns.str.strip()

print("Cleaned column list →", data_frame.columns.tolist())
print("Class Dist")
print(data_frame[target_column].value_counts())

numeric_cols = [
    "duration","orig_bytes","resp_bytes","missed_bytes",
    "orig_pkts","orig_ip_bytes","resp_pkts","resp_ip_bytes"
]

for col in numeric_cols:
    data_frame[col] = pd.to_numeric(data_frame[col], errors="coerce")   # NaN for bad rows

bool_cols = ["local_orig","local_resp"]

for col in bool_cols:
    data_frame[col] = data_frame[col].map({"T":True, "F":False, "true":True, "false":False}).astype('bool')


data_frame[target_column]=data_frame[target_column].astype(int)

print("DF")
print(data_frame.columns.to_list())

X = data_frame.drop(columns=[target_column])      # drop the label column only
y = data_frame[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


def expand_set_column(df, col_name):
    """
    For a column that contains a CSV‑like string of tokens (e.g. "S,ShAD"),
    create one new column per token with value 0/1.
    Returns a list of the newly created column names.
    """
    # collect every distinct token that appears anywhere in the column
    token_set = set()
    for val in df[col_name].dropna():
        token_set.update(val.split(","))
    # create a new column per token
    new_cols = []
    for token in token_set:
        new_name = f"{col_name}__{token}"
        df[new_name] = df[col_name].fillna("").apply(
            lambda x: int(token in x.split(","))
        )
        new_cols.append(new_name)
    return new_cols

# Apply to the two set‑type columns (mutates `df` in‑place)
history_flags   = expand_set_column(data_frame, "history")
tunnel_flags    = expand_set_column(data_frame, "tunnel_parents")


#Lists of column names for the three groups
# ---- numeric (continuous) ------------------------------------------------
numeric_features = [
    "duration","orig_bytes","resp_bytes","missed_bytes",
    "orig_pkts","orig_ip_bytes","resp_pkts","resp_ip_bytes"
]

# add the binary flag columns created above (they are also numeric)
numeric_features += history_flags + tunnel_flags

# ---- boolean ------------------------------------------------------------
bool_features = ["local_orig","local_resp"]

# ---- low‑cardinality categoricals ----------------------------------------
categorical_features = [
    "id.orig_h","id.resp_h","proto","service","conn_state"
]

# Normalizer for each numeric column (learn mean & variance on the training set)
normalizer = {}
for name in numeric_features:
    norm = layers.Normalization()
    # reshape to (n_samples, 1) because the layer expects a 2‑D tensor
    norm.adapt(np.array(X_train[name]).reshape(-1, 1))
    normalizer[name] = norm

# StringLookup + One‑Hot for each categorical column
cat_lookup = {}
cat_onehot = {}
for name in categorical_features:
    # treat missing values as a separate token
    lookup = layers.StringLookup(output_mode="int", mask_token=None)
    lookup.adapt(X_train[name].fillna("___MISSING___"))
    cat_lookup[name] = lookup

    # one‑hot (you could swap this for an Embedding if you have many categories)
    onehot = layers.CategoryEncoding(num_tokens=lookup.vocab_size(), output_mode="binary")
    cat_onehot[name] = onehot


def df_to_dataset(df, label, shuffle=True, batch_size=8192):
    """
    Convert a pandas DataFrame + label Series to a tf.data.Dataset that yields
    a dictionary of feature tensors + the label tensor.
    """
    df = df.copy()
    ds = tf.data.Dataset.from_tensor_slices((dict(df), label.values))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = df_to_dataset(X_train, y_train, shuffle=True)
val_ds   = df_to_dataset(X_test,  y_test,  shuffle=False)


def build_model(input_dim, hidden_units=[256, 128, 64], dropout=0.3):
    inputs = layers.Input(shape=(input_dim,), name="features")
    x = inputs
    for units in hidden_units:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
    # final logit
    logits = layers.Dense(1, name="logits")(x)
    # sigmoid gives the probability of the *malicious* class
    outputs = layers.Activation("sigmoid", name="prob")(logits)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="suricata_flow_classifier")
    return model

# Build the model – we know the input dimension from the preprocessing step
input_dim = x_processed.shape[1]
model = build_model(input_dim)
model.summary()


# Compute the positive‑class weight (inverse of its frequency)
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"pos_weight = {pos_weight:.3f}")

# BinaryCrossentropy with `from_logits=False` because we already have a sigmoid output.
loss = tf.keras.losses.BinaryCrossentropy()
# Apply the weight manually inside the loss using `sample_weight` later,
# or use `tf.nn.weighted_cross_entropy_with_logits`.
# Here we pass the weight through `sample_weight` in `fit()`.

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=loss,
    metrics=[
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)


BATCH_SIZE = 8192
EPOCHS = 15

# Create a *mapped* dataset that runs the preprocessing layer on the fly.
def map_preprocess(features, label):
    return preprocess_features(features), tf.cast(label, tf.float32)

train_prepared = train_ds.map(map_preprocess)
val_prepared   = val_ds.map(map_preprocess)

history = model.fit(
    train_prepared,
    epochs=EPOCHS,
    validation_data=val_prepared,
    # class‑weight (same as `pos_weight` for the positive class)
    class_weight={0: 1.0, 1: pos_weight},
    verbose=2
)
