import pathlib
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


DATA_PATH      = pathlib.Path("./data/conn.log.labeled")   # ← change if needed
BATCH_SIZE     = 8192
EPOCHS         = 15
RANDOM_STATE   = 42
TEST_SIZE      = 0.20
LEARNING_RATE  = 1e-3
DROPOUT        = 0.3
HIDDEN_UNITS   = [256, 128, 64]

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    comment="#",
    low_memory=False,
)

#  Cleaning 

def cast_bool_to_float(x):
    """Keras‑compatible function that casts a bool tensor to float32."""
    return tf.cast(x, tf.float32)


df.columns = df.columns.str.strip()
label_map = {"Benign": 0, "Malicious": 1}
df["label"] = df["label"].map(label_map).astype(int)

# num columns
NUMERIC_COLS = [
    "duration", "orig_bytes", "resp_bytes", "missed_bytes",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
]
for c in NUMERIC_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# bools
BOOL_MAP = {"T": True, "F": False, "true": True, "false": False}
BOOL_COLS = ["local_orig", "local_resp"]
for c in BOOL_COLS:
    df[c] = df[c].map(BOOL_MAP).astype(bool)


# Expand the two set‑type columns

def expand_set_column(df_: pd.DataFrame, col_name: str) -> list[str]:
    token_set = set()
    for v in df_[col_name].dropna():
        token_set.update(v.split(","))
    new_names = []
    for token in token_set:
        new_name = f"{col_name}__{token}"
        df_[new_name] = df_[col_name].fillna("").apply(
            lambda x: int(token in x.split(","))
        )
        new_names.append(new_name)
    return new_names


history_flags = expand_set_column(df, "history")
tunnel_flags  = expand_set_column(df, "tunnel_parents")

# Train / test split 

FEATURE_COLUMNS = (
    NUMERIC_COLS
    + history_flags
    + tunnel_flags
    + BOOL_COLS
    + ["id.orig_h", "id.resp_h", "proto", "service", "conn_state"]
)

X = df[FEATURE_COLUMNS]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

# Pre-processing

# Normalisation nums
numeric_features = NUMERIC_COLS + history_flags + tunnel_flags
normalizer = {}
for name in numeric_features:
    norm = layers.Normalization()
    norm.adapt(np.array(X_train[name]).reshape(-1, 1))
    normalizer[name] = norm

# String lookup/category coding
categorical_features = ["id.orig_h", "id.resp_h", "proto", "service", "conn_state"]
cat_lookup = {}
cat_onehot = {}
for name in categorical_features:
    lookup = layers.StringLookup(output_mode="int", mask_token=None)
    lookup.adapt(X_train[name].fillna("___MISSING___"))
    cat_lookup[name] = lookup

    onehot = layers.CategoryEncoding(
        num_tokens=lookup.vocabulary_size(),
        output_mode="binary",
    )
    cat_onehot[name] = onehot

# Keras model assembly

def make_preprocessor() -> tf.keras.Model:
    # numeric inputs -------------------------------------------------
    num_in = {
        n: tf.keras.Input(shape=(1,), name=n, dtype=tf.float32)
        for n in numeric_features
    }
    num_out = [normalizer[n](num_in[n]) for n in numeric_features]

    # bool inputs ----------------------------------------------------
    bool_in = {
        n: tf.keras.Input(shape=(1,), name=n, dtype=tf.bool)
        for n in BOOL_COLS
    }
    # <-- use the named function instead of a lambda
    bool_out = [
        tf.keras.layers.Lambda(cast_bool_to_float, name=f"{n}_float")(bool_in[n])
        for n in BOOL_COLS
    ]

    # categorical inputs --------------------------------------------
    cat_in = {
        n: tf.keras.Input(shape=(1,), name=n, dtype=tf.string)
        for n in categorical_features
    }
    cat_out = [
        cat_onehot[n](cat_lookup[n](cat_in[n]))
        for n in categorical_features
    ]

    # concatenate ----------------------------------------------------
    concatenated = tf.keras.layers.Concatenate()(
        num_out + bool_out + cat_out
    )

    return tf.keras.Model(
        inputs={**num_in, **bool_in, **cat_in},
        outputs=concatenated,
        name="preprocessor",
    )


preprocessor = make_preprocessor()


# Build Model 


def build_classifier(input_dim: int,
                    hidden_units: list[int] = HIDDEN_UNITS,
                    dropout: float = DROPOUT) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = inputs
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    logits = tf.keras.layers.Dense(1, name="logits")(x)
    prob   = tf.keras.layers.Activation("sigmoid", name="prob")(logits)
    return tf.keras.Model(inputs, prob, name="suricata_flow_classifier")


classifier = build_classifier(preprocessor.output_shape[1])
classifier.summary()

# TF data pipelines

def df_to_dataset(df_: pd.DataFrame,
                  label_: pd.Series,
                  shuffle: bool = True,
                  batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((dict(df_), label_.values))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df_), seed=RANDOM_STATE,
                        reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


train_ds = df_to_dataset(X_train, y_train, shuffle=True)
val_ds   = df_to_dataset(X_test,  y_test,  shuffle=False)


def map_preprocess(features, label):
    """Run the Keras preprocessor and cast label to float."""
    return preprocessor(features), tf.cast(label, tf.float32)


train_prepared = train_ds.map(map_preprocess)
val_prepared   = val_ds.map(map_preprocess)

# Compile 

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f" Positive‑class weight = {pos_weight:.3f}")

classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

#Start Training
history = classifier.fit(
    train_prepared,
    epochs=EPOCHS,
    validation_data=val_prepared,
    class_weight={0: 1.0, 1: pos_weight},
    verbose=2,
)

print("\n Training finished.")

# Save the model 

model_dir = pathlib.Path("./saved_model")
model_dir.mkdir(parents=True, exist_ok=True)

preprocessor.save("./saved_model/preprocessor_tf.keras")
classifier.save("./saved_model/classifier_tf.keras")


print("Complete Training")


# -----------------------------------------------------------------
# 1️⃣  Named function that will replace the original Lambda layer
# -----------------------------------------------------------------
def cast_bool_to_float(x):
    """Cast a bool tensor to float32 – exactly the behaviour used in training."""
    return tf.cast(x, tf.float32)

# -----------------------------------------------------------------
# 2️⃣  Load the two artefacts (pre‑processor + classifier)
# -----------------------------------------------------------------
model_dir = pathlib.Path("./saved_model")



preprocessor = tf.keras.models.load_model(
    model_dir / "preprocessor_tf.keras",
    custom_objects={"cast_bool_to_float": cast_bool_to_float},
)

classifier = tf.keras.models.load_model(model_dir / "classifier_tf.keras")

# -----------------------------------------------------------------
# 3️⃣  Discover the expected feature names & dtypes from the pre‑processor
# -----------------------------------------------------------------
# Input tensors are a list; their names are stored in the tensor's .name
input_names = [tensor.name.split(":")[0] for tensor in preprocessor.inputs]
input_map   = dict(zip(input_names, preprocessor.inputs))

numeric_features = [n for n, t in input_map.items() if t.dtype == tf.float32]
bool_features    = [n for n, t in input_map.items() if t.dtype == tf.bool]
string_features  = [n for n, t in input_map.items() if t.dtype == tf.string]

print("\n=== Model expects the following feature groups ===")
print(f"numeric ({len(numeric_features)}): {numeric_features[:5]} …")
print(f"bool    ({len(bool_features)}): {bool_features}")
print(f"string  ({len(string_features)}): {string_features[:5]} …\n")

# -----------------------------------------------------------------
# 4️⃣  Helper – turn a plain Python dict into the TensorFlow dict the model wants
#     (each value is a **batch‑size‑1** tensor → shape (1,))
# -----------------------------------------------------------------
def make_tf_feature_dict(raw_dict: dict) -> dict:
    """Convert raw python values → tf.Tensors of shape (1,) with the correct dtype."""
    tf_dict = {}
    # numeric (float32)
    for name in numeric_features:
        value = raw_dict.get(name, 0.0)               # default = 0.0 if missing
        tf_dict[name] = tf.convert_to_tensor([value], dtype=tf.float32)

    # bool (bool)
    for name in bool_features:
        value = raw_dict.get(name, False)            # default = False if missing
        tf_dict[name] = tf.convert_to_tensor([value], dtype=tf.bool)

    # string (string)
    for name in string_features:
        value = raw_dict.get(name, "___MISSING___")   # same placeholder used in training
        tf_dict[name] = tf.convert_to_tensor([str(value)], dtype=tf.string)

    return tf_dict

# -----------------------------------------------------------------
# 5️⃣  Example: inference on a **single** new flow
# -----------------------------------------------------------------
# ----------------------------------------------------------------------
# Flow 1  – first line of the file
# ----------------------------------------------------------------------
flow_1 = {
    # ---- numeric columns --------------------------------------------------
    "duration":        0,          # “‑” in the source file
    "orig_bytes":      3.139211,
    "resp_bytes":      0,
    "missed_bytes":    0,          # “‑”
    "orig_pkts":       0,           # the value that appeared in the 'orig_pkts' column
    "orig_ip_bytes":   3,
    "resp_pkts":       180,
    "resp_ip_bytes":   0,

    # ---- boolean columns -------------------------------------------------
    "local_orig":      True,          # “S0” → non‑empty → True
    "local_resp":      False,         # “‑”

    # ---- categorical / string columns -----------------------------------
    "id.orig_h":       "CrDn63WjJEmrWGjqf",
    "id.resp_h":       "41040",
    "proto":           "80",
    "service":         "tcp",
    "conn_state":      "0",

    # ---- optional flag columns (expanded set columns) --------------------
    # The original `history` field contained the token “S”.
    "history__S":      1,
    # The `tunnel_parents` column contained a single value “0”.
    "tunnel_parents__0": 1,
}
# ----------------------------------------------------------------------
# Flow 2  – second line of the file
# ----------------------------------------------------------------------
flow_2 = {
    "duration":        0,
    "orig_bytes":      3.116726,
    "resp_bytes":      0,
    "missed_bytes":    0,
    "orig_pkts":       0,
    "orig_ip_bytes":   3,
    "resp_pkts":       180,
    "resp_ip_bytes":   0,

    "local_orig":      True,
    "local_resp":      False,

    "id.orig_h":       "CYee9y292pdYQE2S2g",
    "id.resp_h":       "49012",
    "proto":           "6667",
    "service":         "tcp",
    "conn_state":      "0",

    # expanded‑set flags
    "history__S":      1,
    "tunnel_parents__0": 1,
}


# Build the TF‑compatible dict (each entry now has shape (1,))
features_tf = make_tf_feature_dict(flow_1)


# Run through the two‑step pipeline
def run_prob(features_tf):
    dense_vec   = preprocessor(features_tf)          # shape (1, D)
    prob_tensor = classifier(dense_vec)             # shape (1, 1)

    probability = prob_tensor.numpy().item()        # python float
    threshold   = 0.5                               # you can change this
    pred_label  = int(probability >= threshold)

    print("=== Single‑flow inference ===")
    print(f"Malicious probability = {probability:.4f}")
    print(f"Predicted label       = {pred_label} (0 = benign, 1 = malicious)")

run_prob(features_tf)

features_tf = make_tf_feature_dict(flow_2)

run_prob(features_tf)