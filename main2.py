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
# filtered_df = df[~df['label'].isin(label_map.keys()), 'label']

# print(filtered_df)



df["label"] = df["label"].map(label_map).astype(int)

# num columns
NUMERIC_COLS = [
    "duration", "orig_bytes", "resp_bytes", "missed_bytes",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
]
for c in NUMERIC_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# bools
BOOL_MAP = {"T": True, "F": False, "true": True, "false": False, "-": False, "": False}
BOOL_COLS = ["local_orig", "local_resp"]
for c in BOOL_COLS:
    df[c] = df[c].map(BOOL_MAP).fillna(False).astype(bool)


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
print(X)

X.to_csv("dataConverted.csv", index=False)

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


print("Complete")