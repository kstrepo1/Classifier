#!/usr/bin/env python3
# --------------------------------------------------------------
#  Suricata flow classifier – inference script (Option B)
# --------------------------------------------------------------

import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

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
# ----------------------------------------------------------------------
# flow_1  –  corresponds to the first line of the original table
# ----------------------------------------------------------------------
flow_1 = {
    # ---- numeric columns --------------------------------------------------
    "duration":        0,            # “‑” in the source file
    "orig_bytes":      3.139211,     # value that appears after the “service” column
    "resp_bytes":      0,
    "missed_bytes":    0,            # “‑”
    "orig_pkts":       3,
    "orig_ip_bytes":   180,
    "resp_pkts":       0,
    "resp_ip_bytes":   0,

    # ---- boolean columns -------------------------------------------------
    "local_orig":      True,         # conn_state = “S0” → non‑empty → True
    "local_resp":      False,        # “‑”

    # ---- categorical / string columns -----------------------------------
    "id.orig_h":       "CrDn63WjJEmrWGjqf",   # uid field (kept as‑is)
    "id.resp_h":       "41040",                # source port
    "proto":           "80",                   # destination port
    "service":         "tcp",                  # protocol name
    "conn_state":      "0",                    # stripped from “S0”

    # ---- optional flag columns (expanded set columns) --------------------
    # history contained the token “S”
    "history__S":      1,

    # tunnel_parents column contained a single value “0” (if present)
    "tunnel_parents__0": 1,
}

flow_2 = {
    #Num
    "duration": , 
    "orig_bytes": , 
    "resp_bytes": , 
    "missed_bytes": ,
    "orig_pkts": , 
    "orig_ip_bytes": , 
    "resp_pkts": , 
    "resp_ip_bytes": ,

    #Bool
    "local_orig": , 
    "local_resp": 


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

features_tf = make_tf_feature_dict(flow_3)

run_prob(features_tf)

features_tf = make_tf_feature_dict(flow_4)

run_prob(features_tf)

features_tf = make_tf_feature_dict(flow_5)

run_prob(features_tf)

# -----------------------------------------------------------------
# 6️⃣  OPTIONAL – batch inference from a CSV / DataFrame
# -----------------------------------------------------------------
def df_to_feature_dict(df: pd.DataFrame) -> dict:
    """Convert an entire DataFrame to the same dict format used above."""
    tf_dict = {}
    for name in numeric_features:
        tf_dict[name] = tf.convert_to_tensor(df[name].values.astype(np.float32))
    for name in bool_features:
        tf_dict[name] = tf.convert_to_tensor(df[name].values.astype(bool))
    for name in string_features:
        tf_dict[name] = tf.convert_to_tensor(df[name].astype(str).values,
                                             dtype=tf.string)
    return tf_dict

# Uncomment the block below if you have a TSV/CSV file of new flows:
"""
csv_path = pathlib.Path("./data/new_flows.tsv")
new_df = pd.read_csv(csv_path, sep="\t", comment="#")
new_df.columns = new_df.columns.str.strip()

# ---- apply the same cleaning you used for training ----
for c in numeric_features:
    new_df[c] = pd.to_numeric(new_df[c], errors="coerce").fillna(0)
for c in bool_features:
    BOOL_MAP = {"T": True, "F": False,
                "true": True, "false": False,
                True: True, False: False}
    new_df[c] = new_df[c].map(BOOL_MAP).fillna(False)

batch_features = df_to_feature_dict(new_df)
batch_probs = classifier(preprocessor(batch_features)).numpy().reshape(-1)

new_df["malicious_prob"] = batch_probs
new_df["predicted_label"] = (batch_probs >= threshold).astype(int)

print("\n=== First 5 predictions on the batch file ===")
print(new_df.head())
# Optionally store the results:
# new_df.to_csv("new_flows_with_predictions.tsv", sep="\t", index=False)
"""

# --------------------------------------------------------------
# End of script
# --------------------------------------------------------------
