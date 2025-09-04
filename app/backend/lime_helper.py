import pandas as pd
import joblib
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

model_class = joblib.load(r"C:\Users\aleks\OneDrive\Documents\inzynierka\segmentation\models_paths\best_model_LightGBM3.pkl")
label_encoder = model_class["label_encoder"]

df1 = pd.read_csv(r'C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_single_cropped3\features_train.csv')
df2 = pd.read_csv(r'C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_single_cropped3\features_val.csv')
df_test = pd.read_csv(r'C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_single_cropped3\features_test.csv')
df_train = pd.concat([df1, df2], ignore_index=True)


TARGET = "class"
feature_names = list(df_train.drop(columns=[TARGET]).columns)

# 
X_train = df_train.drop(columns=[TARGET])
y_train = label_encoder.transform(df_train[TARGET])

df_test = df_test[df_test[TARGET].notna()].copy()
X_test = df_test.drop(columns=[TARGET])
y_test = label_encoder.transform(df_test[TARGET])

X_train_np = X_train.values
X_test_np  = X_test.values

explainer = LimeTabularExplainer(
    training_data=X_train_np,
    feature_names=feature_names,
    class_names=list(label_encoder.classes_),
    mode="classification",
    random_state=42
)

predict_fn = lambda data: model_class["model"].predict_proba(
    pd.DataFrame(data, columns=feature_names)
)


