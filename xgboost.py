import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Load and preprocess the dataset
spotify_df = pd.read_csv('/Users/ayrafraihan/PycharmProjects/CPIProject/data_moods.csv')

# Add calculated features
spotify_df['duration_min'] = spotify_df['length'] / 60000
spotify_df['intensity'] = spotify_df['loudness'] * spotify_df['tempo']
categories = ['sad', 'mellow', 'upbeat', 'happy']
spotify_df['mood'] = pd.cut(x=spotify_df['valence'], bins=[0, 0.25, 0.5, 0.75, 1.0], labels=categories)

# Add high energy and fast-paced categories
spotify_df['mood'] = spotify_df['mood'].cat.add_categories(['high energy', 'fast-paced'])
spotify_df.loc[spotify_df['energy'] >= 0.7, 'mood'] = 'high energy'
spotify_df.loc[spotify_df['tempo'] >= 120, 'mood'] = 'fast-paced'

# Select features and target variable
features = ['danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo']
X = spotify_df[features]
y = spotify_df['mood']

# Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Train XGBoost classifier
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(label_encoder.classes_), use_label_encoder=False)
xgb_model.fit(X_train, y_train)

# Predict on the entire test set
y_pred = xgb_model.predict(X_test)

# Convert the predictions and true labels back to their original class labels for interpretation
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap for All Moods (XGBoost)")
plt.show()

print("Classification Report for All Moods:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
