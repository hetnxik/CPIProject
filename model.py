import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras

warnings.filterwarnings('ignore')

# Load the dataset
spotify_df = pd.read_csv('./data_moods.csv')  # Assuming the dataset is loaded locally

# Feature engineering
spotify_df['duration_min'] = spotify_df['length'] / 60000  # Use 'length' instead of 'duration_ms'
spotify_df['intensity'] = spotify_df['loudness'] * spotify_df['tempo']

# Define categories for the mood variable
categories = ['sad', 'mellow', 'upbeat', 'happy']

# Create a categorical variable for the overall mood of the song
spotify_df['mood'] = pd.cut(
    x=spotify_df['valence'],
    bins=[0, 0.25, 0.5, 0.75, 1.0],
    labels=categories
)

# Add new categories to the mood variable
spotify_df['mood'] = spotify_df['mood'].cat.add_categories(['high energy', 'fast-paced'])
spotify_df.loc[spotify_df['energy'] >= 0.7, 'mood'] = 'high energy'
spotify_df.loc[spotify_df['tempo'] >= 120, 'mood'] = 'fast-paced'

# Select a subset of features for clustering (excluding 'track_genre' for now)
features_temp = ['danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness',
                 'liveness', 'valence', 'tempo']
spotify_df_temp = spotify_df[features_temp]

# Feature engineering for clustering
spotify_df_temp['tempo_variance'] = spotify_df_temp['tempo'].rolling(window=10).var()
spotify_df_temp['speechiness_variance'] = spotify_df_temp['speechiness'].rolling(window=10).var()
spotify_df_temp = spotify_df_temp.dropna()

# Use KMeans clustering to identify genre clusters
X = spotify_df_temp.values
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
spotify_df_temp['genre'] = kmeans.labels_

# Map genre cluster labels to genre names
genre_names = {0: 'Rock', 1: 'Pop', 2: 'Hip Hop', 3: 'Electronic', 4: 'Other'}
spotify_df_temp['genre_name'] = spotify_df_temp['genre'].apply(lambda x: genre_names[x])

# Create a 'liked' column based on the popularity score
spotify_df['liked'] = spotify_df['popularity'] > 50  # Adjust the threshold if needed

# Select features for training the model
features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'speechiness',
            'tempo', 'valence']
target = 'liked'

# Split data into training and test sets
train_df, test_df = train_test_split(spotify_df, test_size=0.3, random_state=39)
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# Define the models
logreg = LogisticRegression()
rf = RandomForestClassifier()
xgb = XGBClassifier()

# Train and evaluate models
models = [logreg, rf, xgb]
model_names = ['Logistic Regression', 'Random Forest', 'XGBoost']


def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    accuracies = []
    for model, name in zip(models, model_names):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append((name, accuracy))
        print(f'{name} Accuracy: {round(accuracy * 100, 2)}%')
        print(f'{name} Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    return accuracies


# Train and evaluate all models
train_and_evaluate(models, X_train, y_train, X_test, y_test)


# Define the neural network
def build_and_train_nn(X_train, y_train, X_test, y_test):
    model = keras.Sequential([
        keras.layers.Dense(64, input_shape=(len(features),), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='relu')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, verbose=0)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Neural Network Test accuracy:', test_acc)
    return test_acc


# Train and evaluate the neural network
accuracy_nn = build_and_train_nn(X_train, y_train, X_test, y_test)


# Create a function to recommend songs based on the mood
def recommend_songs(mood, spotify_df, top_n=10):
    mood_songs = spotify_df[spotify_df['mood'] == mood]
    if mood_songs.empty:
        print(f"No songs found for mood: {mood}")
        return None

    recommended_songs = mood_songs[['name', 'artist', 'mood']].sample(n=top_n)
    return recommended_songs


# Example: Recommend songs for a happy mood
# mood_input = 'happy'
# recommendations = recommend_songs(mood_input, spotify_df)
# print(f"Top {len(recommendations)} songs for the mood '{mood_input}':")
# print(recommendations)

# Plot accuracies of the models
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Network'],
    'Accuracy': [accuracy_score(y_test, logreg.predict(X_test)) * 100,
                 accuracy_score(y_test, rf.predict(X_test)) * 100,
                 accuracy_score(y_test, xgb.predict(X_test)) * 100,
                 accuracy_nn * 100]
})

ax = sns.barplot(data=results, x='Model', y='Accuracy', palette="mako")
for i in ax.containers:
    ax.bar_label(i)
plt.show()
