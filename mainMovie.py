# %% [markdown]
# # Movie Recommendation System - Wildan Aziz Hidayat

# %% [markdown]
# ## Import Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# %% [markdown]
# ## Data Loading

# %%
movies = pd.read_csv('/home/wildanaziz/movie-recommendation-system/movie_dataset/movies.csv')
ratings = pd.read_csv('/home/wildanaziz/movie-recommendation-system/movie_dataset/ratings.csv')

print("Total samples and shape in movies dataset: ", movies.shape)
print("Total samples and shape in ratings dataset: ", ratings.shape)

# %%
movies.info()

# %%
ratings.info()

# %%
ratings.describe()

# %%
print("lowest rating: ", ratings['rating'].min())
print("highest rating: ", ratings['rating'].max())

# %%
plt.figure(figsize=(10, 6))
bins=np.arange(0.5, 5.5, 0.5)
plt.hist(ratings['rating'], bins=bins)
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Distribution of Ratings')
plt.show()

# %%
total_user = len(ratings['userId'].unique())
total_movie = len(ratings['movieId'].unique())
print("Total user: ", total_user)
print("Total movie: ", total_movie)

plt.figure(figsize=(10, 6))
plt.pie([total_user, total_movie], labels=['User', 'Movie'], autopct='%1.1f%%')
plt.title('Distribution of User and Movie')
plt.show()

# %% [markdown]
# ## Data Preparation

# %%
movies.head()

# %% [markdown]
# #### Genre to List

# %%
movies['genres'] = movies['genres'].str.split('|')

# %%
movies.head()

# %% [markdown]
# ### Checking Missing Value in Movies

# %%
movies.isnull().sum()

# %%
genreList = [genre for genres in movies['genres'] for genre in genres]

uniqueGenre = pd.Series(genreList).unique()
print("Total genre: ", len(uniqueGenre))
print("Unique genre: ", uniqueGenre)

# %% [markdown]
# #### genreList to str

# %%
movies['genres_string'] = movies['genres'].apply(lambda x: ' '.join(g.replace(' ', '') for g in x))
movies.head()

# %% [markdown]
# ## Rating Prep

# %% [markdown]
# ### Checking Missing Values

# %%
ratings.isnull().sum()

# %% [markdown]
# ### Encode userId and movieId

# %%
user_ids = ratings['userId'].unique().tolist()
print("Total user_id: ", len(user_ids))

user_encoded = {user: i for i, user in enumerate(user_ids)}
print("encoded user_id: ", user_encoded)

user_decoded = {i: user for user, i in enumerate(user_ids)}
print("decoded user_id: ", user_decoded)

# %%
movie_ids = ratings['movieId'].unique().tolist()
print("Total movie_id: ", len(movie_ids))

movie_encoded = {movie: i for i, movie in enumerate(movie_ids)}
print("encoded movie_id: ", movie_encoded)

movie_decoded = {i: movie for movie, i in enumerate(movie_ids)}
print("decoded movie_id: ", movie_decoded)

# %% [markdown]
# ### Map into new columns

# %%
ratings['user_id'] = ratings['userId'].map(user_encoded)
ratings['movie_id'] = ratings['movieId'].map(movie_encoded)

# %% [markdown]
# ### Splitting Data

# %%
ratings = ratings.sample(frac=1, random_state=42)
ratings

# %%
ratings_fix = ratings[['userId', 'movieId', 'rating', 'user_id', 'movie_id']]
ratings_fix

# %%
X = ratings_fix[['user_id', 'movie_id']]

y = ratings_fix['rating'].apply(lambda x: (x - 0.5) / (5.0 - 0.5)).values

train_indices = int(ratings_fix.shape[0] * 0.8)
X_train, X_val, y_train, y_val = (
    X[:train_indices],
    X[train_indices:],
    y[:train_indices],
    y[train_indices:],
)

print("X_train shape: ", X_train.shape)
print("X_val shape: ", X_val.shape)
print("y_train shape: ", y_train.shape)
print("y_val shape: ", y_val.shape)
print("Training data: ", X)
print("Target data: ", y)

# %% [markdown]
# ## Content-Based Filtering

# %%
data_content_based = movies
data_content_based.sample(10)

# %% [markdown]
# ### TF-IDF

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf = TfidfVectorizer()
tf_idf.fit(data_content_based['genres_string'])

tf_idf.get_feature_names_out()

# %%
tf_idf_matrix = tf_idf.transform(data_content_based['genres_string'])

tf_idf_matrix.shape

# %%
pd.DataFrame(tf_idf_matrix.todense(), columns=tf_idf.get_feature_names_out(), index=data_content_based['title']).sample(10, axis=1).sample(5, axis=0)

# %% [markdown]
# ### Cosine Similarity

# %%
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(tf_idf_matrix, tf_idf_matrix)
cosine_sim

# %% [markdown]
# ### cosine_sim DataFrame

# %%
cosine_sim_df = pd.DataFrame(cosine_sim, index=data_content_based['title'], columns=data_content_based['title'])
print(cosine_sim_df.shape)

cosine_sim_df.sample(10, axis=1).sample(5, axis=0)

# %% [markdown]
# ### Top-N Recommendations

# %%
def get_movie_recommendations(movie_name, similarity_data=cosine_sim_df, items=data_content_based[['title', 'genres']], k=5):
  

  index = similarity_data.loc[:, movie_name].to_numpy().argpartition(
      range(-1, -k, -1)
  )

  closest_similarity = similarity_data.columns[index[-1:-(k+2):-1]]

  closest_similarity = closest_similarity.drop(movie_name, errors='ignore')

  pd.set_option('display.max_columns', None)
  return pd.DataFrame(closest_similarity).merge(items).head(k)
     

# %%
movie_name_input = input("Enter movie name: ")
data_content_based[data_content_based['title'].str.contains(movie_name_input, case=False)]

# %%
get_movie_recommendations("Sherlock Holmes (2010)", k=10)

# %% [markdown]
# ### Collaborative Filtering

# %%
class RecommenderNet(tf.keras.Model):
 
  # Insialisasi fungsi
  def __init__(self, total_user, total_movie, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.total_user = total_user
    self.total_movie = total_movie
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding( # layer embedding user
        total_user,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(total_user, 1) # layer embedding user bias
    self.movie_embedding = layers.Embedding( # layer embeddings movies
        total_movie,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.movie_bias = layers.Embedding(total_movie, 1) # layer embedding movies bias
 
  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0]) # memanggil layer embedding 1
    user_bias = self.user_bias(inputs[:, 0]) # memanggil layer embedding 2
    movie_vector = self.movie_embedding(inputs[:, 1]) # memanggil layer embedding 3
    movie_bias = self.movie_bias(inputs[:, 1]) # memanggil layer embedding 4
 
    dot_user_movie = tf.tensordot(user_vector, movie_vector, 2) 
 
    x = dot_user_movie + user_bias + movie_bias
    
    return tf.nn.sigmoid(x) # activation sigmoid

# %%
model = RecommenderNet(total_user, total_movie, 100) # inisialisasi model
 
# model compile
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.MeanAbsoluteError()]
)

# %%
# Definisikan callback EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_mean_absolute_error', # Metrik yang dipantau pada data validasi
    min_delta=0.001,                  # Perubahan minimum untuk dianggap perbaikan
    patience=7,                      # Jumlah epoch tanpa perbaikan sebelum berhenti
    restore_best_weights=True         # Mengembalikan bobot terbaik setelah berhenti
)

# Melatih model dengan callback EarlyStopping
models = model.fit(
    x=X_train,
    y=y_train,
    batch_size=32,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]        # Tambahkan callback di sini
)

# %%
plt.plot(models.history['mean_absolute_error'])
plt.plot(models.history['val_mean_absolute_error'])
plt.title('model_metrics')
plt.ylabel('mean_absolute_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%
movies_df = movies

# get sample user
user_id = ratings_fix.user_id.sample(1).iloc[0]
movie_watched_by_user = ratings_fix[ratings_fix.user_id == user_id]

movie_not_watched = movies_df[~movies_df['movieId'].isin(movie_watched_by_user.movieId.values)]['movieId']
movie_not_watched = list(
    set(movie_not_watched)
    .intersection(set(movie_encoded.keys()))
)

movie_not_watched = [[movie_encoded.get(x)] for x in movie_not_watched]
user_encoder = user_encoded.get(user_id)
user_movies_array = np.hstack(
    ([[user_encoder]] * len(movie_not_watched), movie_not_watched)
)

# %%
ratings = model.predict(user_movies_array).flatten()

top_ratings_indices = ratings.argsort()[-10:][::1]
recommended_movies_ids = [
    movie_decoded.get(movie_not_watched[x][0]) for x in top_ratings_indices
]

print(f"Showing recommendations for user: {user_id}")
print("=" * 40)

print("Movies with high ratings from user")
print("-" * 40)

top_movies_user = (
    movie_watched_by_user.sort_values(
        by = 'rating',
        ascending=False
    )
    .head(5)
    .movieId.values
)

anime_df_rows = movies_df[movies_df['movieId'].isin(top_movies_user)]
for row in anime_df_rows.itertuples():
  print(f"{row.title} : {', '.join(row.genres)}")

print('-' * 40)
print("Top 10 movies recommendations")
print('-' * 40)

recommended_movies = movies_df[movies_df['movieId'].isin(recommended_movies_ids)]
for row in recommended_movies.itertuples():
  print(f"{row.title} : {', '.join(row.genres)}")


