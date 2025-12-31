import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Movie Recommendation System")
st.markdown("---")

@st.cache_data
def load_data():
    movies = pd.read_csv('movie_dataset/movies.csv')
    ratings = pd.read_csv('movie_dataset/ratings.csv')
    return movies, ratings

def display_dataframe(df, **kwargs):
    df_display = df.copy()
    for col in df_display.columns:
        if df_display[col].dtype == 'object':
            df_display[col] = df_display[col].astype(str)
    st.dataframe(df_display, **kwargs)

@st.cache_data
def preprocess_movies(movies):
    movies_processed = movies.copy()
    movies_processed['genres'] = movies_processed['genres'].str.split('|')
    movies_processed['genres_string'] = movies_processed['genres'].apply(
        lambda x: ' '.join(g.replace(' ', '') for g in x)
    )
    return movies_processed

@st.cache_resource
def prepare_content_based(movies_processed):
    tf_idf = TfidfVectorizer()
    tf_idf_matrix = tf_idf.fit_transform(movies_processed['genres_string'])
    cosine_sim = cosine_similarity(tf_idf_matrix, tf_idf_matrix)
    cosine_sim_df = pd.DataFrame(
        cosine_sim, 
        index=movies_processed['title'], 
        columns=movies_processed['title']
    )
    return cosine_sim_df

def get_movie_recommendations(movie_name, similarity_data, items, k=10):
    try:
        index = similarity_data.loc[:, movie_name].to_numpy().argpartition(
            range(-1, -k, -1)
        )
        closest_similarity = similarity_data.columns[index[-1:-(k+2):-1]]
        closest_similarity = closest_similarity.drop(movie_name, errors='ignore')
        return pd.DataFrame(closest_similarity).merge(items).head(k)
    except KeyError:
        return None

# Define RecommenderNet model
class RecommenderNet(tf.keras.Model):
    def __init__(self, total_user, total_movie, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.total_user = total_user
        self.total_movie = total_movie
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            total_user,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(total_user, 1)
        self.movie_embedding = layers.Embedding(
            total_movie,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.movie_bias = layers.Embedding(total_movie, 1)
    
    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        x = dot_user_movie + user_bias + movie_bias
        return tf.nn.sigmoid(x)

@st.cache_data
def prepare_collaborative_data(ratings):
    user_ids = ratings['userId'].unique().tolist()
    user_encoded = {user: i for i, user in enumerate(user_ids)}
    user_decoded = {i: user for user, i in user_encoded.items()}
    
    movie_ids = ratings['movieId'].unique().tolist()
    movie_encoded = {movie: i for i, movie in enumerate(movie_ids)}
    movie_decoded = {i: movie for movie, i in movie_encoded.items()}
    
    ratings_processed = ratings.copy()
    ratings_processed['user_id'] = ratings_processed['userId'].map(user_encoded)
    ratings_processed['movie_id'] = ratings_processed['movieId'].map(movie_encoded)
    
    return ratings_processed, user_encoded, user_decoded, movie_encoded, movie_decoded, len(user_ids), len(movie_ids)

@st.cache_resource
def load_or_train_model(ratings_processed, total_user, total_movie):
    model_path = 'model.weights.h5'
    
    ratings_shuffled = ratings_processed.sample(frac=1, random_state=42)
    ratings_fix = ratings_shuffled[['userId', 'movieId', 'rating', 'user_id', 'movie_id']]
    
    X = ratings_fix[['user_id', 'movie_id']].values
    y = ratings_fix['rating'].apply(lambda x: (x - 0.5) / (5.0 - 0.5)).values
    
    train_indices = int(ratings_fix.shape[0] * 0.8)
    X_train = X[:train_indices]
    X_val = X[train_indices:]
    y_train = y[:train_indices]
    y_val = y[train_indices:]
    
    model = RecommenderNet(total_user, total_movie, 100)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )
    
    dummy_input = np.array([[0, 0]])
    _ = model(dummy_input)
    
    if os.path.exists(model_path):
        with st.spinner("Loading model..."):
            model.load_weights(model_path)
        st.success("Model loaded successfully! Ready for recommendations.")
    else:
        st.warning("No pre-trained model found. Training new model...")
        st.info("**Tip:** Run `python train_model.py` once to avoid this wait in the future.")
        with st.spinner("Training model... This may take 2-5 minutes. Please wait..."):
            early_stopping = EarlyStopping(
                monitor='val_mean_absolute_error',
                min_delta=0.001,
                patience=7,
                restore_best_weights=True
            )
            
            history = model.fit(
                x=X_train,
                y=y_train,
                batch_size=32,
                epochs=50,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=0
            )
            
            model.save_weights(model_path)
            st.success("Model trained and saved successfully! Subsequent loads will be instant.")
    
    return model, X_train, X_val, y_train, y_val

def main():
    with st.spinner("Loading data..."):
        movies, ratings = load_data()
        movies_processed = preprocess_movies(movies)
    
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Choose a page:",
        ["Home", "Data Exploration", "Content-Based Filtering", "Collaborative Filtering", "ℹAbout"]
    )
    
    if page == "Home":
        st.header("Welcome to Movie Recommendation System!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Movies", f"{len(movies):,}")
            st.metric("Total Users", f"{ratings['userId'].nunique():,}")
        
        with col2:
            st.metric("Total Ratings", f"{len(ratings):,}")
            st.metric("Average Rating", f"{ratings['rating'].mean():.2f}")
        
        st.markdown("---")
        st.subheader("About This System")
        st.write("""
        This movie recommendation system uses two approaches:
        
        1. **Content-Based Filtering**: Recommends movies similar to what you like based on movie genres.
        2. **Collaborative Filtering**: Recommends movies based on user behavior and preferences.
        
        Use the sidebar to navigate through different sections!
        """)
    
    elif page == "Data Exploration":
        st.header("Data Exploration")
        
        tab1, tab2, tab3 = st.tabs(["Movies Dataset", "Ratings Dataset", "Visualizations"])
        
        with tab1:
            st.subheader("Movies Dataset")
            st.write(f"**Shape:** {movies.shape}")
            display_dataframe(movies.head(20), use_container_width=True)
            
            st.subheader("Dataset Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Missing Values:**")
                st.write(movies.isnull().sum())
            with col2:
                st.write("**Data Types:**")
                st.write(movies.dtypes)
        
        with tab2:
            st.subheader("Ratings Dataset")
            st.write(f"**Shape:** {ratings.shape}")
            display_dataframe(ratings.head(20), use_container_width=True)
            
            st.subheader("Statistical Summary")
            st.write(ratings.describe())
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Missing Values:**")
                st.write(ratings.isnull().sum())
            with col2:
                st.write(f"**Rating Range:** {ratings['rating'].min()} - {ratings['rating'].max()}")
        
        with tab3:
            st.subheader("Data Visualizations")
            
            # Rating distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Rating Distribution**")
                fig, ax = plt.subplots(figsize=(8, 5))
                bins = np.arange(0.5, 5.5, 0.5)
                ax.hist(ratings['rating'], bins=bins, edgecolor='black', color='skyblue')
                ax.set_xlabel('Rating')
                ax.set_ylabel('Count')
                ax.set_title('Distribution of Ratings')
                st.pyplot(fig)
            
            with col2:
                st.write("**User vs Movie Distribution**")
                total_user = len(ratings['userId'].unique())
                total_movie = len(ratings['movieId'].unique())
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.pie([total_user, total_movie], labels=['Users', 'Movies'], 
                       autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
                ax.set_title('Distribution of Users and Movies')
                st.pyplot(fig)
            
            st.write("**Top 10 Genres**")
            genreList = [genre for genres in movies_processed['genres'] for genre in genres]
            genre_counts = pd.Series(genreList).value_counts().head(10)
            fig, ax = plt.subplots(figsize=(10, 5))
            genre_counts.plot(kind='barh', ax=ax, color='coral')
            ax.set_xlabel('Count')
            ax.set_ylabel('Genre')
            ax.set_title('Top 10 Movie Genres')
            st.pyplot(fig)
    
    elif page == "Content-Based Filtering":
        st.header("Content-Based Movie Recommendations")
        st.write("Get movie recommendations based on genre similarity!")
        
        with st.spinner("Preparing recommendation model..."):
            cosine_sim_df = prepare_content_based(movies_processed)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("Search for a movie:", "")
        with col2:
            num_recommendations = st.slider("Number of recommendations:", 5, 20, 10)
        
        if search_query:
            matching_movies = movies_processed[
                movies_processed['title'].str.contains(search_query, case=False, na=False)
            ][['title', 'genres']].head(10)
            
            if not matching_movies.empty:
                st.subheader("Search Results:")
                display_dataframe(matching_movies, use_container_width=True)
                
                selected_movie = st.selectbox(
                    "Select a movie to get recommendations:",
                    matching_movies['title'].tolist()
                )
                
                if st.button("Get Recommendations", type="primary"):
                    recommendations = get_movie_recommendations(
                        selected_movie,
                        cosine_sim_df,
                        movies_processed[['title', 'genres']],
                        k=num_recommendations
                    )
                    
                    if recommendations is not None:
                        st.subheader(f"Movies similar to '{selected_movie}':")
                        
                        recommendations_display = recommendations.copy()
                        recommendations_display['genres'] = recommendations_display['genres'].apply(
                            lambda x: ', '.join(x) if isinstance(x, list) else x
                        )
                        recommendations_display.index = range(1, len(recommendations_display) + 1)
                        
                        display_dataframe(recommendations_display, use_container_width=True)
                    else:
                        st.error("Could not generate recommendations for this movie.")
            else:
                st.warning("No movies found matching your search.")
    
    elif page == "Collaborative Filtering":
        st.header("Collaborative Filtering Recommendations")
        st.write("Get personalized movie recommendations based on user preferences!")
        
        with st.spinner("Preparing collaborative filtering model..."):
            ratings_processed, user_encoded, user_decoded, movie_encoded, movie_decoded, total_user, total_movie = prepare_collaborative_data(ratings)
            model, X_train, X_val, y_train, y_val = load_or_train_model(ratings_processed, total_user, total_movie)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            user_input = st.number_input(
                "Enter User ID (or leave empty for random):",
                min_value=0,
                max_value=ratings['userId'].max(),
                value=0,
                step=1
            )
        with col2:
            num_recommendations = st.slider("Number of recommendations:", 5, 20, 10)
        
        if st.button("Generate Recommendations", type="primary"):
            if user_input > 0 and user_input in user_encoded:
                user_id = user_encoded[user_input]
                original_user_id = user_input
            else:
                user_id = ratings_processed['user_id'].sample(1).iloc[0]
                original_user_id = user_decoded[user_id]
                st.info(f"Using random User ID: {original_user_id}")
            
            movie_watched_by_user = ratings_processed[ratings_processed['user_id'] == user_id]
            
            if len(movie_watched_by_user) > 0:
                st.subheader(f"User {original_user_id}'s Top Rated Movies:")
                top_movies_user = (
                    movie_watched_by_user.sort_values(by='rating', ascending=False)
                    .head(5)
                    .movieId.values
                )
                
                movies_df_rows = movies[movies['movieId'].isin(top_movies_user)].copy()
                movies_df_rows['rating'] = movies_df_rows['movieId'].map(
                    movie_watched_by_user.set_index('movieId')['rating']
                )
                movies_df_rows = movies_df_rows[['title', 'genres', 'rating']]
                movies_df_rows.index = range(1, len(movies_df_rows) + 1)
                display_dataframe(movies_df_rows, use_container_width=True)
                
                movie_not_watched = movies[
                    ~movies['movieId'].isin(movie_watched_by_user.movieId.values)
                ]['movieId']
                movie_not_watched = list(
                    set(movie_not_watched).intersection(set(movie_encoded.keys()))
                )
                
                if len(movie_not_watched) > 0:
                    movie_not_watched_encoded = [[movie_encoded.get(x)] for x in movie_not_watched]
                    user_encoder = user_id
                    user_movies_array = np.hstack(
                        ([[user_encoder]] * len(movie_not_watched_encoded), movie_not_watched_encoded)
                    )
                    
                    with st.spinner("Generating recommendations..."):
                        predicted_ratings = model.predict(user_movies_array, verbose=0).flatten()
                    
                    top_ratings_indices = predicted_ratings.argsort()[-num_recommendations:][::-1]
                    recommended_movies_ids = [
                        movie_decoded.get(movie_not_watched_encoded[x][0]) 
                        for x in top_ratings_indices
                    ]
                    
                    st.subheader(f"🎬 Top {num_recommendations} Recommended Movies:")
                    recommended_movies = movies[movies['movieId'].isin(recommended_movies_ids)].copy()
                    recommended_movies['predicted_rating'] = recommended_movies['movieId'].map(
                        dict(zip(recommended_movies_ids, 
                                [predicted_ratings[i] * 4.5 + 0.5 for i in top_ratings_indices]))
                    )
                    recommended_movies = recommended_movies[['title', 'genres', 'predicted_rating']]
                    recommended_movies['predicted_rating'] = recommended_movies['predicted_rating'].round(2)
                    recommended_movies.index = range(1, len(recommended_movies) + 1)
                    display_dataframe(recommended_movies, use_container_width=True)
                else:
                    st.warning("This user has watched all movies!")
            else:
                st.error("User ID not found or has no ratings.")
    
    elif page == "About":
        st.header("About This Project")
        
        st.markdown("""
        ### Movie Recommendation System
        
        **Developer:** Wildan Aziz Hidayat
        
        #### Project Overview
        This application implements a hybrid movie recommendation system using two main approaches:
        
        #### 1. Content-Based Filtering
        - Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** for feature extraction
        - Calculates **Cosine Similarity** between movie genres
        - Recommends movies similar to what the user has watched
        
        #### 2. Collaborative Filtering
        - Uses **Neural Network** with embedding layers
        - Implements **Matrix Factorization** technique
        - Learns user preferences and movie characteristics
        - Uses **EarlyStopping** to prevent overfitting
        
        #### Dataset
        - **Source:** MovieLens
        - **Movies:** 10,329 movies with genres
        - **Ratings:** 105,339 ratings from users
        - **Rating Scale:** 0.5 to 5.0
        
        #### Technologies Used
        - **Frontend:** Streamlit
        - **Machine Learning:** TensorFlow/Keras, Scikit-learn
        - **Data Processing:** Pandas, NumPy
        - **Visualization:** Matplotlib, Seaborn
        
        #### Model Architecture
        The collaborative filtering model uses:
        - **Embedding Size:** 100
        - **Regularization:** L2 (1e-6)
        - **Activation:** Sigmoid
        - **Loss Function:** Binary Crossentropy
        - **Optimizer:** Adam (learning_rate=0.001)
        - **Metric:** Mean Absolute Error
        
        #### Key Features
        - Interactive data exploration
        - Real-time movie recommendations
        - Both content-based and collaborative filtering
        - Visual analytics and insights
        - User-friendly interface
        
        ---
        """)

if __name__ == "__main__":
    main()
