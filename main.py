import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def splot():
    # Plotting histogram for the 'rating' column
    plt.figure(figsize=(8, 6))
    sns.histplot(ratings['rating'], kde=True, bins=20, color='blue')

    # Adding labels and title
    plt.title('Distribution of Movie Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')

    # Show plot
    plt.show()

def content_based(movie_id):
    # Stor relevant columns
    movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))  # Split genres into a list

    # Combine movie titles and genres into a single text column
    movies['content'] = movies['title'] + " " + movies['genres'].apply(lambda x: " ".join(x))

    # Convert content into a matrix of TF-IDF features
    tfidf = TfidfVectorizer(stop_words='english')
    content_matrix = tfidf.fit_transform(movies['content'])

    # Calculate pairwise cosine similarity between items
    content_similarity = cosine_similarity(content_matrix)

    # Find the index corresponding to the given movie_id
    if movie_id not in movies['movieId'].values:
        raise ValueError("Movie ID not found in the dataset.")

    movie_index = movies.index[movies['movieId'] == movie_id][0]

    # Retrieve similarity scores for the selected movie
    similarity_scores = content_similarity[movie_index]

    # Create a DataFrame with similarity scores and corresponding movie title
    similar_movies = pd.DataFrame({
        'movieId': movies['movieId'],
        'title': movies['title'],
        'similarity': similarity_scores
    })

    # Sort movies by similarity scores in descending order
    similar_movies = similar_movies.sort_values(by='similarity', ascending=False)

    return similar_movies

def user_based(user_id):
    # Merge the ratings with the movie titles from the movies dataframe
    ratings_with_titles = pd.merge(ratings, movies[['movieId', 'title']], on='movieId', how='left')

    # Create the user-movie ratings matrix
    user_movie_matrix = ratings_with_titles.pivot_table(index='userId', columns='title', values='rating')

    # Fill NaN values with 0 (since KNN cannot handle NaNs)
    user_movie_matrix_filled = user_movie_matrix.fillna(0)

    # Initialize the KNN model
    knn = NearestNeighbors(metric='cosine', algorithm='brute')

    # Fit the model on the user-movie matrix
    knn.fit(user_movie_matrix_filled.values)

    # Choose a target user
    target_user_index = user_movie_matrix.index.get_loc(user_id)  # Get the row index for userId

    # Find the K nearest neighbors
    distances, indices = knn.kneighbors([user_movie_matrix_filled.iloc[target_user_index]],
                                        n_neighbors=6)  # 6 includes the target user

    # Get the indices of the neighbors (excluding the target user)
    neighbor_indices = indices.flatten()[1:]

    # Extract the ratings of the neighbors
    neighbor_ratings = user_movie_matrix_filled.iloc[neighbor_indices]

    # Compute the mean rating for each movie across neighbors
    predicted_ratings = neighbor_ratings.mean(axis=0)

    # Sort movies by predicted ratings
    recommended_movies = predicted_ratings.sort_values(ascending=False)

    # Exclude movies the target user has already rated
    already_rated = user_movie_matrix.loc[1].dropna().index  # Movies rated by userId=1
    recommended_movies = recommended_movies.drop(index=already_rated, errors='ignore')

    return recommended_movies

# Load the datasets using relative paths from the 'data' directory
ratings = pd.read_csv("data/ratings.csv", sep=",")
movies = pd.read_csv("data/movies.csv", sep=",")
links = pd.read_csv("data/links.csv", sep=",")

# Fill missing 'tmdbId' values with a placeholder (e.g., 0)
links['tmdbId'] = links['tmdbId'].fillna(0)


print(content_based(1))
print(user_based(1))


# Next step is to merge them both together.
