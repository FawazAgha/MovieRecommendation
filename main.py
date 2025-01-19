import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def add_to_dict(target_dict, count_dict, new_dict):
    """
    Adds the values from a new dictionary to the target dictionary, updating both the sum and count.

    Parameters:
    target_dict (dict): Dictionary storing the sum of values for each genre.
    count_dict (dict): Dictionary storing the count of ratings for each genre.
    new_dict (dict): Dictionary containing new genre ratings to be added to the target_dict.

    Returns:
    None: The dictionaries target_dict and count_dict are modified in place.
    """
    for key, value in new_dict.items():
        # Update the sum
        target_dict[key] = target_dict.get(key, 0) + value
        # Update the count
        count_dict[key] = count_dict.get(key, 0) + 1


def rating_for_recommendations(recommendations, top_n=10):
    """
    Calculates the weighted average rating for each genre based on the top N recommended movies.
    The weighting is based on the similarity score of the movies.

    Parameters:
    recommendations (pd.DataFrame): A dataframe containing movie recommendations with their weighted similarity scores.
    top_n (int): The number of top recommended movies to consider for the calculation. Default is 10.

    Returns:
    None: Displays a bar plot of the average ratings per genre.
    """
    movies_names = recommendations['title'][:top_n].values
    all_genres = set()  # Use a set to avoid duplicates automatically

    # Dictionaries to store sum and count
    value_sum = {}
    value_count = {}

    for movie_name in movies_names:
        # Find the row in the 'movies' DataFrame corresponding to the movie title
        movie_row = movies[movies['title'] == movie_name]

        if not movie_row.empty:
            # Extract the genres for the matched movie
            movie_genres = movie_row.iloc[0]['genres'].split('|')  # Access the first row and split genres
            movie_similarity = (
                round(recommendations[recommendations['title'] == movie_name]['weighted_similarity'].values[0], 3))
            all_genres.update(movie_genres)
            add_to_dict(value_sum, value_count, dict.fromkeys(movie_genres, movie_similarity))

    # Calculate average for each genre
    average_genre = {genre: 100 * value_sum[genre] / value_count[genre] for genre in value_sum}

    # Sort the genres based on the average rating (descending order)
    sorted_genre = dict(sorted(average_genre.items(), key=lambda item: item[1], reverse=True))

    # Step 5: Plot the average rating per genre
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(sorted_genre.keys()), y=list(sorted_genre.values()), hue=None, legend=False)

    # Add labels and title
    plt.title(f'Average Rating per Genre for Recommended Movies')
    plt.xlabel('Genres')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

    # Show plot
    plt.show()


def rating_per_genre(user_id):
    """
    Calculates the weighted average rating per genre for a specific user, considering how many times
    each genre has been rated by the user.

    Parameters:
    user_id (int): The ID of the user whose genre ratings are to be calculated.

    Returns:
    None: Displays a bar plot of the weighted average rating per genre.
    """
    # Filter the ratings DataFrame to only include rows where the userId matches the provided user_id
    user_ratings = ratings[ratings['userId'] == user_id]

    # If the user has not rated any movies, raise an error
    if user_ratings.empty:
        raise ValueError(f"No ratings found for userId {user_id}.")

    # Merge user ratings with the movie data to get movie titles and genres
    user_movies = pd.merge(user_ratings, movies, on='movieId')

    # Explode the 'genres' column (if it contains a list) to handle multiple genres for each movie
    user_movies = user_movies.explode('genres')  # Assuming 'genres' is already split into a list

    # If 'genres' is a string, split it into a list (this handles cases where genres are stored as a string of genres)
    user_movies['genres'] = user_movies['genres'].apply(lambda g: g.split('|') if isinstance(g, str) else g)

    # Explode again to handle cases where movies have multiple genres
    user_movies = user_movies.explode('genres')  # Explode to handle multiple genres per movie

    # Group by genres and calculate the sum and count of ratings for each genre
    genre_ratings = user_movies.groupby('genres')['rating'].agg(['sum', 'count'])

    # Calculate the weighted average of ratings per genre
    genre_ratings['weighted_avg'] = genre_ratings['sum'] / genre_ratings['count']

    # Weight the average rating by the count of ratings, so genres with more ratings are emphasized
    genre_ratings['weighted_avg'] = genre_ratings['weighted_avg'] * genre_ratings['count'] / genre_ratings[
        'count'].max()

    # Sort genres by the weighted average rating in descending order
    genre_ratings = genre_ratings.sort_values(by='weighted_avg', ascending=False)

    # Create a bar plot to visualize the weighted average ratings by genre
    plt.figure(figsize=(10, 6))
    sns.barplot(x=genre_ratings.index, y=genre_ratings['weighted_avg'])

    # Add title and labels to the plot
    plt.title(f'Average Rating per Genre for User {user_id}')
    plt.xlabel('Genres')
    plt.ylabel('Weighted Average Rating')

    # Rotate the x-axis labels to make them more readable
    plt.xticks(rotation=45, ha='right')

    # Display the plot
    plt.show()


def content_based(movie_id):
    """
    Recommends similar movies based on the content (title + genres) of a given movie using TF-IDF and cosine similarity.

    Parameters:
    movie_id (int): The ID of the movie for which similar movies are to be recommended.

    Returns:
    pd.DataFrame: A DataFrame containing the movie IDs, titles, and their cosine similarity scores.
    """
    # Split the 'genres' column into a list of genres and store it in a new column 'split_genres'
    movies['split_genres'] = movies['genres'].apply(lambda g: g.split('|') if isinstance(g, str) else g)

    # Combine the movie title and the split genres into a new column 'content' for content-based recommendations
    movies['content'] = movies['title'] + " " + movies['split_genres'].apply(lambda x: " ".join(x))

    # Initialize the TF-IDF vectorizer to convert the 'content' column into a matrix of TF-IDF features
    tfidf = TfidfVectorizer(stop_words='english')

    # Apply the TF-IDF vectorizer to the 'content' column to generate the content matrix
    content_matrix = tfidf.fit_transform(movies['content'])

    # Check if the given movie_id exists in the dataset, raise an error if not
    if movie_id not in movies['movieId'].values:
        raise ValueError("Movie ID not found in the dataset.")

    # Find the index of the movie with the given movie_id
    movie_index = movies.index[movies['movieId'] == movie_id][0]

    # Extract the TF-IDF vector of the target movie
    movie_vector = content_matrix[movie_index]

    # Calculate the cosine similarity between the target movie and all other movies
    similarity_scores = cosine_similarity(movie_vector, content_matrix).flatten()

    # Create a DataFrame with movieId, title, and similarity score for each movie
    similar_movies = pd.DataFrame({
        'movieId': movies['movieId'],
        'title': movies['title'],
        'similarity': similarity_scores
    })

    # Sort the movies based on similarity scores in descending order (most similar movies first)
    similar_movies = similar_movies.sort_values(by='similarity', ascending=False)

    # Return the sorted DataFrame of similar movies
    return similar_movies


def user_based(user_id):
    """
    Recommends movies for a user based on collaborative filtering using nearest neighbors.

    Parameters:
    user_id (int): The ID of the user for whom movie recommendations are to be generated.

    Returns:
    pd.Series: A sorted series of recommended movie titles and their predicted ratings.
    """
    # Merge the ratings dataset with the movie titles dataset to get movie names alongside ratings
    ratings_with_titles = pd.merge(ratings, movies[['movieId', 'title']], on='movieId', how='left')

    # Create a user-movie ratings matrix where each row is a user and each column is a movie title
    user_movie_matrix = ratings_with_titles.pivot_table(index='userId', columns='title', values='rating')

    # Fill missing ratings (NaN) with 0 as NearestNeighbors cannot handle NaNs
    user_movie_matrix_filled = user_movie_matrix.fillna(0)

    # Initialize the KNN model using cosine similarity as the distance metric and brute-force algorithm for computation
    knn = NearestNeighbors(metric='cosine', algorithm='brute')

    # Fit the KNN model with the user-movie matrix values (ratings data)
    knn.fit(user_movie_matrix_filled.values)

    # Find the index of the target user in the user-movie matrix
    target_user_index = user_movie_matrix.index.get_loc(user_id)

    # Get the nearest neighbors (other users similar to the target user)
    distances, indices = knn.kneighbors([user_movie_matrix_filled.iloc[target_user_index]], n_neighbors=6)

    # Extract the indices of the neighbors (exclude the target user which is the first in the list)
    neighbor_indices = indices.flatten()[1:]

    # Get the ratings data of the neighboring users
    neighbor_ratings = user_movie_matrix_filled.iloc[neighbor_indices]

    # Calculate the predicted ratings for each movie by averaging the ratings of the neighbors
    predicted_ratings = neighbor_ratings.mean(axis=0)

    # Sort the predicted ratings in descending order to recommend the highest-rated movies first
    recommended_movies = predicted_ratings.sort_values(ascending=False)

    # Identify movies that the target user has already rated
    already_rated = user_movie_matrix.loc[1].dropna().index

    # Remove the movies already rated by the target user from the recommended movies
    recommended_movies = recommended_movies.drop(index=already_rated, errors='ignore')

    # Return the sorted list of recommended movies
    return recommended_movies


def content_based_user_focus(user_id, top_n=10):
    """
    Provides movie recommendations for a user based on content-based filtering and the user's high ratings.
    The recommendations are weighted based on how highly the user rated the liked movies.

    Steps:
    1. Identify movies rated highly (4 or higher) by the user.
    2. Generate content-based recommendations for each of these liked movies.
    3. Calculate a weighted similarity score for each recommendation based on the user's rating.
    4. Sort recommendations by weighted similarity and exclude movies already rated by the user.
    5. Return the top N recommendations.

    Parameters:
    user_id (int): The ID of the user for whom movie recommendations are to be generated.
    top_n (int): The number of top recommended movies to return. Default is 10.

    Returns:
    pd.DataFrame: A DataFrame containing the top N recommended movies and their weighted similarity scores.

    Raises:
    ValueError: If the user has not rated any movies 4 or higher.
    """
    # Step 1: Find movies the user has rated highly
    user_ratings = ratings[ratings['userId'] == user_id]
    liked_movies = user_ratings[user_ratings['rating'] >= 4]  # Movies rated 4 or higher

    if liked_movies.empty:
        raise ValueError(f"User {user_id} has no highly rated movies.")

    # Step 2: Get recommendations based on each liked movie
    similar_movies_list = []
    for movie_id in liked_movies['movieId']:
        similar_movies = content_based(movie_id)  # Get content-based recommendations
        similar_movies['weight'] = liked_movies[liked_movies['movieId'] == movie_id]['rating'].values[0]  # Weight by user's rating
        similar_movies_list.append(similar_movies)

    # Combine all similar movies into one DataFrame
    combined_recommendations = pd.concat(similar_movies_list)

    # Step 3: Aggregate similarity scores
    combined_recommendations = (
        combined_recommendations.groupby('movieId', as_index=False)
        .agg({'similarity': 'mean', 'weight': 'mean'})
    )

    # Calculate a weighted similarity score
    combined_recommendations['weighted_similarity'] = (
        combined_recommendations['similarity'] * combined_recommendations['weight']
    )

    # Step 4: Merge with movie titles and sort by weighted similarity
    combined_recommendations = combined_recommendations.merge(
        movies[['movieId', 'title']], on='movieId'
    )
    combined_recommendations = combined_recommendations.sort_values(
        by='weighted_similarity', ascending=False
    )

    # Step 5: Exclude movies the user has already rated
    already_rated = user_ratings['movieId']
    final_recommendations = combined_recommendations[
        ~combined_recommendations['movieId'].isin(already_rated)
    ]

    # Return the top N recommendations
    return final_recommendations[['movieId', 'title', 'weighted_similarity']].head(top_n)

# Load the datasets using relative paths from the 'data' directory
ratings = pd.read_csv("data/ratings.csv", sep=",")
movies = pd.read_csv("data/movies.csv", sep=",")
links = pd.read_csv("data/links.csv", sep=",")


# Example
user_id = 1  # Specify the target user
top_n = 20
recommendations = content_based_user_focus(user_id,top_n)

print(recommendations)
print(user_based(user_id))
rating_per_genre(user_id)
rating_for_recommendations(recommendations,top_n)