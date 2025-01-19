# Movie Recommendation System

This is a movie recommendation system built using Python, utilizing content-based and user-based collaborative filtering techniques. The system provides movie recommendations to users based on their previous ratings and movie genres, as well as suggesting movies with similar content (such as title and genre).

## Plots Generated

The system generates various plots to visually represent the results of the recommendation system, including:

1. **Genre Rating Analysis**: Bar plots showing the average ratings per genre for a specific user or a list of recommended movies.
   
2. **Weighted Genre Ratings for Recommendations**: A bar plot of the weighted average ratings for each genre based on the top N recommended movies.

These plots help users understand the distribution of ratings and the recommendations provided by the system.


## Features
- **Content-Based Recommendations**: Recommends movies based on the content (title + genres) of a given movie using TF-IDF vectorization and cosine similarity.
- **User-Based Recommendations**: Provides movie recommendations based on collaborative filtering, using nearest neighbors to find users with similar tastes.
- **Genre Rating Analysis**: Displays bar plots showing the average ratings per genre for a specific user or a list of recommended movies.
- **Weighted Recommendations**: The system weights recommendations based on how highly the user rated movies they liked.

## Requirements
To run the system, you need Python 3.11 and the following dependencies:
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

You can install the required dependencies using pip:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pip install pandas matplotlib seaborn scikit-learn

## Dataset
The system requires the following datasets in CSV format:

- `ratings.csv`: Contains user ratings for movies.
- `movies.csv`: Contains information about movies (titles, genres).
- `links.csv`: Contains additional information related to movies (optional).

These files should be stored in a folder named `data`.
### Example of the datasets:
- ratings.csv should have columns like userId, movieId, and rating.
- movies.csv should have columns like movieId, title, and genres.
- links.csv contains links between movie IDs and external movie databases (like IMDb).
  
## How to Use
1. Clone or download the repository.
2. Place the dataset files (`ratings.csv`, `movies.csv`, `links.csv`) in the `data` directory.
3. Modify the code to load the datasets from the appropriate path if needed.
4. Run the `MovieRecommendationSystem.py` file.
   
## Example Usage
There is an example after all the function defenitions in the main.py file.

It should look somethink like this:

ratings = pd.read_csv("data/ratings.csv",sep=",")

movies = pd.read_csv("data/movies.csv", sep=",")

links = pd.read_csv("data/links.csv", sep=",")



#Example

user_id = 1  # Specify the target user

top_n = 20

recommendations = content_based_user_focus(user_id,top_n)


print(recommendations)

print(user_based(user_id))

rating_per_genre(user_id)

rating_for_recommendations(recommendations,top_n)


## Functions

### `add_to_dict(target_dict, count_dict, new_dict)`
Adds values from a new dictionary to the target dictionary, updating the sum and count for each genre.

- **Parameters:**
  - `target_dict`: The target dictionary that will be updated.
  - `count_dict`: A dictionary containing the count of occurrences.
  - `new_dict`: A dictionary with new values to be added.

- **Returns:** Updated `target_dict` and `count_dict`.

---

### `rating_for_recommendations(recommendations, top_n=10)`
Calculates the weighted average rating for each genre based on the top N recommended movies and displays a bar plot.

- **Parameters:**
  - `recommendations`: List of movie recommendations.
  - `top_n`: The number of top movies to consider (default is 10).

- **Returns:** Bar plot of the weighted average ratings for each genre.

---

### `rating_per_genre(user_id)`
Calculates the weighted average rating per genre for a specific user and displays a bar plot.

- **Parameters:**
  - `user_id`: The ID of the user for whom the ratings are calculated.

- **Returns:** Bar plot of the weighted average ratings per genre for the user.

---

### `content_based(movie_id)`
Recommends similar movies based on the content (title + genres) of a given movie using TF-IDF and cosine similarity.

- **Parameters:**
  - `movie_id`: The ID of the movie to find similar ones for.

- **Returns:** List of recommended movies based on content similarity.

---

### `user_based(user_id)`
Recommends movies for a user based on collaborative filtering using nearest neighbors.

- **Parameters:**
  - `user_id`: The ID of the user to generate recommendations for.

- **Returns:** List of recommended movies based on collaborative filtering.

---

### `content_based_user_focus(user_id, top_n=10)`
Provides movie recommendations for a user based on content-based filtering and the user’s high ratings, weighted by how highly the user rated the liked movies.

- **Parameters:**
  - `user_id`: The ID of the user for whom recommendations are generated.
  - `top_n`: The number of top movies to consider (default is 10).

- **Returns:** List of movie recommendations based on the user’s ratings and content similarity.

