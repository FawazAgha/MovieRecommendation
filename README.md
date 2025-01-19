# Movie Recommendation System
This is a movie recommendation system built using Python, utilizing content-based and user-based collaborative filtering techniques. The system provides movie recommendations to users based on their previous ratings and movie genres, as well as suggesting movies with similar content (such as title and genre).
Features
•	Content-Based Recommendations: Recommends movies based on the content (title + genres) of a given movie using TF-IDF vectorization and cosine similarity.
•	User-Based Recommendations: Provides movie recommendations based on collaborative filtering, using nearest neighbors to find users with similar tastes.
•	Genre Rating Analysis: Displays bar plots showing the average ratings per genre for a specific user or a list of recommended movies.
•	Weighted Recommendations: The system weights recommendations based on how highly the user rated movies they liked.
Requirements
•	Python 3.11
•	Pandas
•	Matplotlib
•	Seaborn
•	scikit-learn
You can install the required dependencies using pip:
bash
pip install pandas matplotlib seaborn scikit-learn
Dataset
The system requires the following datasets in CSV format:
•	ratings.csv: Contains user ratings for movies.
•	movies.csv: Contains information about movies (titles, genres).
•	links.csv: Contains additional information related to movies (optional).
These files should be stored in a folder named data.
Example of the datasets:
•	ratings.csv should have columns like userId, movieId, and rating.
•	movies.csv should have columns like movieId, title, and genres.
•	links.csv contains links between movie IDs and external movie databases (like IMDb).
How to Use
1.	Clone or download the repository.
2.	Place the dataset files (ratings.csv, movies.csv, links.csv) in the data directory.
3.	Modify the code to load the datasets from the appropriate path if needed.
4.	Run the MovieRecommendationSystem.py file.
Example Usage
To get recommendations for a specific user:
There is an example at the bottom of the page after all the functions. See:
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
Functions
add_to_dict(target_dict, count_dict, new_dict)
Adds values from a new dictionary to the target dictionary, updating the sum and count for each genre.
rating_for_recommendations(recommendations, top_n=10)
Calculates the weighted average rating for each genre based on the top N recommended movies and displays a bar plot.
rating_per_genre(user_id)
Calculates the weighted average rating per genre for a specific user and displays a bar plot.
content_based(movie_id)
Recommends similar movies based on the content (title + genres) of a given movie using TF-IDF and cosine similarity.
user_based(user_id)
Recommends movies for a user based on collaborative filtering using nearest neighbors.
content_based_user_focus(user_id, top_n=10)
Provides movie recommendations for a user based on content-based filtering and the user’s high ratings, weighted by how highly the user rated the liked movies.

![image](https://github.com/user-attachments/assets/c692febf-f68f-4f1a-96d1-15e4864094c6)
