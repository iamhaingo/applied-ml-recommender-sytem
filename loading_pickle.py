import pickle
import numpy as np
import matplotlib.pyplot as plt
import csv


# Function to load an array from a Pickle file
def load_array_from_pickle(file_path: str):
    """
    file_path: path to the pickle file
    """
    try:
        with open(file_path, "rb") as file:
            loaded_array = pickle.load(file)
        # print(f"Array loaded from {file_path} successfully.")
        return loaded_array
    except Exception as e:
        print(f"Error while loading the array: {str(e)}")
        return None


# List of file paths
# List of file paths for your pickles
file_paths = [
    "./pickles/data_by_movie.pickle",
    "./pickles/data_by_user.pickle",
    "./pickles/user_sys_to_id.pickle",
    "./pickles/user_id_to_sys.pickle",
    "./pickles/movie_sys_to_id.pickle",
    "./pickles/movie_id_to_sys.pickle",
    "./pickles/user_matrix.pickle",
    "./pickles/movie_matrix.pickle",
    "./pickles/user_bias.pickle",
    "./pickles/movie_bias.pickle",
]

# Load arrays from the pickle files
loaded_arrays = [load_array_from_pickle(file_path) for file_path in file_paths]

# Unpack the loaded arrays
(
    data_by_movie,
    data_by_user,
    user_sys_to_id,
    user_id_to_sys,
    movie_sys_to_id,
    movie_id_to_sys,
    user_matrix,
    movie_matrix,
    user_bias,
    movie_bias,
) = loaded_arrays

path_movie = "./ml-latest-small/movies.csv"

dict_movies = {}

with open(path_movie, "r") as data_file:
    data_reader = csv.reader(data_file, delimiter=",")
    next(data_reader, None)
    for row in data_reader:
        movie_id, title, genre = row
        dict_movies[int(movie_id)] = title


def movie_search_grep(dict_movies, search_term):
    results = {}
    for sys_id, movie_name in dict_movies.items():
        if search_term.lower() in movie_name.lower():
            results[sys_id] = movie_name
    print("Sys_id, movie")
    for sys_id, movie in results.items():
        print(sys_id, movie)


def predict(movie_id):
    """
    Returns a list of tuple (movie_id,score) of top 10 recommendations for the choosen movie

    """

    print(
        movie_id,
        movie_id_to_sys[movie_id],
        dict_movies[int(float(movie_id_to_sys[movie_id]))],
    )
    # Create a dummy user
    dummy_user_vector = movie_matrix[movie_id].copy()

    # Get the prediction of all movie vectors
    recommender = [
        (i, (dummy_user_vector @ movie_matrix[i]) + 0.05 * movie_bias[i])
        for i in range(len(movie_matrix))
    ]

    # Sort the recommender from the worse to the best
    recommender = sorted(recommender, key=lambda x: x[1])

    # Return the top 10 recommendations
    top_10 = recommender[-10:]

    recommended_movies = [(movie_id, score) for movie_id, score in top_10]
    for rec_movie_id, score in recommended_movies:
        rec_movie_name = dict_movies[int(float(movie_id_to_sys[rec_movie_id]))]
        print(rec_movie_name)


movie_search_grep(dict_movies, search_term="")
# print("\n")
# print(movie_sys_to_id)
# print(int(float(movie_id_to_sys[1])))

# predict(1)
# print("\n")

# print(movie_sys_to_id["148671.0"])

# print("\n")
# predict(7602)
