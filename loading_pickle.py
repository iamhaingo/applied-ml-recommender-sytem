import pickle


# Function to load an array from a Pickle file
def load_array_from_pickle(file_path: str):
    """
    file_path: path to the pickle file
    """
    try:
        with open(file_path, "rb") as file:
            loaded_array = pickle.load(file)
        print(f"Array loaded from {file_path} successfully.")
        return loaded_array
    except Exception as e:
        print(f"Error while loading the array: {str(e)}")
        return None


# List of file paths
file_paths = [
    "data_by_movie.pickle",
    "data_by_user.pickle",
    "user_sys_to_id.pickle",
    "user_id_to_sys.pickle",
    "movie_sys_to_id.pickle",
    "movie_id_to_sys.pickle",
    "user_matrix.pickle",
    "movie_matrix.pickle",
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
) = loaded_arrays
