import pickle


# Function to load an array from a Pickle file
def load_array_from_pickle(file_path: str):
    """
    file_path: path to the picklefile
    """
    try:
        with open(file_path, "rb") as file:
            loaded_array = pickle.load(file)
        print(f"Array loaded from {file_path} successfully.")
        return loaded_array
    except Exception as e:
        print(f"Error while loading the array: {str(e)}")
        return None


data_by_movie = load_array_from_pickle("data_by_movie.pickle")
data_by_user = load_array_from_pickle("data_by_user.pickle")
user_sys_to_id = load_array_from_pickle("user_sys_to_id.pickle")
user_id_to_sys = load_array_from_pickle("user_id_to_sys.pickle")
movie_sys_to_id = load_array_from_pickle("movie_sys_to_id.pickle")
movie_id_to_sys = load_array_from_pickle("movie_id_to_sys.pickle")
user_matrix = load_array_from_pickle("user_matrix.pickle")
movie_matrix = load_array_from_pickle("movie_matrix.pickle")
