import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import csv

# Set the Seaborn style
sns.set(style="whitegrid", font_scale=1.1)
sns.set_style("ticks")
sns.color_palette("Set1")
sns.set_context("talk")


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

# path_movie = "./ml-latest-small/movies.csv"
path_movie = "./ml-25m/movies.csv"

dict_movies = {}

with open(path_movie, "r") as data_file:
    data_reader = csv.reader(data_file, delimiter=",")
    next(data_reader, None)
    for row in data_reader:
        system_id, title, genre = row
        dict_movies[int(system_id)] = title


def movie_search_grep(search_term):
    results = []
    for sys_id, movie_name in dict_movies.items():
        if search_term.lower() in movie_name.lower():
            results.append((sys_id, movie_name))
    return results


def predict(movie_id):
    # Create a dummy user
    dummy_user_vector = movie_matrix[movie_id].copy()

    # Get the prediction of all movie vectors
    recommender = [
        (i, (dummy_user_vector @ movie_matrix[i]) + 0.05 * movie_bias[i])
        for i in range(len(movie_matrix))
    ]

    # Sort the recommender from the worse to the best
    recommender = sorted(recommender, key=lambda x: x[1])

    # Return the top x recommendations
    topx = recommender[-3:]

    recommended_movies = [
        (rec_movie_id, dict_movies[movie_id_to_sys[rec_movie_id]])
        for rec_movie_id, score in topx
    ]
    rec_id, rec_name = zip(*recommended_movies)

    return rec_id, rec_name


def extract_coordinates(rec_ids):
    x = []
    y = []
    for i in rec_ids:
        x.append(movie_matrix[i][0])
        y.append(movie_matrix[i][4])
    return x, y


def plot_recommendations(movie_ids_and_colors):
    plt.figure(
        figsize=(6, 6),
    )  # Increase the figure size

    for movie_id, color in movie_ids_and_colors:
        top = predict(movie_sys_to_id[movie_id])
        rec_id, rec_name = top
        x_test, y_test = extract_coordinates(rec_id)

        # Assign a unique color to each set of recommendations
        plt.scatter(
            x_test,
            y_test,
            marker="o",
            label=f"Recommendations for Movie {dict_movies[movie_id]}",
            c=[color],
        )

        # Annotate each point with the movie title with some offset to prevent overlap
        text_offset = 0.04  # Adjust this value as needed
        for i, txt in enumerate(rec_name):
            plt.annotate(
                txt, (x_test[i] - text_offset, y_test[i] + text_offset), fontsize=8
            )

    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.title("Movie Recommendations")
    plt.legend()
    plt.grid(True)

    # Adjust the plot margins to ensure text is not cut off
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08)

    # Save the plot to a PDF file
    plt.savefig("recommendation.pdf", format="pdf")

    plt.show()


# You can call this function with a list of movie IDs and corresponding colors.
movie_ids_and_colors = [
    (6350, "b"),
    # (121231, "g"),
    (364, "r"),
    (4246, "k"),
]  # Replace with movie IDs and colors as needed.

plot_recommendations(movie_ids_and_colors)
