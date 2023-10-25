import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

start_time = time.time()

# Set the Seaborn style
sns.set(style="whitegrid")
sns.set_style("ticks")
sns.color_palette("Set1")

# File paths
SMALL_UDATA_PATH = "/home/haingo/Documents/python-stuff/ml-100k/u.data"
SMALL_PATH = "/home/haingo/Documents/python-stuff/ml-latest-small/ratings.csv"
BIG_PATH = "/home/haingo/Documents/python-stuff/ml-25m/ratings.csv"


def read_data(file_path: str, file_type: str):
    """
    Read data from a CSV file or a .data file and return the extracted data.

    file_path (str): The path to the input file.
    file_type (str): The type of file ('csv' or 'data').

    Returns: A list containing the extracted data.
    """
    if file_type == "csv":
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)[:, :3].astype(float)
    elif file_type == "data":
        data = []
        with open(file_path, "r", encoding="utf-8") as data_file:
            data_reader = csv.reader(data_file, delimiter="\t")
            for row in data_reader:
                system_user_id, system_movie_id, system_rating, _ = row
                data.append(
                    [int(system_user_id), int(system_movie_id), float(system_rating)]
                )
    else:
        raise ValueError("Unsupported file type. Use 'csv' or 'data'.")
    return data


def partitioning_data(data):
    """
    Parsing the csv data and partitioning it into training and test set

    data: the rating data in rating csv
    """
    # Shuffling and splitting point
    np.random.shuffle(data)

    split_point = 0.9 * len(data)

    # Initialization
    user_sys_to_id = {}
    user_id_to_sys = []
    movie_sys_to_id = {}
    movie_id_to_sys = []

    # First build the mappings.
    for index in range(len(data)):
        user_sys = data[index][0]
        movie_sys = data[index][1]

        # Take care of the user data structure
        if user_sys not in user_sys_to_id:
            user_id_to_sys.append(user_sys)
            user_sys_to_id[user_sys] = len(user_sys_to_id)

        # Take care of the movie data structure
        if movie_sys not in movie_sys_to_id:
            movie_id_to_sys.append(movie_sys)
            movie_sys_to_id[movie_sys] = len(movie_sys_to_id)

    # Initialize with empty list all the trainings data.
    data_by_user_train = [[] for i in range(len(user_id_to_sys))]
    data_by_movie_train = [[] for i in range(len(movie_id_to_sys))]

    # Initialize with empty list all the test data.
    data_by_user_test = [[] for i in range(len(user_id_to_sys))]
    data_by_movie_test = [[] for i in range(len(movie_id_to_sys))]

    # Create all the data structure using in a loop
    for index in range(len(data)):
        user_sys = data[index][0]
        movie_sys = data[index][1]
        rating = data[index][2]

        user_index = user_sys_to_id[user_sys]
        movie_index = movie_sys_to_id[movie_sys]

        if index < split_point:
            # Insert into the sparse user and item *training* matrices.
            data_by_user_train[user_index].append((movie_index, float(rating)))
            data_by_movie_train[movie_index].append((user_index, float(rating)))

        else:
            # Insert into the sparse user and item *test* matrices.
            data_by_user_test[user_index].append((movie_index, float(rating)))
            data_by_movie_test[movie_index].append((user_index, float(rating)))

    return (
        user_sys_to_id,
        user_id_to_sys,
        movie_sys_to_id,
        movie_id_to_sys,
        data_by_user_train,
        data_by_movie_train,
        data_by_user_test,
        data_by_movie_test,
    )


def plot_power_law(data_by_user, data_by_movie, filename=None):
    """
    Plot the power_law
    """
    # Calculate the number of movies rated per user and users rating each movie
    num_rating_user = [len(a) for a in data_by_user]
    num_rating_movie = [len(a) for a in data_by_movie]

    # Create a scatter plot
    _, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(
        num_rating_user,
        [num_rating_user.count(i) for i in num_rating_user],
        marker=".",
        label="User",
    )
    ax.scatter(
        num_rating_movie,
        [num_rating_movie.count(i) for i in num_rating_movie],
        marker=".",
        label="Movie",
    )

    # Set the y and x scales to logarithmic
    ax.set_yscale("log")
    ax.set_xscale("log")

    # Add labels and legend
    ax.legend(["Users", "Movies"])

    # Add grid lines
    ax.grid(True, linestyle="--", alpha=0.5)

    # Add a title
    ax.set_title("Distribution of Ratings per User and Movie", fontsize=16)

    # Label x and y axes
    ax.set_xlabel("Number of Ratings", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)

    # Customize tick labels
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: int(y)))

    # Increase font size for labels and legend
    ax.tick_params(axis="both", which="both", labelsize=12)
    ax.legend(fontsize=12)

    # Show the plot
    plt.tight_layout()
    if filename is not None:
        plt.savefig(f"{filename}.pdf")
    plt.show()


# Read data
# small_data = read_data(SMALL_UDATA_PATH, "data")
# small_data = read_data(SMALL_PATH, "csv")
small_data = read_data(BIG_PATH, "csv")

# Partitioning the data for training and testing
(
    user_sys_to_id,
    user_id_to_sys,
    movie_sys_to_id,
    movie_id_to_sys,
    data_by_user_train,
    data_by_movie_train,
    data_by_user_test,
    data_by_movie_test,
) = partitioning_data(small_data)

# Initialization
LAMBDA, TAU, GAMMA = 0.005, 0.05, 0.4

LATENT_DIMS = 15
NUM_ITERATIONS = 10

sigma = np.sqrt(1 / np.sqrt(LATENT_DIMS))

NUM_USERS = len(data_by_user_train)
NUM_MOVIES = len(data_by_movie_train)

user_matrix = np.random.normal(0, sigma, size=(NUM_USERS, LATENT_DIMS))
movie_matrix = np.random.normal(0, sigma, size=(NUM_MOVIES, LATENT_DIMS))
user_bias = np.zeros(NUM_USERS)
movie_bias = np.zeros(NUM_MOVIES)

loss_list_train, rmse_list_train, loss_list_test, rmse_list_test = [], [], [], []


def update_bias_user(user_id, data_by_user):
    """
    data_by_user: sparse matrix by user
    Returns: user_bias
    """
    bias = 0
    movie_counter = 0
    for movie_id, rating in data_by_user[user_id]:
        bias += LAMBDA * (
            rating
            - (
                np.matmul(user_matrix[user_id].T, movie_matrix[movie_id])
                + movie_bias[movie_id]
            )
        )
        movie_counter += 1
    bias = bias / (LAMBDA * movie_counter + GAMMA)

    return bias


def update_bias_movie(movie_id, data_by_movie):
    """
    data_by_movie: sparse matrix by movie
    Returns: user_movie
    """
    bias = 0
    user_counter = 0
    for user_id, rating in data_by_movie[movie_id]:
        bias += LAMBDA * (
            rating
            - (
                np.matmul(movie_matrix[movie_id].T, user_matrix[user_id])
                + user_bias[user_id]
            )
        )
        user_counter += 1
    bias = bias / (LAMBDA * user_counter + GAMMA)

    return bias


def update_matrix_user(user_id, data_by_user):
    """
    data_by_user: sparse matrix by user
    Returns: user_matrix
    """
    if not data_by_user[user_id]:
        return user_matrix[user_id]

    right_summation = 0
    left_summation = 0
    for movie_id, rating in data_by_user[user_id]:
        left_summation += LAMBDA * np.outer(
            movie_matrix[movie_id], movie_matrix[movie_id]
        )
        right_summation += (
            LAMBDA
            * movie_matrix[movie_id]
            * (rating - user_bias[user_id] - movie_bias[movie_id])
        )

    left_summation += TAU * np.eye(LATENT_DIMS)

    left_term = np.linalg.inv(left_summation)

    return np.matmul(left_term, right_summation)


def update_matrix_movie(movie_id, data_by_movie):
    """
    data_by_movie: sparse matrix by movie
    Returns: movie_matrix
    """
    if not data_by_movie[movie_id]:
        return movie_matrix[movie_id]

    right_summation = 0
    left_summation = 0
    for user_id, rating in data_by_movie[movie_id]:
        left_summation += LAMBDA * np.outer(user_matrix[user_id], user_matrix[user_id])
        right_summation += (
            LAMBDA
            * user_matrix[user_id]
            * (rating - movie_bias[movie_id] - user_bias[user_id])
        )

    left_summation += TAU * np.eye(LATENT_DIMS)

    left_term = np.linalg.inv(left_summation)

    return np.matmul(left_term, right_summation)


def calculate_loss(data_by_user):
    """
    data_by_user: sparse matrix by user
    Returns: loss
    """
    # Initialisation
    summation = 0.0
    # Loss
    for user_id in range(NUM_USERS):
        for movie_id, rating in data_by_user[user_id]:
            predicted_rating = (
                np.dot(user_matrix[user_id].T, movie_matrix[movie_id])
                + user_bias[user_id]
                + movie_bias[movie_id]
            )
            prediction_error = rating - predicted_rating
            summation += prediction_error**2

    #  Regularization terms
    user_bias_reg_term = 0.5 * GAMMA * np.dot(user_bias.T, user_bias)
    movie_bias_reg_term = 0.5 * GAMMA * np.dot(movie_bias.T, movie_bias)

    user_matrix_reg_term = 0
    for user_id in range(NUM_USERS):
        user_matrix_reg_term += np.matmul(user_matrix[user_id], user_matrix[user_id])
    user_matrix_reg_term *= 0.5 * TAU

    movie_matrix_reg_term = 0
    for movie_id in range(NUM_MOVIES):
        movie_matrix_reg_term += np.matmul(
            movie_matrix[movie_id], movie_matrix[movie_id]
        )
    movie_matrix_reg_term *= 0.5 * TAU

    final_loss = (-0.5) * summation * LAMBDA - (
        user_bias_reg_term
        + movie_bias_reg_term
        + user_matrix_reg_term
        + movie_matrix_reg_term
    )

    return final_loss


def calculate_rmse(data_by_user):
    """
    data_by_user: sparse matrix by user
    Returns: rmse
    """
    diff = 0
    num_rating = 0
    for user_id in range(NUM_USERS):
        for movie_id, rating in data_by_user[user_id]:
            diff += (
                rating
                - (
                    user_matrix[user_id].T @ movie_matrix[movie_id]
                    + user_bias[user_id]
                    + movie_bias[movie_id]
                )
            ) ** 2
            num_rating += 1

    mse = diff / num_rating
    return np.sqrt(mse)


def plot_metric(metric_type: str):
    """
    Plot either log-likelihood or RMSE based on the metric_type argument.

    metric_type (str): Either 'log-likelihood' or 'rmse'.
    """
    if metric_type == "log-likelihood":
        metric_list_train = loss_list_train
        metric_list_test = loss_list_test
        metric_label = "Log-likelihood"
        filename = "log-likelihood.pdf"
    elif metric_type == "rmse":
        metric_list_train = rmse_list_train
        metric_list_test = rmse_list_test
        metric_label = "RMSE"
        filename = "rmse.pdf"
    else:
        print("Invalid metric_type. Use 'log-likelihood' or 'rmse'.")
        return

    # Set the figure size
    plt.figure(figsize=(10, 10))  # Adjust the width and height as needed
    # Plot training metric in medium violet red
    plt.plot(range(NUM_ITERATIONS), metric_list_train, label="Training")

    # Plot test metric in blue
    plt.plot(range(NUM_ITERATIONS), metric_list_test, label="Test")

    plt.xlabel("Iteration")
    plt.ylabel(metric_label)
    plt.title(metric_label)

    # Show a legend to distinguish lines
    plt.legend()

    plt.grid(True, linestyle="--", alpha=0.3)

    # Save the plot
    plt.savefig(filename)

    # Display the plot
    plt.show()


def plot_embedding(filename: str):
    """
    Plot the embedding
    movie_matrix: trained movie matrix
    user_matrix: trained user matrix
    """
    V_1 = [item[0] for item in movie_matrix]
    V_2 = [item[1] for item in movie_matrix]
    U_1 = [item[0] for item in user_matrix]
    U_2 = [item[1] for item in user_matrix]

    fig, ax = plt.subplots()
    ax.scatter(V_1, V_2, c="b", alpha=0.5, label="Movie")
    ax.scatter(U_1, U_2, c="g", alpha=0.5, label="User")

    ax.legend()
    plt.savefig(filename)

    plt.show()


# Training
for i in range(NUM_ITERATIONS):
    movie_bias = np.array(
        [
            update_bias_movie(movie_id, data_by_movie_train)
            for movie_id in range(NUM_MOVIES)
        ]
    )
    movie_matrix = np.array(
        [
            update_matrix_movie(movie_id, data_by_movie_train)
            for movie_id in range(NUM_MOVIES)
        ]
    )
    user_bias = np.array(
        [update_bias_user(user_id, data_by_user_train) for user_id in range(NUM_USERS)]
    )
    user_matrix = np.array(
        [
            update_matrix_user(user_id, data_by_user_train)
            for user_id in range(NUM_USERS)
        ]
    )

    loss = calculate_loss(data_by_user_train)
    rmse = calculate_rmse(data_by_user_train)
    print(f"Iteration: {i}, rmse train: {rmse}")

    loss_list_train.append(loss)
    rmse_list_train.append(rmse)

    loss = calculate_loss(data_by_user_test)
    rmse = calculate_rmse(data_by_user_test)
    print(f"Iteration: {i}, rmse test: {rmse}")

    loss_list_test.append(loss)
    rmse_list_test.append(rmse)


# Plot and save the power law figure.
# plot_power_law(data_by_user_train, data_by_movie_train, "power_law_train")
# plot_power_law(data_by_user_test, data_by_movie_test, "power_law_test")

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.5f} seconds")


# Function to save an array to a file using Pickle
def save_array_to_pickle(array, file_path: str):
    """
    array: array to save
    file_path: where to save the array on the disk
    """
    try:
        with open(file_path, "wb") as file:
            pickle.dump(array, file)
        print(f"Array saved to {file_path} successfully.")
    except Exception as e:
        print(f"Error while saving the array: {str(e)}")


data_to_save = [
    (data_by_movie_train, "./pickles/data_by_movie.pickle"),
    (data_by_user_train, "./pickles/data_by_user.pickle"),
    (user_sys_to_id, "./pickles/user_sys_to_id.pickle"),
    (user_id_to_sys, "./pickles/user_id_to_sys.pickle"),
    (movie_sys_to_id, "./pickles/movie_sys_to_id.pickle"),
    (movie_id_to_sys, "./pickles/movie_id_to_sys.pickle"),
    (user_matrix, "./pickles/user_matrix.pickle"),
    (movie_matrix, "./pickles/movie_matrix.pickle"),
    (user_bias, "./pickles/user_bias.pickle"),
    (movie_bias, "./pickles/movie_bias.pickle"),
]


# Saving in pickle
for data, filename in data_to_save:
    save_array_to_pickle(data, filename)


plot_metric("log-likelihood")
plot_metric("rmse")
plot_embedding("Embedding.pdf")
