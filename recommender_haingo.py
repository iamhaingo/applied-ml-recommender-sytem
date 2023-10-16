import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the Seaborn style
sns.set(style="whitegrid")
sns.set_style("ticks")
sns.color_palette("Set1")

# Reading the small data
# FILE_PATH = "/home/haingo/Documents/python-stuff/ml-latest-small/ratings.csv"
# small_data = np.loadtxt(FILE_PATH, delimiter=",", skiprows=1)[:, :3].astype(str)

UDATA_PATH = "/home/haingo/Documents/python-stuff/ml-100k/u.data"

small_data = []

with open(UDATA_PATH, "r") as data_file:
    data_reader = csv.reader(data_file, delimiter="\t")

    for row in data_reader:
        user_id, movie_id, rating, timestamp = row

        small_data.append([user_id, movie_id, float(rating)])


def partitioning_data(data):
    """
    Parsing the csv data and partitioning it into training and test set
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


# Partitioning the data for training and testing
(
    _,
    _,
    _,
    _,
    data_by_user_train,
    data_by_movie_train,
    data_by_user_test,
    data_by_movie_test,
) = partitioning_data(small_data)


# Initialization
LAMBDA, TAU, GAMMA = 0.001, 0.01, 0.5

LATENT_DIMS = 5
NUM_ITERATIONS = 15

sigma = np.sqrt(1 / np.sqrt(LATENT_DIMS))

NUM_USERS = len(data_by_user_train)
NUM_MOVIES = len(data_by_movie_train)

user_matrix = np.random.normal(0, sigma, size=(NUM_USERS, LATENT_DIMS))
movie_matrix = np.random.normal(0, sigma, size=(NUM_MOVIES, LATENT_DIMS))
user_bias = np.zeros(NUM_USERS)
movie_bias = np.zeros(NUM_MOVIES)

loss_list_train, rmse_list_train, loss_list_test, rmse_list_test = [], [], [], []


def update_bias_user(data_by_user):
    """
    data_by_user: sparse matrix by user
    Returns: user_bias
    """
    for user_id in range(NUM_USERS):
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
        user_bias[user_id] = bias
    return user_bias


def update_bias_movie(data_by_movie):
    """
    data_by_movie: sparse matrix by movie
    Returns: user_movie
    """
    for movie_id in range(NUM_MOVIES):
        bias = 0
        user_counter = 0
        for user_id, rating in data_by_movie[movie_id]:
            bias += LAMBDA * (
                rating
                - (
                    np.matmul(movie_matrix[user_id].T, user_matrix[user_id])
                    + user_bias[user_id]
                )
            )
            user_counter += 1
        bias = bias / (LAMBDA * user_counter + GAMMA)
        movie_bias[movie_id] = bias
    return movie_bias


def update_matrix_user(data_by_user):
    """
    data_by_user: sparse matrix by user
    Returns: user_matrix
    """
    for user_id in range(NUM_USERS):
        if not data_by_user[user_id]:
            continue

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

        user_matrix[user_id] = np.matmul(left_term, right_summation)

    return user_matrix


def update_matrix_movie(data_by_movie):
    """
    data_by_movie: sparse matrix by movie
    Returns: movie_matrix
    """
    for movie_id in range(NUM_MOVIES):
        if not data_by_movie[movie_id]:
            continue

        right_summation = 0
        left_summation = 0
        for user_id, rating in data_by_movie[movie_id]:
            left_summation += LAMBDA * np.outer(
                user_matrix[user_id], user_matrix[user_id]
            )
            right_summation += (
                LAMBDA
                * user_matrix[user_id]
                * (rating - movie_bias[movie_id] - user_bias[user_id])
            )

        left_summation += TAU * np.eye(LATENT_DIMS)

        left_term = np.linalg.inv(left_summation)

        movie_matrix[movie_id] = np.matmul(left_term, right_summation)

    return movie_matrix


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
    user_bias_reg_term = 0.5 * GAMMA * np.sum(user_bias**2)
    movie_bias_reg_term = 0.5 * GAMMA * np.sum(movie_bias**2)

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


def plot_loss():
    """
    Ploting the loss of the test and the training set
    """
    # Set the figure size
    plt.figure(figsize=(10, 10))  # Adjust the width and height as needed
    # Plot training loss in medium violet red
    plt.plot(range(NUM_ITERATIONS), loss_list_train, label="Training")

    # Plot test loss in blue
    plt.plot(range(NUM_ITERATIONS), loss_list_test, label="Test")

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss")

    # Show a legend to distinguish lines
    plt.legend()

    plt.grid(True, linestyle="--", alpha=0.3)

    # Save the plot as 'loss.pdf'
    plt.savefig("loss.pdf")

    # Display the plot
    plt.show()


def plot_rmse():
    """
    Ploting the rmse of the test and the training set
    """
    # Set the figure size
    plt.figure(figsize=(10, 10))  # Adjust the width and height as needed

    # Plot training RMSE in medium violet red
    plt.plot(range(NUM_ITERATIONS), rmse_list_train, label="Training")

    # Plot test RMSE in blue
    plt.plot(range(NUM_ITERATIONS), rmse_list_test, label="Test")

    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title("Root Mean Square Error")

    # Show a legend to distinguish lines
    plt.legend()

    plt.grid(True, linestyle="--", alpha=0.3)

    # Save the plot as 'rmse.pdf'
    plt.savefig("rmse.pdf")

    # Display the plot
    plt.show()


# Training
for i in range(NUM_ITERATIONS):
    movie_bias = update_bias_movie(data_by_movie_train)
    movie_matrix = update_matrix_movie(data_by_movie_train)
    user_bias = update_bias_user(data_by_user_train)
    user_matrix = update_matrix_user(data_by_user_train)

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

# Call the function with your data
plot_loss()
plot_rmse()
