import pickle
import seaborn as sns
import matplotlib.pyplot as plt

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


file_paths_2 = [
    "./pickles_2/loss_list_train.pickle",
    "./pickles_2/loss_list_test.pickle",
    "./pickles_2/rmse_list_train.pickle",
    "./pickles_2/rmse_list_test.pickle",
]

# Load arrays from the pickle files
loaded_arrays_2 = [load_array_from_pickle(file_path) for file_path in file_paths_2]

# Unpack the loaded arrays
(
    loss_list_train_2,
    loss_list_test_2,
    rmse_list_train_2,
    rmse_list_test_2,
) = loaded_arrays_2


file_paths_15 = [
    "./pickles_15/loss_list_train.pickle",
    "./pickles_15/loss_list_test.pickle",
    "./pickles_15/rmse_list_train.pickle",
    "./pickles_15/rmse_list_test.pickle",
]

# Load arrays from the pickle files
loaded_arrays_15 = [load_array_from_pickle(file_path) for file_path in file_paths_15]

# Unpack the loaded arrays
(
    loss_list_train_15,
    loss_list_test_15,
    rmse_list_train_15,
    rmse_list_test_15,
) = loaded_arrays_15

file_paths_30 = [
    "./pickles_30/loss_list_train.pickle",
    "./pickles_30/loss_list_test.pickle",
    "./pickles_30/rmse_list_train.pickle",
    "./pickles_30/rmse_list_test.pickle",
]

# Load arrays from the pickle files
loaded_arrays_30 = [load_array_from_pickle(file_path) for file_path in file_paths_30]

# Unpack the loaded arrays
(
    loss_list_train_30,
    loss_list_test_30,
    rmse_list_train_30,
    rmse_list_test_30,
) = loaded_arrays_30


plt.figure(figsize=(10, 6))

# Define the range of x values (assuming 10 data points)
x = range(10)

# Modern color choices
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# Data for K=2
plt.plot(x, rmse_list_train_2, label="K=2 (Train)", color=colors[0], linestyle="-")
plt.plot(x, rmse_list_test_2, label="K=2 (Test)", color=colors[0], linestyle="--")

# Data for K=15
plt.plot(x, rmse_list_train_15, label="K=15 (Train)", color=colors[1], linestyle="-")
plt.plot(x, rmse_list_test_15, label="K=15 (Test)", color=colors[1], linestyle="--")

# Data for K=30
plt.plot(x, rmse_list_train_30, label="K=30 (Train)", color=colors[2], linestyle="-")
plt.plot(x, rmse_list_test_30, label="K=30 (Test)", color=colors[2], linestyle="--")

plt.legend(loc="upper right")
plt.xlabel("Data Points")
plt.ylabel("RMSE")
plt.title("RMSE for Different Values of K")
# plt.grid(True)
plt.savefig("multi-k.pdf")
plt.show()
