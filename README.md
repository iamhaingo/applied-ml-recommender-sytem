# Recommender System with Matrix Factorization

## Overview

This repository contains Python scripts for building and evaluating a recommender system using matrix factorization. The code includes data preprocessing, model training, and visualization of the results. Three primary scripts are included:

1. `recommender_haingo.py`: The main script for training the recommender system and saving relevant data.
2. `plotting_embedding.py`: A script for loading the model and generating movie recommendations based on user preferences.
3. `plot_rmse_for_different_k.py`: A script for comparing RMSE values for different values of K (latent dimensions).

## Dependencies

To run the code, you'll need the following dependencies:

- Python 3.x
- NumPy
- Matplotlib
- Seaborn

## Usage

### Training the Recommender System

1. Run the `recommender.py` script to train the recommender system. You can choose the dataset size (small, medium, or large) by chosing the path (e.g. `BIG_PATH` for 25M).

2. The script will perform data preprocessing, train the matrix factorization model, and save relevant data as pickle files.

### Generating Movie Recommendations

1. Use the `plotting_embedding.py` script to load the trained embeddings and generate movie recommendations based on a list of movie IDs and colors. Replace the sample movie IDs and colors with your preferences.

2. The script will create a 2D plot of movie recommendations and save it as "recommendation.pdf."

### Comparing RMSE for Different K

1. Run the `plot_rmse_for_different_k.py` script to load RMSE values (pickles) for different values of K (2, 15, and 30).

2. The script will generate a line plot to compare RMSE values for different K values and save it as "multi-k.pdf."