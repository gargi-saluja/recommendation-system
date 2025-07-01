import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate
from collections import defaultdict
import os
import requests
from zipfile import ZipFile
import joblib

def download_dataset():
    """Download and extract MovieLens 100K dataset"""
    DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    DATA_DIR = "ml-100k"
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        print("Downloading dataset...")
        r = requests.get(DATASET_URL)
        with open("ml-100k.zip", "wb") as f:
            f.write(r.content)
        
        with ZipFile("ml-100k.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        print("Dataset downloaded and extracted")
    
    return DATA_DIR

def load_data(data_dir):
    """Load ratings and movie data"""
    ratings = pd.read_csv(
        os.path.join(data_dir, "u.data"),
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    
    movies = pd.read_csv(
        os.path.join(data_dir, "u.item"),
        sep="|",
        encoding="latin-1",
        names=["item_id", "title", "release_date", "video_release", "imdb_url", 
               "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
               "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
               "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    )[["item_id", "title"]]
    
    return ratings, movies

def eda(ratings):
    """Perform exploratory data analysis"""
    # Rating distribution
    plt.figure(figsize=(10, 5))
    sns.countplot(x='rating', data=ratings, palette='viridis')
    plt.title('Rating Distribution')
    plt.savefig('rating_distribution.png')
    plt.close()
    
    # Ratings per user
    user_ratings = ratings.groupby('user_id').size()
    plt.figure(figsize=(10, 5))
    sns.histplot(user_ratings, bins=50, kde=True)
    plt.title('Ratings per User')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Count')
    plt.savefig('ratings_per_user.png')
    plt.close()
    
    # Ratings per movie
    movie_ratings = ratings.groupby('item_id').size()
    plt.figure(figsize=(10, 5))
    sns.histplot(movie_ratings, bins=50, kde=True)
    plt.title('Ratings per Movie')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Count')
    plt.savefig('ratings_per_movie.png')
    plt.close()

def train_model(ratings):
    """Train and evaluate recommendation model"""
    # Configure dataset
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[["user_id", "item_id", "rating"]], reader)
    
    # Split data
    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
    
    # Initialize and train SVD model
    model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    model.fit(trainset)
    
    # Cross-validation
    cv_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    
    # Test set evaluation
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    
    # Precision and Recall
    precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=4)
    precision_score = np.mean(list(precisions.values()))
    recall_score = np.mean(list(recalls.values()))
    
    return model, {
        "cv_rmse": np.mean(cv_results['test_rmse']),
        "cv_mae": np.mean(cv_results['test_mae']),
        "test_rmse": rmse,
        "test_mae": mae,
        "precision@5": precision_score,
        "recall@5": recall_score
    }

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Calculate precision and recall at k"""
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    
    precisions = {}
    recalls = {}
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
    
    return precisions, recalls

def get_top_n_recommendations(model, user_id, movies, ratings, n=10):
    """Generate top N recommendations for a user"""
    all_movie_ids = ratings['item_id'].unique()
    rated_movies = ratings[ratings['user_id'] == user_id]['item_id'].values
    
    predictions = []
    for movie_id in all_movie_ids:
        if movie_id not in rated_movies:
            pred = model.predict(user_id, movie_id)
            predictions.append((movie_id, pred.est))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n = predictions[:n]
    
    top_n_df = pd.DataFrame(top_n, columns=['item_id', 'estimated_rating'])
    top_n_df = pd.merge(top_n_df, movies, on='item_id')
    
    return top_n_df

def main():
    # Download and load data
    data_dir = download_dataset()
    ratings, movies = load_data(data_dir)
    
    # Perform EDA
    print("\nPerforming EDA...")
    eda(ratings)
    
    # Train and evaluate model
    print("\nTraining model...")
    model, metrics = train_model(ratings)
    
    # Print metrics
    print("\nModel Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Generate recommendations
    user_id = 196
    print(f"\nTop 10 Recommendations for User {user_id}:")
    recommendations = get_top_n_recommendations(model, user_id, movies, ratings)
    print(recommendations)
    
    # Save model
    joblib.dump(model, 'movie_recommender.pkl')
    print("\nModel saved as movie_recommender.pkl")

if __name__ == "__main__":
    main()
