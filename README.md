# Movie Recommendation System

Collaborative filtering system using matrix factorization (SVD) on MovieLens 100K dataset.

## Features
- Data downloading and preprocessing
- Exploratory data analysis (EDA)
- Model training with cross-validation
- Evaluation metrics: RMSE, MAE, Precision@5, Recall@5
- Top-N recommendations generation

## Results
![Rating Distribution](rating_distribution.png)
![Ratings per User](ratings_per_user.png)

**Evaluation Metrics:**
- Test RMSE: 0.93
- Test MAE: 0.73
- Precision@5: 0.75
- Recall@5: 0.42

## Usage

### Jupyter Notebook
```bash
jupyter notebook recommendation_system.ipynb
