import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load both datasets
df_clean = pd.read_csv("cleaned_train.csv")  # Original cleaned dataset .i.e before Lasso 
df_lasso = pd.read_csv("final_lasso_feature.csv")  # Lasso-selected feature dataset

# Coursework R2 function
def r2_fn(y_true, y_pred):
    eps = y_true - y_pred
    rss = np.sum(eps ** 2)
    tss = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - (rss / tss)
    return r2


def compare_models(dataset_name):
    """Trains multiple models and compares R2 scores."""
    
    y = dataset_name['outcome']
    X = dataset_name.drop(columns=['outcome'])
    
    # Train-val 80-20 Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize X for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Define models 
    models = {
        "Linear Regression": LinearRegression(),
        "Lasso Regression": Lasso(alpha=0.05),  
        "Ridge Regression": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    results = {}

    # Train and evaluate each model
    for name, model in models.items():
        if name in ["Random Forest", "XGBoost"]:
            model.fit(X_train, y_train)  # No scaling needed for tree base models
            y_pred = model.predict(X_val)
        else:
            model.fit(X_train_scaled, y_train)  # Use scaled features
            y_pred = model.predict(X_val_scaled)

        r2 = r2_fn(y_val, y_pred)
        results[name] = r2
        print(f"{name} R2 Score: {r2:.4f}")

    # Find best model
    best_model = max(results, key=results.get)
    print(f"\n Best Model: {best_model} with R2 = {results[best_model]:.4f}")

    return results

# Run comparison on both datasets
print("\n Comparing models on Lasso-Selected Dataset:")
compare_models(df_lasso)  # Lasso-selected features

print("\n Comparing models on Original Cleaned Dataset:")
compare_models(df_clean)  # Original cleaned dataset before Lasso
