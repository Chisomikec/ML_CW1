from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import joblib

df = pd.read_csv("cleaned_train.csv")  # Load dataset

# Define outcome and features
y = df['outcome']
X = df.drop(columns=['outcome'])

# Coursework r2 function
def r2_fn(yhat, y_actual):
    eps = y_actual - yhat
    rss = np.sum(eps ** 2)
    tss = np.sum((y_actual - y_actual.mean()) ** 2)
    return 1 - (rss / tss)

# Split Data 80% Train, 20% Validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) 

# Train XGBoost Model 
xgb = XGBRegressor(n_estimators=550, 
                    learning_rate=0.023, 
                    max_depth=3, 
                    min_child_weight=8,    
                    subsample=0.7,
                    colsample_bytree=0.7,
                    reg_alpha=5.0,  
                    reg_lambda=5.0,
                    random_state=42)


xgb.fit(X_train, y_train, 
        eval_set=[(X_val, y_val)], 
        verbose=False)

# Evaluate using coursework R2 function 
r2_train = r2_fn(xgb.predict(X_train), y_train)
r2_val = r2_fn(xgb.predict(X_val), y_val)

print(f" Training R2: {r2_train:.4f}")
print(f" Validation R2: {r2_val:.4f}")

# Save the trained XGBoost model
#joblib.dump(xgb, "final_xgboost_model2.pkl")
#print("Model saved successfully!")

