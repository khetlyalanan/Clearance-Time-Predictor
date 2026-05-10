import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

import joblib

# ==============================================
# PROGRAM START
# ==============================================
print("\n" + "=" * 70)
print(" STUDENT CLEARANCE PERFORMANCE PREDICTOR ")
print("=" * 70)

start_time = time.time()

# ==============================================
# 1. LOAD DATA
# ==============================================
print("\n[1] LOADING DATA...")

try:
    df = pd.read_csv("StudentFiles.csv")

    print("✅ CSV FILE LOADED SUCCESSFULLY!")
    print(f"📌 Rows    : {df.shape[0]}")
    print(f"📌 Columns : {df.shape[1]}")

    print("\n📋 FIRST 5 ROWS:")
    print(df.head())

except Exception as e:
    print("❌ ERROR LOADING CSV FILE")
    print(e)
    exit()

# ==============================================
# 2. PREPROCESSING
# ==============================================
print("\n[2] PREPROCESSING DATA...")

course_map = {'BSIT': 1, 'BSit': 1}
year_map = {'1st': 1, '2nd': 2, '3rd': 3, '4th': 4}
sem_map = {'1st': 1, '2nd': 2}

df['Course'] = df['Course'].map(course_map)
df['Year_Level'] = df['Year_Level'].map(year_map)
df['Semester'] = df['Semester'].map(sem_map)

df = df.fillna(df.mean(numeric_only=True))

print("✅ CATEGORICAL DATA ENCODED")
print("✅ MISSING VALUES HANDLED")

print("\n📊 DATA TYPES:")
print(df.dtypes)

# ==============================================
# 3. CLASSIFICATION
# ==============================================
print("\n" + "=" * 70)
print(" CLASSIFICATION MODEL ")
print("=" * 70)

try:
    X_class = df[
        [
            'Course',
            'Year_Level',
            'Semester',
            'Department_Count',
            'Report_Mat',
            'Prev.Avg_Approval'
        ]
    ]

    y_class = df['Payment_Status']

    print("✅ Features and Target Prepared")

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_class,
        y_class,
        test_size=0.2,
        random_state=42,
        stratify=y_class
    )

    print("✅ Data Split Completed")

    class_model = LogisticRegression(class_weight="balanced", max_iter=2000)

    print("⏳ Training Classification Model...")
    class_model.fit(X_train_c, y_train_c)

    print("✅ Classification Model Trained")

    y_pred_c = class_model.predict(X_test_c)

    accuracy = accuracy_score(y_test_c, y_pred_c)

    print(f"\n🎯 ACCURACY SCORE: {accuracy:.4f}")

    print("\n📊 CONFUSION MATRIX:")
    print(confusion_matrix(y_test_c, y_pred_c))

    print("\n📋 CLASSIFICATION REPORT:")
    print(classification_report(y_test_c, y_pred_c))

    joblib.dump(class_model, "classification_model.pkl")

    print("\n✅ Classification Model Saved!")

except Exception as e:
    print("\n❌ ERROR IN CLASSIFICATION")
    print(e)

# ==============================================
# 4. REGRESSION
# ==============================================
print("\n" + "=" * 70)
print(" REGRESSION MODEL ")
print("=" * 70)

try:
    X_reg = df[
        [
            'Course',
            'Year_Level',
            'Semester',
            'Payment_Status',
            'Department_Count',
            'Report_Mat',
            'Prev.Avg_Approval'
        ]
    ]

    y_reg = df['Total_Days']

    print("✅ Features and Target Prepared")

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg,
        y_reg,
        test_size=0.2,
        random_state=42
    )

    print("✅ Data Split Completed")

    reg_model = LinearRegression()

    print("⏳ Training Regression Model...")
    reg_model.fit(X_train_r, y_train_r)

    print("✅ Regression Model Trained")

    y_pred_r = reg_model.predict(X_test_r)

    mae = mean_absolute_error(y_test_r, y_pred_r)
    mse = mean_squared_error(y_test_r, y_pred_r)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_r, y_pred_r)

    print("\n📈 REGRESSION RESULTS")
    print(f"✅ MAE       : {mae:.4f}")
    print(f"✅ MSE       : {mse:.4f}")
    print(f"✅ RMSE      : {rmse:.4f}")
    print(f"✅ R² SCORE  : {r2:.4f}")

    joblib.dump(reg_model, "regression_model.pkl")

    print("\n✅ Regression Model Saved!")

except Exception as e:
    print("\n❌ ERROR IN REGRESSION")
    print(e)

# ==============================================
# 5. PROGRAM END
# ==============================================
end_time = time.time()

print("\n" + "=" * 70)
print(" ✅ ALL TASKS COMPLETED SUCCESSFULLY! ")
print("=" * 70)

print(f"\n⏱ Total Execution Time: {end_time - start_time:.2f} seconds")