#%% md
# Step 1 : Importing data from Data Warehouse (parquet files)
#%%

import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, expr

from pyspark.sql import functions as F
from pyspark.sql.functions import col, count, sum, when, datediff, current_date

# Initialize Spark session
spark = SparkSession.builder \
    .appName("FHIR Data Pipeline") \
    .getOrCreate()


#Reading from parquet files
file_path = sys.argv[1]
patient_df= spark.read.parquet(f"{file_path}/patient")
claim_df = spark.read.parquet(f"{file_path}/claim")
encounter_with_participant_df = spark.read.parquet(f"{file_path}/encounter")
immunization_df = spark.read.parquet(f"{file_path}/immunization")
procedure_df = spark.read.parquet(f"{file_path}/procedure")
observation_df = spark.read.parquet(f"{file_path}/observation")
diagnostic_report_df = spark.read.parquet(f"{file_path}/diagnostic_report")
condition_df = spark.read.parquet(f"{file_path}/condition")
#%% md
# Step 2: Defining new dataframes for the model 
#%%


# Step 1: Prepare patient and chronic conditions data

from pyspark.sql.functions import col, lit

# Check for "chronic" conditions based on `condition_code_display`
chronic_condition_df = condition_df.filter(
    col("condition_code_display").rlike("(?i)chronic")
)

# Aggregate to create `has_chronic_conditions_df`
has_chronic_conditions_df = chronic_condition_df.groupBy("patient_reference").agg(
    lit(1).alias("has_chronic_conditions")
)

has_chronic_conditions_df_alias = has_chronic_conditions_df.alias("hc")
patient_df_alias = patient_df.alias("p")

# Join `patient_df` with `has_chronic_conditions_df`, carefully selecting columns
patient_df = patient_df_alias.join(
    has_chronic_conditions_df_alias.select(
        col("patient_reference").alias("hc_patient_reference"),
        col("has_chronic_conditions").alias("hc_has_chronic_conditions")
    ),
    F.col("p.patient_id") == F.col("hc_patient_reference"),
    how="left"
).drop("hc_patient_reference")  # Drop duplicate column after join

# Rename `hc_has_chronic_conditions` to `has_chronic_conditions` to unify naming
patient_df = patient_df.withColumnRenamed("hc_has_chronic_conditions", "has_chronic_conditions")

# Fill missing values for `has_chronic_conditions`
patient_df = patient_df.fillna({"has_chronic_conditions": 0})

# Step 2: Aggregate healthcare costs and join
total_cost_df_alias = claim_df.groupBy("patient_reference").agg(
    F.sum(col("total_amount").cast("float")).alias("total_healthcare_cost")
).alias("tc")

patient_df = patient_df.join(
    total_cost_df_alias.select(
        col("patient_reference").alias("tc_patient_reference"),
        col("total_healthcare_cost")
    ),
    F.col("p.patient_id") == F.col("tc_patient_reference"),
    how="left"
).drop("tc_patient_reference")  # Drop duplicate column after join

# Fill missing values for `total_healthcare_cost`
patient_df = patient_df.fillna({"total_healthcare_cost": 0.0})

# Step 3: Add derived features
patient_df = patient_df.withColumn(
    "age", (datediff(current_date(), col("birth_date").cast("date")) / 365.25).cast("int")
).withColumn(
    "is_senior", when(col("age") >= 65, 1).otherwise(0)
).withColumn(
    "region_risk", when(col("state").isin("CA", "NY"), 0.8).otherwise(0.4)
).withColumn(
    "label", when(col("age") > 50, 1).otherwise(0)  # Example logic for labels
)

# Step 4: Select final patient-level features
patient_features = patient_df.select(
    col("state"),
    col("has_chronic_conditions"),
    col("total_healthcare_cost"),
    col("age"),
    col("is_senior"),
    col("region_risk"),
    col("label")
)

# Step 5: Additional feature calculations (encounters, claims, etc.)
encounter_features = encounter_with_participant_df.groupBy("patient_reference").agg(
    F.count("encounter_id").alias("num_encounters"),
    F.sum(when(F.col("status") == "completed", 1).otherwise(0)).alias("completed_encounters"),
    F.min("start_time").alias("first_encounter"),
    F.max("end_time").alias("last_encounter")
)

claim_features = claim_df.groupBy("patient_reference").agg(
    F.count("claim_id").alias("num_claims"),
    F.sum(col("total_amount").cast("float")).alias("total_claim_amount")
)

condition_features = condition_df.groupBy("patient_reference").agg(
    F.count("condition_id").alias("num_conditions")
)

procedure_features = procedure_df.groupBy("subject_reference").agg(
    F.count("procedure_id").alias("num_procedures")
)

immunization_features = immunization_df.groupBy("patient_reference").agg(
    F.count("immunization_id").alias("num_immunizations")
)

diagnostic_features = diagnostic_report_df.groupBy("subject_reference").agg(
    F.count("diagnostic_report_id").alias("num_diagnostic_reports")
)

#%% md
# RISK SCORE PREDICTION : Identifying high-risk patients allows healthcare providers to intervene proactively, allocate resources more effectively, and ultimately improve patient care while minimizing healthcare costs.
#%% md
# MODEL 1 : RANDOM FOREST
#%%
# Check the schema of procedure_features to ensure correct column names
procedure_features.printSchema()


# Proceed with the rest of the code if the column exists in both dataframes

# Adjust the join to use the correct reference
data = patient_features.alias("p") \
    .join(encounter_features.alias("e"), col("p.state") == col("e.patient_reference"), "left") \
    .join(claim_features.alias("c"), col("p.state") == col("c.patient_reference"), "left") \
    .join(condition_features.alias("cond"), col("p.state") == col("cond.patient_reference"), "left") \
    .join(procedure_features.alias("proc"), col("p.state") == col("proc.subject_reference"), "left") \
    .join(immunization_features.alias("imm"), col("p.state") == col("imm.patient_reference"), "left") \
    .join(diagnostic_features.alias("diag"), col("p.state") == col("diag.subject_reference"), "left") \
    .select(
        col("p.state"),
        col("p.has_chronic_conditions"),
        col("p.total_healthcare_cost"),
        col("p.age"),
        col("p.is_senior"),
        col("p.region_risk"),
        col("p.label"),
        col("e.num_encounters"),
        col("e.completed_encounters"),
        col("c.num_claims"),
        col("c.total_claim_amount"),
        col("cond.num_conditions"),
        col("proc.num_procedures"),
        col("imm.num_immunizations"),
        col("diag.num_diagnostic_reports")
    )

# Fill missing values
data = data.fillna(0)

# Convert to Pandas for ML Model (if necessary)
data_pd = data.toPandas()

# Define X and y
X = data_pd.drop("label", axis=1)
y = data_pd["label"]

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=5, random_state=42)

# Preprocessing and Model Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# Define numeric and categorical features
numeric_features = ["total_healthcare_cost", "age", "region_risk", "num_encounters", "completed_encounters",
                    "num_claims", "total_claim_amount", "num_conditions", "num_procedures", "num_immunizations",
                    "num_diagnostic_reports"]
categorical_features = ["state", "has_chronic_conditions", "is_senior"]

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print("Predicted Risk Scores for the Test Data:")
print(y_pred)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R²) Accuracy: {r2}")
#%% md
# MODEL 2 : XGB REGRESSOR
#%%
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Combine the training and test data
combined_data = pd.concat([X_train['state'], X_test['state']], ignore_index=True)

# Fit the encoder on the combined dataset
label_encoder.fit(combined_data)

# Transform both training and testing data
X_train['state'] = label_encoder.transform(X_train['state'])
X_test['state'] = label_encoder.transform(X_test['state'])

# Initialize and train the model
model = XGBRegressor(n_estimators=1000, learning_rate=0.01, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print("Predicted Risk Scores for the Test Data:")
print(y_pred)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R²) Accuracy: {r2}")


from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validated MSE: {-cv_scores.mean()}")

#%% md
# CONCLUSION : Both models are performing well but the models might be overfitting. 
#%% md
# TRYING TO FIND THE BEST PARAMETERS and optimized MSE and R for readmission rate
#%%
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Define the parameter grid for Random Forest
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [10, 15, 20],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [2, 4, 6],
    'regressor__max_features': ['sqrt', 'log2']  # Removed 'auto'
}

# Create the GridSearchCV object
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best parameters
print("Best parameters found: ", grid_search.best_params_)

# Use the best model from grid search
best_model = grid_search.best_estimator_

# Evaluate the best model
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Optimized Mean Squared Error: {mse}")
print(f"Optimized R-squared (R²) Accuracy: {r2}")

#%% md
# READMISSION RATE : Identifying readmission rate helps us predict the probability of a patient returning back based on his demographic features and condiiton requirements
#%% md
# MODEL 1 : RANDOM FOREST
#%%
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Join the dataframes to create a feature set for readmission prediction
readmission_data = patient_features.alias("p") \
    .join(encounter_features.alias("e"), col("p.state") == col("e.patient_reference"), "left") \
    .join(claim_features.alias("c"), col("p.state") == col("c.patient_reference"), "left") \
    .join(condition_features.alias("cond"), col("p.state") == col("cond.patient_reference"), "left") \
    .join(procedure_features.alias("proc"), col("p.state") == col("proc.subject_reference"), "left") \
    .join(immunization_features.alias("imm"), col("p.state") == col("imm.patient_reference"), "left") \
    .join(diagnostic_features.alias("diag"), col("p.state") == col("diag.subject_reference"), "left") \
    .select(
        col("p.state"),
        col("p.has_chronic_conditions"),
        col("p.total_healthcare_cost"),
        col("p.age"),
        col("p.is_senior"),
        col("p.region_risk"),
        col("p.label"),
        col("e.num_encounters"),
        col("e.completed_encounters"),
        col("c.num_claims"),
        col("c.total_claim_amount"),
        col("cond.num_conditions"),
        col("proc.num_procedures"),
        col("imm.num_immunizations"),
        col("diag.num_diagnostic_reports"),
        col("e.first_encounter"),
        col("e.last_encounter")
    )

# Fill missing values
readmission_data = readmission_data.fillna(0)

# Convert to Pandas DataFrame for model training (if needed)
readmission_data_pd = readmission_data.toPandas()

# Define X and y
X = readmission_data_pd.drop("label", axis=1)  # Remove label or other non-feature columns
y = readmission_data_pd["label"]  # Assuming 'label' represents the readmission status

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define numeric and categorical features
numeric_features = ["total_healthcare_cost", "age", "region_risk", "num_encounters", "completed_encounters",
                    "num_claims", "total_claim_amount", "num_conditions", "num_procedures", "num_immunizations",
                    "num_diagnostic_reports"]
categorical_features = ["state", "has_chronic_conditions", "is_senior"]

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Adjusted Random Forest Regressor to prevent overfitting
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=10,               # Limit tree depth to prevent overfitting
        min_samples_split=5,        # Require more samples to split a node
        min_samples_leaf=4,         # Require more samples in leaf nodes
        max_features="sqrt"        # Use only a subset of features for each split
    ))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared (R²) Accuracy: {r2}")
print("Predicted Readmission Rates:")
print(y_pred)

# Cross-validation for more robust evaluation
cross_val_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
print(f"Cross-validated MSE: {-cross_val_scores.mean()}")


#%% md
# TRYING TO FIND THE BEST PARAMETERS and optimized MSE and R for readmission rate
#%%
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Define the parameter grid for Random Forest
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [10, 15, 20],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [2, 4, 6],
    'regressor__max_features': ['sqrt', 'log2']  # Removed 'auto'
}

# Create the GridSearchCV object
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best parameters
print("Best parameters found: ", grid_search.best_params_)

# Use the best model from grid search
best_model = grid_search.best_estimator_

# Evaluate the best model
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Optimized Mean Squared Error: {mse}")
print(f"Optimized R-squared (R²) Accuracy: {r2}")

#%% md
# MODEL 2 : XGB
#%%
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Join the dataframes to create a feature set for readmission prediction
readmission_data = patient_features.alias("p") \
    .join(encounter_features.alias("e"), col("p.state") == col("e.patient_reference"), "left") \
    .join(claim_features.alias("c"), col("p.state") == col("c.patient_reference"), "left") \
    .join(condition_features.alias("cond"), col("p.state") == col("cond.patient_reference"), "left") \
    .join(procedure_features.alias("proc"), col("p.state") == col("proc.subject_reference"), "left") \
    .join(immunization_features.alias("imm"), col("p.state") == col("imm.patient_reference"), "left") \
    .join(diagnostic_features.alias("diag"), col("p.state") == col("diag.subject_reference"), "left") \
    .select(
        col("p.state"),
        col("p.has_chronic_conditions"),
        col("p.total_healthcare_cost"),
        col("p.age"),
        col("p.is_senior"),
        col("p.region_risk"),
        col("p.label"),
        col("e.num_encounters"),
        col("e.completed_encounters"),
        col("c.num_claims"),
        col("c.total_claim_amount"),
        col("cond.num_conditions"),
        col("proc.num_procedures"),
        col("imm.num_immunizations"),
        col("diag.num_diagnostic_reports"),
        col("e.first_encounter"),
        col("e.last_encounter")
    )

# Fill missing values
readmission_data = readmission_data.fillna(0)

# Convert to Pandas DataFrame for model training (if needed)
readmission_data_pd = readmission_data.toPandas()

# Define X and y
X = readmission_data_pd.drop("label", axis=1)  # Remove label or other non-feature columns
y = readmission_data_pd["label"]  # Assuming 'label' represents the readmission status

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define numeric and categorical features
numeric_features = ["total_healthcare_cost", "age", "region_risk", "num_encounters", "completed_encounters",
                    "num_claims", "total_claim_amount", "num_conditions", "num_procedures", "num_immunizations",
                    "num_diagnostic_reports"]
categorical_features = ["state", "has_chronic_conditions", "is_senior"]

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Adjusted XGBRegressor to prevent overfitting
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", xgb.XGBRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=10,               # Limit tree depth to prevent overfitting
        min_samples_split=5,        # Require more samples to split a node
        min_samples_leaf=4,         # Require more samples in leaf nodes
        max_features="sqrt"        # Use only a subset of features for each split
    ))
])

# Train the model
model.fit(X_train, y_train)

# Predict using the trained model
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared (R²) Accuracy: {r2}")
print("Predicted Readmission Rates:")
print(y_pred)

# Cross-validation for more robust evaluation
cross_val_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
print(f"Cross-validated MSE: {-cross_val_scores.mean()}")
