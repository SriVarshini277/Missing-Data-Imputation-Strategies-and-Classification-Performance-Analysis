#load library
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#loading data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
columns = ["ID", "ClumpThickness", "CellSize", "CellShape", 
           "MarginalAdhesion", "EpithelialSize", "BareNuclei", 
           "BlandChromatin", "NormalNucleoli", "Mitoses", "Class"]
bcDf = pd.read_csv(url, names=columns)
print(bcDf.head())

#replace null values with NaN
bcDf = bcDf.replace("?", np.NaN)

#checking datatype for all columns
print(bcDf.dtypes)

#converting BareNuclei column from text to numeric type
bcDf['BareNuclei'] = pd.to_numeric(bcDf['BareNuclei'])
print(bcDf.dtypes)

#finding no of null values in each column
print(bcDf.isnull().sum())

print(f"\nNumber of missing values in BareNuclei: {bcDf['BareNuclei'].isna().sum()}")

#-----------------------------------------------------------------------------------------------------------------
#Method 1 - MEAN IMPUTATION

#Make copy of the data
df1 = bcDf.copy()

#calculate the mean of BareNuclei column
mean_value = df1['BareNuclei'].mean()
print(f"Mean value for BareNuclei: {mean_value:.2f}")

#missing value before imputation
print("Missing values before imputation:", df1['BareNuclei'].isna().sum())

#fill missing values with the mean
df1['BareNuclei'].fillna(mean_value, inplace=True)

#missing value after imputation
print("Missing values after imputation:", df1['BareNuclei'].isna().sum())
#-----------------------------------------------------------------------------------------------------------------
# METHOD 2 - REGRESSION IMPUTATION

#Make a copy of the data
df2 = bcDf.copy()

#Get rows WITHOUT missing values 
complete_data = df2[df2['BareNuclei'].notna()]

#Prepare the data for regression
feature_cols = ["ClumpThickness", "CellSize", "CellShape", 
                "MarginalAdhesion", "EpithelialSize", "BlandChromatin", 
                "NormalNucleoli", "Mitoses"]

X_complete = complete_data[feature_cols]
y_complete = complete_data['BareNuclei']

#missing value before imputation
print("Missing values before imputation:", df2['BareNuclei'].isna().sum())

#Train a regression model
model = LinearRegression()
model.fit(X_complete, y_complete)

#Find rows WITH missing values
missing_mask = df2['BareNuclei'].isna()
X_missing = df2[missing_mask][feature_cols]

#Predict the missing values
predicted_values = model.predict(X_missing)
print("Predicted BareNuclei values for missing rows:")
print(predicted_values)

#Fill in the missing values
df2.loc[missing_mask, 'BareNuclei'] = predicted_values

#missing value after imputation
print("Missing values after imputation:", df2['BareNuclei'].isna().sum())
#-----------------------------------------------------------------------------------------------------------------

# METHOD 3 - REGRESSION WITH PERTURBATION

#Make copy of the data
df3 = bcDf.copy()

#missing value before imputation
print("Missing values before imputation:", df3['BareNuclei'].isna().sum())

# Use the same model from Method 2
# Calculate predictions
predicted_values = model.predict(X_missing)

#Calculate the prediction errors on complete data
predictions_complete = model.predict(X_complete)
residuals = y_complete - predictions_complete
std_error = np.std(residuals)

print(f"Standard error of predictions: {std_error:.2f}")

# Add random noise to predictions
noise = np.random.normal(0, std_error, size=len(predicted_values))
predicted_with_noise = predicted_values + noise

# Fill in the missing values
df3.loc[missing_mask, 'BareNuclei'] = predicted_with_noise

#missing value after imputation
print("Missing values after imputation:", df2['BareNuclei'].isna().sum())

#-----------------------------------------------------------------------------------------------------------------

#METHOD 4 - LISTWISE DELETION

df4 = bcDf.dropna()
print(f"Dataset reduced from {len(bcDf)} to {len(df4)} rows")
#-----------------------------------------------------------------------------------------------------------------

#METHOD 5 - MISSING INDICATOR

#Make a copy of the data
df5 = bcDf.copy()

#Create a new column: 1 if missing, 0 if not
df5['BareNuclei_Missing'] = df5['BareNuclei'].isna().astype(int)

#Fill missing values with mean
df5['BareNuclei'].fillna(mean_value, inplace=True)

print(df5.head())
#-----------------------------------------------------------------------------------------------------------------
print("COMPARING CLASSIFICATION PERFORMANCE")

def evaluate_dataset(dataset, method_name):
    df = bcDf.copy()
    # Remove ID column
    df = dataset.drop('ID', axis=1)
    # Split into features (X) and target (y)
    X = df.drop('Class', axis=1)
    y = df['Class']
    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    knn_accuracy = accuracy_score(y_test, knn_pred)
    # SVM Classifier
    svm = SVC()
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    
    return knn_accuracy, svm_accuracy

# Evaluate each method
print("\nResults (Accuracy scores):")
print("-" * 50)

knn1, svm1 = evaluate_dataset(df1, "Mean Imputation")
print(f"Method 1 (Mean)         - KNN: {knn1:.3f}, SVM: {svm1:.3f}")

knn2, svm2 = evaluate_dataset(df2, "Regression")
print(f"Method 2 (Regression)   - KNN: {knn2:.3f}, SVM: {svm2:.3f}")

knn3, svm3 = evaluate_dataset(df3, "Regression+Noise")
print(f"Method 3 (Reg+Noise)    - KNN: {knn3:.3f}, SVM: {svm3:.3f}")

knn4, svm4 = evaluate_dataset(df4, "Listwise Deletion")
print(f"Method 4 (Deletion)     - KNN: {knn4:.3f}, SVM: {svm4:.3f}")

knn5, svm5 = evaluate_dataset(df5, "Missing Indicator")
print(f"Method 5 (Indicator)    - KNN: {knn5:.3f}, SVM: {svm5:.3f}")
#-----------------------------------------------------------------------------------------------------------------