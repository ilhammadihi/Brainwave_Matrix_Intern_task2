# Example code structure
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve
from imblearn.over_sampling import SMOTE

# Load data
data = pd.read_csv("D:/archive.zip")

# Preprocessing
X = data.drop('Class', axis=1)
y = data['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# Train model
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_res, y_res)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))