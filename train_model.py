import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
import pickle


dataset_folder = os.path.join(os.path.dirname(__file__), "datasets")  

bullying_file = os.path.join(dataset_folder,"Aggressive_All.csv")
non_bullying_file = os.path.join(dataset_folder,"Non_Aggressive_All.csv")

bullying_data = pd.read_csv(bullying_file,encoding='utf-8')  # Update with actual file path
non_bullying_data = pd.read_csv(non_bullying_file,encoding='utf-8')

# Ensure DataFrame type consistency after assigning labels
bullying_data.reset_index(drop=True, inplace=True)
non_bullying_data.reset_index(drop=True, inplace=True)

# Rename the "message" column to "text" for consistency
bullying_data.rename(columns={"Message" : "text"}, inplace=True)
non_bullying_data.rename(columns={"Message" : "text"}, inplace=True)

# Check sample data before labeling
print("Bullying Data Sample (should contain aggressive text):")
print(bullying_data[['text']].sample(10))

print("Non-Bullying Data Sample (should contain non-aggressive text):")
print(non_bullying_data[['text']].sample(10))

# Assign labels: 1 for bullying, 0 for non-bullying
bullying_data["label"] = 1
non_bullying_data["label"] = 0  # Assign label

# Debugging: Check if non_bullying_data still has text values
print("Non-Bullying Data After Label Assignment:")
print(non_bullying_data.sample(5))

# Combine both datasets
df = pd.concat([bullying_data, non_bullying_data],ignore_index=True)

df = df.dropna(subset=['text'])

# Convert the data to a pandas DataFrame
df = df[['text','label']]
df.reset_index(drop=True, inplace=True)
print(df.info())
print(df.sample(10))
print(df.head())

# Split the data into features (X) and labels (y)
X = df['text']
y = df['label']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data (Convert text to numerical features)
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.7, min_df=10, max_features=20000)

# Fit and transform the training data and transform the test data
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Check original class distribution
print("Original class distribution:")
print(y_train.value_counts())

rus = RandomUnderSampler(random_state=42)
X_train_vec, y_train = rus.fit_resample(X_train_vec, y_train)

# Check class distribution after undersampling
print("After undersampling class distribution:")
print(pd.Series(y_train).value_counts())

# Create a LogisticRegression model
lr_model = LogisticRegression(class_weight="balanced", max_iter=1000)

# Train the model on the training data
lr_model.fit(X_train_vec, y_train)

# Test the model
y_pred = lr_model.predict(X_test_vec)

# Print classification report to evaluate performance
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizer to disk
with open('model.pkl', 'wb') as model_file:
    pickle.dump(lr_model, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)
