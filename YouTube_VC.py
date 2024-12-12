#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# In[12]:


# Load dataset
data_path = "US_youtube_trending_data.csv"
columns = [
    "video_id", "title", "publishedAt", "channelId", "channelTitle",
    "categoryId", "trending_date", "tags", "views", "likes", "dislikes",
    "comment_count", "thumbnail_link", "comments_disabled",
    "ratings_disabled", "description"
]


# In[13]:


# Load and preprocess dataset
dataset = pd.read_csv(data_path, names=columns, header=0)


# In[14]:


# Basic data analysis
def basic_data_analysis(df):
    print("Basic Data Analysis")
    print("-------------------------")
    print("Data Types:")
    print(df.dtypes)
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nNumber of Unique Values:")
    print(df.nunique())


# In[15]:


# Perform basic data analysis
basic_data_analysis(dataset)


# In[16]:


# Handle missing values: Impute numerical columns with mean and categorical with mode
def handle_missing_values(df):
    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Impute numerical columns with mean
    imputer_num = SimpleImputer(strategy='mean')
    df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])

    # Impute categorical columns with mode
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
    
    return df

# Apply missing values handling
dataset = handle_missing_values(dataset)


# In[17]:


# Feature engineering: Create new features (e.g., views per like ratio, etc.)
def feature_engineering(df):
    # Create new feature: views per like ratio
    df['views_per_like'] = df['views'] / (df['likes'] + 1)  # Adding 1 to avoid division by zero
    df['views_per_dislike'] = df['views'] / (df['dislikes'] + 1)
    
    # Convert boolean columns to integers (as already done in your original code)
    df["comments_disabled"] = df["comments_disabled"].astype(int)
    df["ratings_disabled"] = df["ratings_disabled"].astype(int)
    
    return df

# Apply feature engineering
dataset = feature_engineering(dataset)


# In[18]:


# Feature selection
features = ["views", "likes", "dislikes", "comment_count", "comments_disabled", "ratings_disabled", "views_per_like", "views_per_dislike"]
target = "categoryId"


# In[19]:


# Clean data by dropping rows with missing target values
cleaned_data = dataset.dropna(subset=[target])[features + [target]]


# In[20]:


# Split data into training and test sets
X = cleaned_data[features]
y = cleaned_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[21]:


# Feature scaling: Standardize numerical features
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Apply feature scaling
X_train_scaled, X_test_scaled = scale_features(X_train, X_test)


# In[22]:


# Train Random Forest Classifier
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train_scaled, y_train)


# In[23]:


# Make predictions
y_pred = model.predict(X_test_scaled)


# In[24]:


# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

# Print results (model evaluation)
print("Random Forest Model Evaluation")
print("-------------------------------")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(report)


# In[ ]:




