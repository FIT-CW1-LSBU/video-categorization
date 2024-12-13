from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Columns for the dataset
columns = [
    "video_id", "title", "publishedAt", "channelId", "channelTitle",
    "categoryId", "trending_date", "tags", "views", "likes", "dislikes",
    "comment_count", "thumbnail_link", "comments_disabled",
    "ratings_disabled", "description"
]

# Load and preprocess dataset
def load_dataset(data_path):
    dataset = pd.read_csv(data_path, names=columns, header=0)
    dataset = handle_missing_values(dataset)
    dataset = feature_engineering(dataset)
    return dataset

# Handle missing values
def handle_missing_values(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    imputer_num = SimpleImputer(strategy='mean')
    df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])

    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

    return df

# Feature engineering
def feature_engineering(df):
    df['views_per_like'] = df['views'] / (df['likes'] + 1)  # Avoid division by zero
    df['views_per_dislike'] = df['views'] / (df['dislikes'] + 1)
    df["comments_disabled"] = df["comments_disabled"].astype(int)
    df["ratings_disabled"] = df["ratings_disabled"].astype(int)
    return df

# Feature scaling
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Flask route to train the model
@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Get the dataset path from the request
        data = request.json
        data_path = data.get("data_path")
        if not data_path:
            return jsonify({"error": "Please provide the 'data_path' in the request body."}), 400

        # Load and preprocess dataset
        dataset = load_dataset(data_path)

        # Define features and target
        features = ["views", "likes", "dislikes", "comment_count", "comments_disabled", "ratings_disabled", "views_per_like", "views_per_dislike"]
        target = "categoryId"

        # Clean and split data
        cleaned_data = dataset.dropna(subset=[target])[features + [target]]
        X = cleaned_data[features]
        y = cleaned_data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

        # Train the model
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)

        # Return the results
        return jsonify({
            "accuracy": f"{accuracy * 100:.2f}%",
            "classification_report": report
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "YouTube Video Categorization API is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
