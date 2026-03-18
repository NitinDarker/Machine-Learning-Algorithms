import random
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# -----------------------------
# Step 1: Generate dataset
# -----------------------------
def generate_demo_dataset(n_samples=200):
    categories1 = ['Red', 'Green', 'Blue']
    categories2 = ['Small', 'Medium', 'Large']
    categories3 = ['Yes', 'No']
    categories4 = ['Urban', 'Rural']

    data = []
    labels = []

    for _ in range(n_samples):
        row = [
            random.choice(categories1),               # categorical 1
            random.uniform(10, 100),                 # numerical 1
            random.choice(categories2),               # categorical 2
            random.uniform(0, 50),                   # numerical 2
            random.choice(categories3),               # categorical 3
            random.uniform(100, 200),                # numerical 3
            random.uniform(1, 10),                   # numerical 4
            random.choice(categories4),               # categorical 4
            random.uniform(500, 1000),               # numerical 5
            random.uniform(0, 1)                     # numerical 6
        ]
        data.append(row)
        # Generate labels (0 or 1) randomly
        labels.append(random.choice([0, 1]))
    return data, labels

# -----------------------------
# Step 2: Encode categorical automatically
# -----------------------------
def encode_categorical_columns_auto(dataset):
    if not dataset:
        return [], {}, []

    n_cols = len(dataset[0])
    mappings = {}
    categorical_indices = []

    # Detect categorical columns
    for col_idx in range(n_cols):
        for row in dataset:
            val = row[col_idx]
            if val is not None:
                if isinstance(val, str) or isinstance(val, bool):
                    categorical_indices.append(col_idx)
                    mappings[col_idx] = {}
                break

    encoded_dataset = []
    for row in dataset:
        encoded_row = row[:]
        for idx in categorical_indices:
            val = row[idx]
            if val not in mappings[idx]:
                mappings[idx][val] = len(mappings[idx])
            encoded_row[idx] = mappings[idx][val]
        encoded_dataset.append(encoded_row)

    return encoded_dataset, mappings, categorical_indices

# -----------------------------
# Step 3: Your NaiveBayesClassifier
# -----------------------------
class NaiveBayesClassifier:
    def __init__(self):
        self.summaries = {}
        self.priors = {}

    def separate_by_class(self, X, y):
        separated = {}
        for i in range(len(X)):
            label = y[i]
            if label not in separated:
                separated[label] = []
            separated[label].append(X[i])
        return separated

    def mean(self, values):
        return sum(values) / len(values)

    def variance(self, values):
        mu = self.mean(values)
        if len(values) <= 1: return 0.0
        return sum((x - mu) ** 2 for x in values) / (len(values) - 1)

    def summarize_dataset(self, dataset):
        summaries = []
        for column in zip(*dataset):
            summaries.append((self.mean(column), self.variance(column)))
        return summaries

    def summarize_by_class(self, X, y):
        separated = self.separate_by_class(X, y)
        summaries = {}
        for label, rows in separated.items():
            summaries[label] = self.summarize_dataset(rows)
        return summaries

    def calculate_priors(self, y):
        priors = {}
        total = len(y)
        for label in y:
            priors[label] = priors.get(label, 0) + 1
        for label in priors:
            priors[label] /= total
        return priors

    def gaussian_probability(self, x, mean, var):
        epsilon = 1e-9
        exponent = math.exp(-((x - mean) ** 2) / (2 * var + epsilon))
        return (1 / math.sqrt(2 * math.pi * var + epsilon)) * exponent + epsilon

    def calculate_posterior(self, row):
        posteriors = {}
        for label, features in self.summaries.items():
            log_prob = math.log(self.priors[label])
            for i in range(len(features)):
                mu, var = features[i]
                prob = self.gaussian_probability(row[i], mu, var)
                log_prob += math.log(prob + 1e-9)
            posteriors[label] = log_prob
        return posteriors

    def predict(self, row):
        posteriors = self.calculate_posterior(row)
        return max(posteriors, key=posteriors.get)

    def fit(self, X, y):
        self.summaries = self.summarize_by_class(X, y)
        self.priors = self.calculate_priors(y)

# -----------------------------
# Step 4: Comparison Function
# -----------------------------
def compare_with_sklearn(encoded_data, labels):
    """
    Splits the data into train/test sets and compares the custom 
    NaiveBayesClassifier against sklearn's GaussianNB.
    """
    # Split data: 70% Train, 30% Test
    X_train, X_test, y_train, y_test = train_test_split(
        encoded_data, labels, test_size=0.3, random_state=42
    )
    
    # 1. Evaluate Custom Model
    custom_model = NaiveBayesClassifier()
    custom_model.fit(X_train, y_train)
    custom_preds = [custom_model.predict(row) for row in X_test]
    custom_acc = accuracy_score(y_test, custom_preds)
    
    # 2. Evaluate Sklearn Model
    sk_model = GaussianNB()
    sk_model.fit(X_train, y_train)
    sk_preds = sk_model.predict(X_test)
    sk_acc = accuracy_score(y_test, sk_preds)
    
    # Print Comparison Table
    print("\n" + "="*40)
    print(f"{'Model Name':<20} | {'Accuracy Score':<15}")
    print("-" * 40)
    print(f"{'Custom Naive Bayes':<20} | {custom_acc:.4%}")
    print(f"{'Sklearn GaussianNB':<20} | {sk_acc:.4%}")
    print("="*40 + "\n")
    
    return custom_acc, sk_acc

# -----------------------------
# Main Execution
# -----------------------------
# if __name__ == "__main__":
#     # Generate and encode data
#     raw_data, labels = generate_demo_dataset(9000)
#     encoded_data, mappings, cat_cols = encode_categorical_columns_auto(raw_data)
    
#     print(f"Dataset generated with {len(raw_data)} samples.")
#     print(f"Categorical columns detected at indices: {cat_cols}")
    
#     # Run the comparison
#     compare_with_sklearn(encoded_data, labels)

if __name__ == "__main__":
    # 1. Load the CSV
    # df = pd.read_csv('Iris.csv')
    df = pd.read_csv('diabetes.csv')

    # 2. Cleanup: Remove 'Id' if it exists, as it's not a real feature
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])

    # 3. Separate Features (X) and Target Label (y)
    # Assuming the last column is the species/label
    target_col = df.columns[-1]
    labels_raw = df[target_col].tolist()
    features_df = df.drop(columns=[target_col])

    # 4. Convert Features to List of Lists for your existing function
    raw_data_list = features_df.values.tolist()

    # 5. Encode Features
    encoded_data, mappings, cat_cols = encode_categorical_columns_auto(raw_data_list)

    # 6. Encode Labels (since Iris species are 'Iris-setosa', etc.)
    # Your model expects numerical labels (0, 1, 2)
    label_mapping = {val: i for i, val in enumerate(set(labels_raw))}
    final_labels = [label_mapping[l] for l in labels_raw]

    print(f"Dataset loaded with {len(encoded_data)} samples.")
    print(f"Features detected: {list(features_df.columns)}")
    print(f"Label classes: {label_mapping}")

    # Run the comparison
    compare_with_sklearn(encoded_data, final_labels)