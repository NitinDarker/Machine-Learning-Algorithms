def encode_categorical_columns_auto(dataset):
    """
    Automatically encode categorical columns in the dataset to integers.
    Categorical columns are detected based on type: str or bool.
    
    Args:
    - dataset: List of lists (rows of data)
    
    Returns:
    - encoded_dataset: Dataset with categorical columns converted to integers
    - mappings: Dict with {col_index: {original_value: encoded_value}} for each categorical col
    - categorical_indices: List of indices that were treated as categorical
    """
    if not dataset:
        return [], {}, []

    n_cols = len(dataset[0])
    mappings = {}
    categorical_indices = []

    # Detect categorical columns
    for col_idx in range(n_cols):
        # check first non-None value in column
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


data = [
    ['Sunny', 85, 'High', False],
    ['Rainy', 70, 'Normal', True],
    ['Overcast', 72, 'High', False],
    ['Sunny', 90, 'Normal', True],
]

encoded_data, mappings, categorical_cols = encode_categorical_columns_auto(data)

print("Encoded data:")
for row in encoded_data:
    print(row)

print("\nCategorical columns detected:", categorical_cols)
print("Mappings:")
for col, mapping in mappings.items():
    print(f"Column {col}: {mapping}")
