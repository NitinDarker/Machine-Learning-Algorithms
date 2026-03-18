import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create the dataset
# data = {
#     "Day": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain"],
#     "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild"],
#     "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal"],
#     "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak"],
#     "PlayTennis": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes"]
# }

# Create DataFrame
# df = pd.DataFrame(data)
# df = df.drop(columns=['Day'])
# Display DataFrame
# print(df)

df = pd.read_csv('Iris.csv')

def entropy(target_column):
    """
    target_column: pandas Series (e.g., df['PlayTennis'])
    """
    values, counts = np.unique(target_column, return_counts=True)
    # ['Overcast' 'Rain' 'Sunny'] -> values
    # [2 4 4] -> count
    probabilities = counts / counts.sum()
    # print(probabilities) #[0.2 0.4 0.4]
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return entropy_value

# print(entropy(df['Outlook']))

def information_gain(df, feature, target):
    """
    df      : pandas DataFrame
    feature : column name (string)
    target  : target column name (string)
    """
    # 1. Parent entropy
    parent_entropy = entropy(df[target])
    # 2. Get unique values of the feature
    feature_values = df[feature].unique()
    # print(feature_values) #['Sunny', 'Overcast', 'Rain']
    weighted_entropy = 0
    for value in feature_values:
        subset = df[df[feature] == value] # df only Sunny if value is Sunny

        weight = len(subset) / len(df)

        weighted_entropy += weight * entropy(subset[target])
        # print(subset)

    # 4. Information Gain
    ig = parent_entropy - weighted_entropy
    return ig


def best_feature(df, target):
    """
    df     : pandas DataFrame
    target : target column name (string)
    """
    
    features = [col for col in df.columns if col != target]
    # print(features)
    ig_values = {}
    for feature in features:
        ig = information_gain(df, feature, target)
        ig_values[feature] = ig

    best_feature = max(ig_values, key=lambda k: ig_values[k])
      
    return best_feature, ig_values
    

# best_col, ig_scores = best_feature(df, 'PlayTennis')

# print("Best Feature:", best_col)
# print("IG Scores:", ig_scores)
    
def is_pure(target_column):
    return len(target_column.unique()) == 1

def majority_class(target_column):
    return target_column.value_counts().idxmax()



def build_tree(df, target, features=None):
    """
    df       : pandas DataFrame
    target   : target column name
    features : list of remaining feature names
    """
    
    
    # 1️ Initialize feature list
    if features is None:
        features = [col for col in df.columns if col != target]
    
    # 2️ Stopping condition: empty dataset
    if df.empty:
        return None
    
    # 3️ Stopping condition: pure node
    if is_pure(df[target]):
        return df[target].iloc[0]
    
    # 4️⃣ Stopping condition: no features left
    if len(features) == 0:
        return majority_class(df[target])
    
    # 5️⃣ Choose best feature
    best_col, _ = best_feature(df, target)
    
    tree = {best_col: {}}
    
    # 6️⃣ Grow branches
    for value in df[best_col].unique():
        subset = df[df[best_col] == value]
        
        # Remove used feature
        remaining_features = [f for f in features if f != best_col]
        
        subtree = build_tree(
            subset.drop(columns=best_col),
            target,
            remaining_features
        )
        
        tree[best_col][value] = subtree
    
    return tree



def count_leaves(tree):
    if not isinstance(tree, dict):
        return 1
    
    root = next(iter(tree))
    return sum(count_leaves(subtree) for subtree in tree[root].values())

def tree_depth(tree):
    if not isinstance(tree, dict):
        return 1
    
    root = next(iter(tree))
    return 1 + max(tree_depth(subtree) for subtree in tree[root].values())


def draw_node(text, xy, xytext, node_type):
    boxstyle = "round" if node_type == "leaf" else "sawtooth"
    
    plt.annotate(
        text,
        xy=xy,
        xytext=xytext,
        bbox=dict(boxstyle=boxstyle, fc="lightblue"),
        arrowprops=dict(arrowstyle="<-"),
        ha='center'
    )


def plot_tree(tree, parent_xy, text, x_min, x_max, y, y_step):
    root = next(iter(tree))
    center_x = (x_min + x_max) / 2
    
    draw_node(root, parent_xy, (center_x, y), "decision")
    
    num_leaves = count_leaves(tree)
    x_step = (x_max - x_min) / num_leaves
    current_x = x_min
    
    for value, subtree in tree[root].items():
        if isinstance(subtree, dict):
            subtree_leaves = count_leaves(subtree)
            next_x = current_x + x_step * subtree_leaves / 2
            
            plt.text((center_x + next_x)/2, y - y_step/2, str(value))
            
            plot_tree(
                subtree,
                (center_x, y),
                value,
                current_x,
                current_x + x_step * subtree_leaves,
                y - y_step,
                y_step
            )
            current_x += x_step * subtree_leaves
        else:
            leaf_x = current_x + x_step / 2
            plt.text((center_x + leaf_x)/2, y - y_step/2, str(value))
            draw_node(subtree, (center_x, y), (leaf_x, y - y_step), "leaf")
            current_x += x_step


def visualize_tree(tree):
    plt.figure(figsize=(12, 8))
    plt.axis('off')
    
    depth = tree_depth(tree)
    plot_tree(
        tree,
        parent_xy=(0.5, 1.0),
        text='',
        x_min=0.0,
        x_max=1.0,
        y=1.0,
        y_step=1.0 / depth
    )
    
    plt.show()


tree = build_tree(df, 'Species')
visualize_tree(tree)