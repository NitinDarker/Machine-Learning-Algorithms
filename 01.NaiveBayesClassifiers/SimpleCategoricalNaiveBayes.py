class SimpleCategoricalNaiveBayes:
    def fit(self, X, y):
        self.classes = list(set(y))          # unique classes
        self.priors = {}                     # P(class)
        self.cond_probs = {}                 # P(feature=value | class)
        self.feature_value_counts = []      # number of unique values per feature

        n = len(y)
        n_features = len(X[0])

        # Calculate priors and initialize cond_probs dict
        for c in self.classes:
            X_c = [X[i] for i in range(n) if y[i] == c]   # samples of class c
            self.priors[c] = len(X_c) / n

            self.cond_probs[c] = [{} for _ in range(n_features)]  # one dict per feature

            for feature_idx in range(n_features):
                counts = {}
                for sample in X_c:
                    val = sample[feature_idx]
                    counts[val] = counts.get(val, 0) + 1
                self.cond_probs[c][feature_idx] = counts

        # Calculate how many unique values each feature has overall (for smoothing)
        for feature_idx in range(n_features):
            values = set([X[i][feature_idx] for i in range(n)])
            self.feature_value_counts.append(len(values))

    def predict(self, x):
        best_class = None
        max_prob = -1

        for c in self.classes:
            prob = self.priors[c]

            for idx, val in enumerate(x):
                counts = self.cond_probs[c][idx]
                count = counts.get(val, 0)
                total = sum(counts.values())

                # Laplace smoothing:
                prob *= (count + 1) / (total + self.feature_value_counts[idx])

            if prob > max_prob:
                max_prob = prob
                best_class = c

        return best_class
