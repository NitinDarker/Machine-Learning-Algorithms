# Naive Bayes Classification using OOPS (From Scratch)

import math


class NaiveBayesClassifier:

    def __init__(self):
        self.summaries = {}
        self.priors = {}

    # --------------------------------------------------
    # Step 1: Separate data by class
    # --------------------------------------------------
    def separate_by_class(self, X, y):
        separated = {}
        for i in range(len(X)):
            label = y[i]
            if label not in separated:
                separated[label] = []
            separated[label].append(X[i])
        return separated

    # Example:
    # print(separate_by_class(X, y))
    # {
    #   0: [[1.0, 20.0], [2.0, 21.0], [3.0, 22.0]],
    #   1: [[8.0, 30.0], [9.0, 31.0], [10.0, 32.0]]
    # }

    # --------------------------------------------------
    # Step 2: Calculate mean
    # --------------------------------------------------
    def mean(self, values):
        return sum(values) / len(values)

    # Example:
    # mean([1,2,3]) → 2.0

    # --------------------------------------------------
    # Step 3: Calculate variance
    # --------------------------------------------------
    def variance(self, values):
        mu = self.mean(values)
        return sum((x - mu) ** 2 for x in values) / len(values)

    # Example:
    # variance([1,2,3]) → 0.6667

    # --------------------------------------------------
    # Step 4: Summarize dataset (mean, variance per feature)
    # --------------------------------------------------
    def summarize_dataset(self, dataset):
        summaries = []
        for column in zip(*dataset):
            summaries.append((self.mean(column), self.variance(column)))
        return summaries

    # Example:
    # summarize_dataset([[1,20],[2,21],[3,22]])
    # [(2.0, 0.6667), (21.0, 0.6667)]

    # --------------------------------------------------
    # Step 5: Summarize by class (TRAINING)
    # --------------------------------------------------
    def summarize_by_class(self, X, y):
        separated = self.separate_by_class(X, y)
        summaries = {}
        for label, rows in separated.items():
            summaries[label] = self.summarize_dataset(rows)
        return summaries

    # Example:
    # {
    #   0: [(2.0, 0.6667), (21.0, 0.6667)],
    #   1: [(9.0, 0.6667), (31.0, 0.6667)]
    # }

    # --------------------------------------------------
    # Step 6: Calculate prior probabilities P(y)
    # --------------------------------------------------
    def calculate_priors(self, y):
        priors = {}
        total = len(y)

        for label in y:
            priors[label] = priors.get(label, 0) + 1

        for label in priors:
            priors[label] /= total

        return priors

    # Example:
    # calculate_priors([0,0,0,1,1,1])
    # {0: 0.5, 1: 0.5}

    # --------------------------------------------------
    # Step 7: Gaussian Probability Density Function
    # --------------------------------------------------
    def gaussian_probability(self, x, mean, var):
        epsilon = 1e-9  # small constant to avoid zero
        exponent = math.exp(-((x - mean) ** 2) / (2 * var + epsilon))
        return (1 / math.sqrt(2 * math.pi * var + epsilon)) * exponent + epsilon

    # --------------------------------------------------
    # Step 8: Calculate posterior probabilities (LOG)
    # --------------------------------------------------
    def calculate_posterior(self, row):
        posteriors = {}

        for label, features in self.summaries.items():
            log_prob = math.log(self.priors[label])

            for i in range(len(features)):
                mu, var = features[i]
                prob = self.gaussian_probability(row[i], mu, var)
                log_prob += math.log(prob + 1e-9) # => if prob == 0 then log(0) => undefined so log(1e-9)=0.000000001

            posteriors[label] = log_prob

        return posteriors


    # Example output:
    # {0: -2.3401, 1: -50.2314}

    # --------------------------------------------------
    # Step 9: Predict class
    # --------------------------------------------------
    def predict(self, row):
        posteriors = self.calculate_posterior(row)
        print("Posterior Probabilities:", posteriors)
        return max(posteriors, key=posteriors.get)

    # --------------------------------------------------
    # Step 10: Train the model
    # --------------------------------------------------
    def fit(self, X, y):
        self.summaries = self.summarize_by_class(X, y)
        self.priors = self.calculate_priors(y)


# ==================================================
# DATASET (Same as Numerical Example)
# ==================================================

X = [
    [1.0, 20.0],
    [2.0, 21.0],
    [3.0, 22.0],
    [8.0, 30.0],
    [9.0, 31.0],
    [10.0, 32.0]
]

y = [0, 0, 0, 1, 1, 1]

# ==================================================
# RUN THE MODEL
# ==================================================

model = NaiveBayesClassifier()
model.fit(X, y)

test_point = [72.5, 1.5]
prediction = model.predict(test_point)

print("Predicted Class:", prediction)
