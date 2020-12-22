from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
import nltk

nltk.download('product_reviews_1')

# Load the reviews from the corpus with movie reviews method
fileids_pos = movie_reviews.fileids('pos')
fileids_neg = movie_reviews.fileids('neg')


# Extract the features from the reviews with movie reviews method
def extract_features(words):
    return dict([(word, True) for word in words])


features_pos = [(extract_features(movie_reviews.words(
    fileids=[f])), 'Positive') for f in fileids_pos]
features_neg = [(extract_features(movie_reviews.words(
    fileids=[f])), 'Negative') for f in fileids_neg]

# Define the train and test split (80% and 20%)
threshold = 0.8
num_pos = int(threshold * len(features_pos))
num_neg = int(threshold * len(features_neg))

# Create training and training datasets
features_train = features_pos[:num_pos] + features_neg[:num_neg]
features_test = features_pos[num_pos:] + features_neg[num_neg:]

# Print the number of datapoints used
print('\nNumber of training datapoints:', len(features_train))
print('Number of test datapoints:', len(features_test))

# Train a Naive Bayes classifier
classifier = NaiveBayesClassifier.train(features_train)
print('\nAccuracy of the classifier:', nltk_accuracy(
    classifier, features_test))

# Test input movie reviews
input_reviews = [
    'This Product is very easy to use',
    'This Product is very bad',
    'This Product is no advantages',
    'This Product is has many advantages',
]

print("\nProduct Prediction Reviews from Amazon.com :")
for review in input_reviews:
    print("\nReview:", review)

    # Compute the probabilities
    probabilities = classifier.prob_classify(extract_features(review.split()))

    # Pick the maximum value
    predicted_sentiment = probabilities.max()

    # Print outputs
    print("Result Sentiment Analytic:", predicted_sentiment)
    print("Propability:", round(probabilities.prob(predicted_sentiment), 2))
