import pandas as pd
import nltk
import string
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# Load the text corpus
with open("alice.txt", "r", encoding="utf-8") as file:
    text = file.read()

print("Total characters:", len(text))
print("\nFirst 500 characters:\n")
print(text[:500])

# Convert text to lowercase
text = text.lower()

# Remove punctuation
text = text.translate(str.maketrans("", "", string.punctuation))

# Tokenization
tokens = word_tokenize(text)
print("Total tokens:", len(tokens))
print(tokens[:20])

from nltk.corpus import stopwords
stop_words = set (stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]
print("Total token:", len(tokens))
print("Tokens after stopwords removal:", len(filtered_tokens))
print("First 50 filtered token:")
print(filtered_tokens[:50])

from collections import Counter
word_freq = Counter(filtered_tokens)
print("Most common 20 words:")
print(word_freq.most_common(20))

# Top 20 frequent words
top_words = word_freq.most_common(20)
words = [item[0] for item in top_words]
counts = [item[1] for item in top_words]
plt.figure(figsize=(12, 6))
plt.bar(words, counts)
plt.xticks(rotation=45)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top 20 Most Frequent Words")
plt.tight_layout()
plt.show()

# Autocomplete function
def autocomplete(prefix, tokens, n=10):
    suggestions = []
    for word in tokens:
        if word.startswith(prefix):
            if word not in suggestions:
                suggestions.append(word)
        if len(suggestions) >= n:
            break
    return suggestions

# Test autocomplete
prefix = "alice"
suggestions = autocomplete(prefix, filtered_tokens)
print("\nAutocomplete suggestions for:", prefix)
print(suggestions)

# Frequency-based Autocomplete
def autocomplete(prefix, tokens, word_freq, n=10):
    matches = []
    for word in word_freq:
        if word.startswith(prefix):
            matches.append((word, word_freq[word]))

    # Sort by frequency
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:n]

# Test with different prefixes
for prefix in ["ch", "ca", "wo", "th"]:
    suggestions = autocomplete(prefix, filtered_tokens, word_freq)
    print("\nSuggestions for:", prefix)
    print(suggestions)

# Autocorrect using Levenshtein distance
def levenshtein_distance(word1, word2):
    rows = len(word1) + 1
    cols = len(word2) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            if word1[i - 1] == word2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[-1][-1]

def autocorrect(word, word_freq, max_distance=4):
    candidates = []

    for candidate in word_freq:
        distance = levenshtein_distance(word, candidate)

        if distance <= max_distance:
            candidates.append(
                (candidate, distance, word_freq[candidate])
            )
    candidates.sort(key=lambda x: (x[1], -x[2]))
    return candidates[:5]

# Test autocorrect
test_words = [
    "alcie",
    "advenntures",
    "wondrland",
    "chaptr"
]
for word in test_words:
    result = autocorrect(word, word_freq)
    print("\nAutocorrect for:", word)
    print(result)

    # Interactive Autocorrect
while True:
    user_word = input("\nEnter a word (or type 'exit' to stop): ").lower()

    if user_word == "exit":
        print("Autocorrect stopped.")
        break
    suggestions = autocorrect(user_word, word_freq)
    if suggestions:
        print("Suggestions:")
        for word, distance, frequency in suggestions:
            print(f"{word}  | distance: {distance} | frequency: {frequency}")
    else:
        print("No suitable correction found.")

# Visualization - Top 10 Most Common Words
top_words = dict(word_freq.most_common(10))
plt.figure(figsize=(10, 5))
plt.bar(top_words.keys(), top_words.values())
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top 10 Most Common Words")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Word Frequency Visualization

words = [word for word, count in word_freq.most_common(15)]
counts = [count for word, count in word_freq.most_common(15)]
plt.figure(figsize=(10, 6))
plt.bar(words, counts)
plt.title("Top 15 Most Common Words")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(words, counts)
plt.title("Top 15 Most Common Words")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Autocorrect accuracy evaluation

test_data = {
    "alcie": "alice",
    "chaptr": "chapter",
    "advenntures": "adventures",
    "wondrland": "wonderland",
    "beginnig": "beginning",
    "rabbit": "rabbit",
    "queen": "queen",
    "garde": "garden",
    "hatr": "hat",
    "caterpilar": "caterpillar",
    "gryphon": "gryphon",
    "madhtter": "mad",
    "croquet": "croquet",
    "duch": "duchess",
    "turtl": "turtle",
    "sister": "sister",
    "story": "story",
    "garden": "garden",
    "looking": "looking",
    "wonder": "wonder"
}
correct = 0
total = len(test_data)
for misspelled, expected in test_data.items():
    suggestions = autocorrect(misspelled, word_freq, max_distance=4)
    predicted = suggestions[0][0] if suggestions else None
    if predicted == expected:
        correct += 1
    print(f"{misspelled} -> {predicted} | Expected: {expected}")
accuracy = correct / total
print("\nAutocorrect Accuracy:", round(accuracy * 100, 2), "%")


# Precision and Recall for Autocorrect

true_positive = 0
false_positive = 0
false_negative = 0
for misspelled, expected in test_data.items():
    suggestions = autocorrect(misspelled,word_freq,max_distance=4)
    predicted = suggestions[0][0] if suggestions else None
    if predicted == expected:
        true_positive += 1
    elif predicted is not None:
        false_positive += 1
    else:
        false_negative += 1
precision = (true_positive /(true_positive + false_positive)
    if (true_positive + false_positive) > 0
    else 0
)
recall = (true_positive /(true_positive + false_negative)
    if (true_positive + false_negative) > 0
    else 0
)
print("\n--- Autocorrect Evaluation ---")
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))

# Autocomplete Precision and Recall

autocomplete_tests = {
    "al": "alice",
    "chap": "chapter",
    "adven": "adventures",
    "wond": "wonder",
    "rabb": "rabbit",
    "quee": "queen",
    "gard": "garden",
    "cater": "caterpillar",
    "sist": "sister",
    "look": "looking"
}
tp = 0
fp = 0
fn = 0
for prefix, expected in autocomplete_tests.items():
    suggestions = autocomplete(prefix,filtered_tokens,word_freq,n=3)
    predicted_words = [word for word, frequency in suggestions]
    print(f"\nPrefix: {prefix}")
    print("Predictions:", predicted_words)
    print("Expected:", expected)
    if expected in predicted_words:
        tp += 1
    else:
        fn += 1
    fp += len([word for word in predicted_words if word != expected])
precision_auto = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_auto = tp / (tp + fn) if (tp + fn) > 0 else 0
print("\n--- Autocomplete Evaluation ---")
print("Precision:", round(precision_auto, 3))
print("Recall:", round(recall_auto, 3))


# Autocomplete Precision and Recall

autocomplete_tests = {
    "al": "alice",
    "chap": "chapter",
    "adven": "adventures",
    "wond": "wonder",
    "rabb": "rabbit",
    "quee": "queen",
    "gard": "garden",
    "cater": "caterpillar",
    "sist": "sister",
    "look": "looking"
}
tp = 0
fp = 0
fn = 0
for prefix, expected in autocomplete_tests.items():
    suggestions = autocomplete(prefix,filtered_tokens,word_freq, n=3)
    predicted_words = [word for word, frequency in suggestions]
    print(f"\nPrefix: {prefix}")
    print("Predictions:", predicted_words)
    print("Expected:", expected)
    if expected in predicted_words:
        tp += 1
    else:
        fn += 1
    fp += len([word for word in predicted_words if word != expected])
precision_auto = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_auto = tp / (tp + fn) if (tp + fn) > 0 else 0
print("\n--- Autocomplete Evaluation ---")
print("Precision:", round(precision_auto, 3))
print("Recall:", round(recall_auto, 3))

from collections import defaultdict, Counter

# -------- BIGRAM MODEL --------
bigram_model = defaultdict(Counter)
for i in range(len(filtered_tokens) - 1):
    word1 = filtered_tokens[i]
    word2 = filtered_tokens[i + 1]
    bigram_model[word1][word2] += 1

def bigram_autocomplete(prefix, n=3):
    results = Counter()
    for word in bigram_model:
        if word.startswith(prefix):
            for next_word, count in bigram_model[word].items():
                results[next_word] += count
    return results.most_common(n)

# -------- TRIGRAM MODEL --------
trigram_model = defaultdict(Counter)
for i in range(len(filtered_tokens) - 2):
    word1 = filtered_tokens[i]
    word2 = filtered_tokens[i + 1]
    word3 = filtered_tokens[i + 2]
    context = (word1, word2)
    trigram_model[context][word3] += 1
def trigram_autocomplete(word1, word2, n=3):
    return trigram_model[(word1, word2)].most_common(n)

# -------- COMPARISON --------
print("\n--- Bigram Autocomplete ---")
print("Prefix: alice")
print(bigram_autocomplete("alice", 3))

print("\n--- Trigram Autocomplete ---")
print("Context: project6 gutenberg")
print(trigram_autocomplete("project", "gutenberg", 3))

# -------- WORD FREQUENCY VISUALIZATION --------
from collections import Counter
word_frequency = Counter(filtered_tokens)
top_words = word_frequency.most_common(10)
words = [item[0] for item in top_words]
counts = [item[1] for item in top_words]
plt.figure(figsize=(10, 5))
plt.bar(words, counts)
plt.title("Top 10 Most Frequent Words")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# -------- PERFORMANCE METRICS --------

from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Calculate F1 Score
f1 = 2 * (precision_auto * recall_auto) / (precision_auto + recall_auto)

print("\n--- Performance Metrics ---")
print("Precision:", round(precision_auto, 3))
print("Recall:", round(recall_auto, 3))
print("F1 Score:", round(f1, 3))

# Performance graph
metrics = ["Precision", "Recall", "F1 Score"]
values = [precision_auto, recall_auto, f1]

plt.figure(figsize=(8, 5))
plt.bar(metrics, values)

plt.title("Autocomplete Performance Metrics")
plt.xlabel("Metrics")
plt.ylabel("Score")
plt.ylim(0, 1.1)

plt.tight_layout()
plt.show()