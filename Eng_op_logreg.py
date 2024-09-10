import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from operator import itemgetter
import argparse
import matplotlib.pyplot as plt
import os
from tabulate import tabulate

# Download necessary NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')

# Argument parser for selecting n-grams
parser = argparse.ArgumentParser(description="Text Classification with N-gram TF-IDF Vectorization")
parser.add_argument('--ngram', type=str, choices=['unigram', 'bigram', 'trigram'], default='unigram',
                    help="Choose n-gram type: unigram, bigram, or trigram")
args = parser.parse_args()

# Load the dataset
data = pd.read_excel('Data.xlsx')

# Preprocessing
neutral_data = data[data['label'] == 'Neutral']
indoctrination_data = data[data['label'] == 'Indoctrination']
print("indoctrination data")
print(indoctrination_data)

# Extract sentences for neutral data
neutral_sentences = []
for paragraph in neutral_data['text_en']:
    sentences = sent_tokenize(paragraph)
    neutral_sentences.extend(sentences)

# Create a new DataFrame for balanced text data
neutral_df = pd.DataFrame(neutral_sentences, columns=['text_en'])
neutral_df['label'] = 'Neutral'
print("neutral data")
print(neutral_df)

# Combine the neutral sentences with indoctrination data
combined_data = pd.concat([neutral_df, indoctrination_data], ignore_index=True)
print("combined data")
print(combined_data)

# Function to count tokens
def count_tokens(text):
    tokens = text.split()
    return len(tokens)

# Apply the function to the specific column
combined_data['token_count'] = combined_data['text_en'].apply(count_tokens)

# Remove stopwords
stop_words = stopwords.words('english')
combined_data['text_en'] = combined_data['text_en'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

# Filter out sentences with a token count of 1
filtered_data = combined_data[combined_data['token_count'] != 1]

# Determine the n-gram range based on user input
if args.ngram == 'unigram':
    ngram_range = (1, 1)
elif args.ngram == 'bigram':
    ngram_range = (2, 2)
elif args.ngram == 'trigram':
    ngram_range = (3, 3)
else:
    raise ValueError("Invalid n-gram choice. Choose from 'unigram', 'bigram', or 'trigram'.")

# Define output file names based on the n-gram type
output_dir = 'outputs_eng_logreg'
os.makedirs(output_dir, exist_ok=True)
file_prefix = os.path.join(output_dir, args.ngram)

# TF-IDF Vectorization with specified N-grams
tfidf = TfidfVectorizer(max_features=5000, ngram_range=ngram_range)
X = tfidf.fit_transform(filtered_data['text_en'])
y = filtered_data['label'].apply(lambda x: 1 if x == 'Indoctrination' else 0)
feature_names = tfidf.get_feature_names_out()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
clf = LogisticRegression(random_state=0).fit(X_train, y_train)

# Predictions and Evaluation
y_pred = clf.predict(X_test)

# Calculate accuracy and display classification report
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Neutral', 'Indoctrination'])

# Save confusion matrix plot
plt.figure(figsize=(8, 6))
disp.plot()
plt.title('Confusion Matrix')
plt.savefig(f"{file_prefix}_confusion_matrix.png")
plt.close()

# Display model coefficients
coefficients = clf.coef_.tolist()[0]
feature_coeff = {feature_names[i]: coefficients[i] for i in range(len(feature_names))}
feature_sorted = sorted(feature_coeff.items(), key=itemgetter(1), reverse=True)

# Top 10 Positive and Negative Features
top_positive_features = feature_sorted[0:11]
top_negative_features = feature_sorted[-10:]

# Generate and save feature importance chart
def plot_feature_importance(features, title, filename):
    plt.figure(figsize=(10, 6))
    plt.barh([feature for feature, _ in features], [coef for _, coef in features], color='skyblue')
    plt.title(title)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_feature_importance(top_positive_features, 'Top 10 Positive Features', f"{file_prefix}_top_positive_features.png")
plot_feature_importance(top_negative_features, 'Top 10 Negative Features', f"{file_prefix}_top_negative_features.png")

# Display results in tabular form using tabulate
print("\nTop 10 Positive Features:")
print(tabulate(top_positive_features, headers=['Feature', 'Coefficient'], tablefmt='grid'))

print("\nTop 10 Negative Features:")
print(tabulate(top_negative_features, headers=['Feature', 'Coefficient'], tablefmt='grid'))

# Save feature coefficients
with open(f"{file_prefix}_feature_coefficients.txt", 'w') as f:
    f.write("Top 10 Positive Features:\n")
    f.write(tabulate(top_positive_features, headers=['Feature', 'Coefficient'], tablefmt='grid'))
    f.write("\n\nTop 10 Negative Features:\n")
    f.write(tabulate(top_negative_features, headers=['Feature', 'Coefficient'], tablefmt='grid'))

# Save predictions
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions_df.to_csv(f"{file_prefix}_predictions.csv", index=False)

# Report summary
print("\nReport Summary:")
print(tabulate(predictions_df.head(10), headers='keys', tablefmt='grid'))  # Display first 10 predictions
print(f"\nAccuracy: {accuracy:.2f}")

# Save report summary to a file
with open(f"{file_prefix}_report_summary.txt", 'w') as f:
    f.write(tabulate(predictions_df, headers='keys', tablefmt='grid'))
    f.write(f"\n\nAccuracy: {accuracy:.2f}\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))
