# Author: Juan Gabriel Cespedes Pacheco

import os
import pandas as pd
from data_loader import load_data, get_available_files

# Show available files
print("Available review files:")
available_files = get_available_files()
for file_path, category, label in available_files:
    print(f"  {file_path} -> category: {category}, label: {label}")
print()

# Load labeled data
print("Loading labeled data...")
texts, labels, categories = load_data(include_unlabeled=False)

# Convert to DataFrame
labeled_data = []
for text, label, category in zip(texts, labels, categories):
    labeled_data.append({
        'review_text': text,
        'label': label,
        'category': category,
        'rating': 5.0 if label == 'positive' else 1.0  # Approximate rating based on label
    })

full_df = pd.DataFrame(labeled_data)
print(f"Loaded {len(full_df)} labeled reviews")

# Load unlabeled data
print("\nLoading unlabeled data...")
unlabeled_texts, unlabeled_labels, unlabeled_categories = load_data(include_unlabeled=True)

# Filter to get only unlabeled data
unlabeled_data = []
for text, label, category in zip(unlabeled_texts, unlabeled_labels, unlabeled_categories):
    if label == 'unlabeled':
        unlabeled_data.append({
            'review_text': text,
            'category': category
        })

if unlabeled_data:
    unlabeled_df = pd.DataFrame(unlabeled_data)
    print(f"Loaded {len(unlabeled_df)} unlabeled reviews")
else:
    print("Warning: No unlabeled datasets were loaded.")

# Shuffle the dataset randomly
full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

# CLEANING: Keep only relevant columns for sentiment classification
full_df = full_df[["category", "rating", "review_text", "label"]]

full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)


# Calculate word count per review
full_df["word_count"] = full_df["review_text"].str.split().apply(len)

# Generate a full descriptive summary including percentiles up to 90%
word_count_summary = full_df["word_count"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])

print("\nDescriptive statistics for review word counts:")
print(word_count_summary)

import matplotlib.pyplot as plt

# Plot the distribution of word counts per review
plt.figure(figsize=(10, 6))
plt.hist(full_df["word_count"], bins=50, color='skyblue', edgecolor='black')
plt.title("Distribution of Word Count per Review")
plt.xlabel("Number of Words")
plt.ylabel("Number of Reviews")
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.show()

#### Cleaing Outliers
Q1 = full_df["word_count"].quantile(0.25)
Q3 = full_df["word_count"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

filtered_df = full_df[(full_df["word_count"] >= lower_bound) & (full_df["word_count"] <= upper_bound)]

# Display the number of reviews after filtering
print(f"\nNumber of reviews after removing outliers: {len(filtered_df)}")

# Descriptive statistics for the filtered dataset
filtered_summary = filtered_df["word_count"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])

print("\nDescriptive statistics for word counts (outliers removed):")
print(filtered_summary)

# # Count missing (NaN or empty string) values in each column
# missing_counts = full_df.isna().sum() + (full_df == "").sum()

# print("\nMissing values per column:")
# print(missing_counts)

# prints a review
# print("\nReview #40000:")
# for column, value in full_df.iloc[7999].items():  # Index starts at 0
#     print(f"{column}: {value}")

