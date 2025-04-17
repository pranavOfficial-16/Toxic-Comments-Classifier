# Toxic Comments Classifier

### Project Overview

This project aims to build a machine learning model capable of classifying Wikipedia comments into six different categories of toxicity: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. It tackles this as a multi-label classification problem, where each comment can be assigned one or more toxicity labels. The final model outputs the probability of a comment belonging to each specific toxicity type.

### Data

The dataset originates from the Jigsaw Toxic Comment Classification Challenge on Kaggle and contains comments from Wikipedia talk pages, pre-labeled by human raters for each of the six toxicity categories.

### Workflow

1.  **Setup:** Imported necessary libraries including `pandas`, `nltk`, `re`, `seaborn`, `matplotlib`, and `scikit-learn`. Loaded the training data (`train.csv`).
2.  **Exploratory Data Analysis (EDA):**
    *   Checked data integrity (no missing values).
    *   Examined raw comment text to identify cleaning needs (e.g., newline characters, punctuation).
    *   Analyzed the distribution of labels, revealing a significant class imbalance, with most comments being non-toxic and some categories (like `threat`) being very rare.
3.  **Text Preprocessing:** Cleaned the `comment_text` by:
    *   Removing numbers and words containing numbers.
    *   Removing punctuation.
    *   Converting text to lowercase.
    *   Removing newline characters.
    *   Removing non-ASCII characters.
4.  **Handling Class Imbalance:** To address the imbalance identified in EDA, created separate, smaller, more balanced datasets for training *each* toxicity label individually. This involved undersampling the majority (non-toxic) class or ensuring a minimum representation (e.g., 20%) for very rare minority (toxic) classes.
5.  **Feature Extraction:** Utilized `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) to convert the preprocessed text comments into numerical feature vectors suitable for machine learning models. Unigrams (`ngram_range=(1,1)`) were used.
6.  **Model Training & Evaluation:**
    *   Trained several standard classification algorithms for *each* toxicity label on its respective balanced dataset:
        *   Logistic Regression
        *   K-Nearest Neighbors (KNN)
        *   Bernoulli Naive Bayes
        *   Multinomial Naive Bayes
        *   Linear Support Vector Classifier (SVC)
        *   Random Forest Classifier
    *   Evaluated the models using the F1 score, which is suitable for imbalanced datasets.
7.  **Model Comparison:** Compared the F1 scores across all models and all toxicity labels. LinearSVC and Random Forest generally showed the best performance.
8.  **Prediction:** Demonstrated how to use a trained model (Random Forest chosen for its `predict_proba` capability) along with its corresponding fitted `TfidfVectorizer` to predict the probability of new, unseen comments belonging to a specific toxicity category.
9.  **Model Persistence (Pickling):** Implemented functionality (using `pickle`) to save the trained `TfidfVectorizer` and the chosen `RandomForestClassifier` model for each toxicity label. This allows the models to be easily loaded and reused for future predictions without the need for retraining.

### Conclusion

The project successfully preprocesses text data, addresses class imbalance, trains and evaluates multiple models for multi-label toxic comment classification, and demonstrates prediction on new data. The Random Forest model, along with the TF-IDF vectorizer, was selected and prepared for persistence, enabling efficient reuse.

