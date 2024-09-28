# Financial Market News Sentiment Analysis

This project aims to analyze the sentiment of top 25 financial market news headlines. The sentiment prediction model helps to gauge market mood, which can potentially guide investment decisions.

## Project Overview

The goal of this project is to classify news headlines as either positive, negative, or neutral based on their sentiment. We used Natural Language Processing (NLP) techniques for feature extraction and a machine learning model for classification.

### Key Steps:
1. **Data Import and Description**: We start by loading the dataset containing the top 25 financial market headlines of the day.
2. **Feature Selection**: We preprocess the text and prepare it for machine learning using CountVectorizer to convert headlines into a bag-of-words model.
3. **Train-Test Split**: The dataset is split into training and testing sets to evaluate model performance.
4. **Model Training**: A RandomForestClassifier is used to train the model and learn from the news headline data.
5. **Model Evaluation**: After training, the model is evaluated using:
   - `classification_report`: To get precision, recall, and F1-score.
   - `confusion_matrix`: To understand the distribution of predicted vs actual classes.
   - `accuracy_score`: To assess the overall accuracy of the model.

## Tools and Libraries
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: For machine learning and evaluation tools:
  - `CountVectorizer` for converting text data into numerical form
  - `RandomForestClassifier` for classification
  - `classification_report`, `confusion_matrix`, and `accuracy_score` for model evaluation
- **Numpy**: For numerical computations
- **Matplotlib/Seaborn**: For visualization of results (optional)

## Dataset

The dataset consists of the top 25 daily financial market news headlines. The features include:
- **Headlines**: Financial market news.
- **Sentiment**: The target variable, categorized into `positive`, `negative`, or `neutral`.

## Project Workflow

1. **Data Preprocessing**:
   - Imported data and cleaned it (if necessary).
   - Transformed the text headlines into a bag-of-words model using `CountVectorizer`.

2. **Feature Extraction**:
   - Applied `CountVectorizer` to convert the headlines into numerical form suitable for machine learning.

3. **Model Building**:
   - Trained a RandomForestClassifier using the processed features.

4. **Evaluation**:
   - Used `classification_report` for detailed performance metrics such as precision, recall, and F1-score.
   - Generated a `confusion_matrix` to visualize prediction accuracy and misclassifications.
   - Measured the overall accuracy of the model using `accuracy_score`.

## Results

The model's performance was assessed based on its ability to accurately classify the sentiment of the headlines. The confusion matrix and classification report provided insights into precision, recall, and F1-scores across all classes.

## Conclusion

This project demonstrates how machine learning can be used to predict the sentiment of financial market news headlines, providing valuable insight into market mood that can potentially influence investment strategies.

## Future Work

- **Hyperparameter Tuning**: Experiment with different machine learning models and hyperparameters to improve performance.
- **Larger Dataset**: Train the model on a more extensive dataset to enhance the model's robustness.
- **Sentiment Categories**: Explore fine-tuned sentiment categories (e.g., strong positive, mild positive) for more nuanced analysis.
