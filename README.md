<div align="center">
<h1 align="center">Sentiment Analysis with NLTK</h1>

  <p align="center">
    Classifies text sentiment using various NLTK and scikit-learn classifiers.
  </p>
</div>

## Table of Contents

  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#key-features">Key Features</a></li>
      </ul>
    </li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>

## About The Project

This project performs sentiment analysis on text data using a combination of NLTK (Natural Language Toolkit) and scikit-learn classifiers. It trains multiple classifiers, including Naive Bayes, MultinomialNB, BernoulliNB, Logistic Regression, SGDClassifier, LinearSVC, and NuSVC, and then combines their predictions using a voting classifier to improve accuracy. The script also includes functionality for feature extraction, document processing, and accuracy evaluation.

### Key Features

- **Multiple Classifiers:** Utilizes several NLTK and scikit-learn classifiers for sentiment analysis.
- **Voting Classifier:** Combines the predictions of multiple classifiers to improve overall accuracy.
- **Feature Extraction:** Extracts relevant features from text data using word tokenization and part-of-speech tagging.
- **Accuracy Evaluation:** Evaluates the accuracy of individual classifiers and the voting classifier using a testing set.
- **Customizable Word Types:** Allows customization of allowed word types for feature extraction (e.g., adjectives, adverbs, verbs).
- **Pickle Serialization:** Saves processed documents and word features using pickle for later use.

## Built With

- [Python](https://www.python.org/)
- [NLTK](https://www.nltk.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Statistics Module](https://docs.python.org/3/library/statistics.html)
- [Pickle Module](https://docs.python.org/3/library/pickle.html)

## Getting Started

To get started with this project, you need to have Python and the required libraries installed. Follow the instructions below to set up the environment and run the script.

### Prerequisites

- Python (3.6 or higher)
- NLTK
- scikit-learn
- statistics

You can install the required libraries using pip:

  ```sh
  pip install nltk scikit-learn
  ```

Additionally, you need to download the `movie_reviews` corpus and punkt tokenizer for NLTK:

  ```python
  import nltk
  nltk.download('movie_reviews')
  nltk.download('punkt')
  nltk.download('averaged_perceptron_tagger')
  ```

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/surjahead/sentimentanalysis.git
   ```

2. Navigate to the project directory:
   ```sh
   cd sentimentanalysis
   ```

3.  **Important**: Modify the file paths for `short_pos` and `short_neg` to point to your local positive and negative text files.  The current paths `E:/positive.txt` and `E:/negative.txt` are unlikely to work without modification.  Create these files or obtain them from a suitable source.

4. Run the SentimentAnalysis.py script:
   ```sh
   python surjahead-sentimentanalysis/SentimentAnalysis.py
   ```

## Acknowledgments
- This project was a result of following sentdex's "NLTK with Python3 for Natural Language Processing" tutoral. Check it out [here](https://www.youtube.com/playlist?list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL).

