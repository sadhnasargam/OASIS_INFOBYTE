# Task 5: Autocomplete and Autocorrect Data Analytics

## Project Overview

This project analyses the efficiency of Autocomplete and Autocorrect systems using Natural Language Processing (NLP).

A real text corpus was cleaned and processed to build:

- Frequency-based N-gram autocomplete
- Edit-distance-based autocorrect
- Performance evaluation and visualizations

---

## Objectives

- Perform NLP preprocessing on a text corpus.
- Build a Bigram/Trigram-based autocomplete model.
- Generate Top 3 predictions for at least 10 input prefixes.
- Correct at least 20 deliberately misspelled words.
- Calculate accuracy, precision and recall.
- Compare different approaches.
- Visualize frequent words and autocorrect performance.

---

## TechNOLOGY USE

"Python"
"Pandas" 
"NLTK" 
"Matplotlib"
"scikit-learn"
"PySpellChecker" 
"Collections" 
"Matplotlib" 


## Dataset

A large publicly available text corpus was used for training and testing the NLP models.

The text was cleaned using:

- Lowercasing
- Tokenization
- Punctuation removal
- Stopword removal

### Preprocessing Screenshot

"NLP Preprocessing" (images/preprocessing.png)

---

### Autocomplete

A frequency-based N-gram model was implemented to predict the next word.

The model was tested on 10+ input prefixes and the Top 3 predictions were displayed.

### Autocomplete Results

![alt text](<Top 15 Most Common Words.png>)

### Autocorrect

An edit-distance-based approach was used to identify and correct spelling mistakes.

The model was tested on 20+ deliberately misspelled words.

---

### Performance Evaluation

The systems were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score

### Performance / Results

![alt text](<Performance.png>)

---

### Visualizations

**Top 20 Most Frequent Words**

![alt text](<Top 20 Most Frequent Words.png>)

**Top 10 Most Common Words**

![alt text](<Top 10 Most Common Words.png>)

**Top 15 Most Common Words**

![alt text](<Top 15 Most Common Words.png>)

**Top 15 Most Frequent Words**

![alt text](<Top 15 Words.png>)

---
### Limitations

The implementation has limited contextual understanding and depends heavily on the training corpus.

Unlike modern systems such as Google Keyboard, it does not use:

- Deep Learning
- Transformers
- Personalised vocabulary
- Advanced context understanding
- Large Language Models

---

### Future Scope

- LSTM/Transformer-based autocomplete
- Context-aware autocorrect
- Personalised predictions
- Multilingual support
- Real-time keyboard integration
---

## Author

Sadhna Kumari


