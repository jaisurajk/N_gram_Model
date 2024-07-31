## N-gram modeling

### Goal

Our goal for this project is for our N-gram model to have a low perplexity score which proves that our model is very efficient in predicting how a story unfolds based on existing information given to it.

### Overview

This document provides an overview of the N-gram modeling project we worked on under Professor Jeffrey Flanagan that uses additive smoothing and linear interpolation to calculate the perplexity scores of various stories.

- In this project, we calculated the preplexity scores of the unigram, bigram, and trigram language models for our training, development, and test sets.

- We used additive smoothing values 1, 0.5, and 0.1 to analyze how various smoothing values affect the perplexity values of the unigram, bigram, and trigram language models

- We applied hyperparameter tuning on various lambda values in a linear interpolation equation to analyze how various smoothing values affect the perplexity values of the unigram, bigram, and trigram language models

### Source Datasets

The datasets we used are .tokens files 1b_benchmark.dev, 1b_benchmark.test, and 1b_benchmark.train. 

- The 1b_benchmark.train file provided to us amassed 65,000 lines is a summary of various news reports about the social, cultural, and political events that occurred during the years that that was influcened by Obama's presidency. 

- The 1b_benchmark.test and 1b_benchmark.dev files are .tokens files with a length of approximate 12,000 lines each that are summaries from different new sources on the social, cultural, and political events that occurred during the years that that was influcened by Obama's presidency. 

- The 1b_benchmark.train file is significantly bigger than the 1b_benchmark.test and 1b_benchmark.dev files as this helped expose the model to various clusters of text in which the models can retrieve perplexity scores from.

- The larger .train file helped the N-gram model understand the various stories narrated in the smaller .dev and .test files as it gave the model familiarity with what the social, cultural, and political stories are about to fully grasp the context behind these social, cultural, and political events. This helps the model predict how any event unfolds based on the context given.

- In order to predict the next event or set of events in a story, the model predicts the next words and phrases that would be implemented to describe the story by using unigrams (to predict the next word), bigrams (to predict the next two words), and trigrams (the next three words).

### Running the Model

#### To run any model (Unigram, Bigram, or Trigram), you use the command:
- "python main.py -m Unigram" for running the Unigram model
- "python main.py -m Bigram" for running the Bigram model
- "python main.py -m Trigram" for running the Trigram model

#### To run the models using additive smoothing, you use the command:
- "python main.py -m Unigram -s [smoothing value] " for running the Unigram model
- "python main.py -m Bigram -s [smoothing value] " for running the Bigram model
- "python main.py -m Tigram -s [smoothing value] " for running the Trigram model

- The smoothing values we implemented are 1, 0.5, and 0.1
- Hence, if we were to run the Bigram model using a smoothing value of 0.5, the command would look like: 
"python main.py -m Bigram -s 0.5"

#### Linear Interpolation

- We implemnted linear interpolation for the trigram model
- To run the model on linear interpolation, we used the command:
"python main.py -m Trigram -i 1"

