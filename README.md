## N-gram modeling

### Goal

Our goal for this project is for our N-gram model to have a low perplexity score, proving that our model is very efficient in predicting how a story unfolds based on existing information.

### Overview

This document provides an overview of the N-gram modeling project we worked on under Professor Jeffrey Flanagan which uses additive smoothing and linear interpolation to calculate the perplexity scores of various stories.

- In this project, we calculated the perplexity scores of the unigram, bigram, and trigram language models for our training, development, and test sets.

- We used additive smoothing values 1, 0.5, and 0.1 to analyze how various smoothing values affect the perplexity values of the unigram, bigram, and trigram language models

- We applied hyperparameter tuning on various lambda values in a linear interpolation equation to analyze how various smoothing values affect the perplexity values of the unigram, bigram, and trigram language models

### Source Datasets

The datasets we used are .tokens files 1b_benchmark.dev, 1b_benchmark.test, and 1b_benchmark.train. 
The data has been pre-split into training, dev, and test splits into data/, with a CSV file for each split.

- The 1b_benchmark.train file provided to us amassed 65,000 lines and is a summary of various news reports about the social, cultural, and political events that occurred during the years that Obama's presidency influenced that. 

- The 1b_benchmark.test and 1b_benchmark.dev files are .tokens files with a length of approximately 12,000 lines each that are summaries from different news sources on the social, cultural, and political events that occurred during the years that were influenced by Obama's presidency. 

- The 1b_benchmark.train file is significantly bigger than the 1b_benchmark.test and 1b_benchmark.dev files as this helped expose the model to various clusters of text from which the models can retrieve perplexity scores.

- The larger .train file helped the N-gram model understand the various stories narrated in the smaller .dev and .test files as it gave the model familiarity with what the social, cultural, and political stories are about to fully grasp the context behind these social, cultural, and political events. This helps the model predict how any event unfolds based on the context given.

- To predict the next event or set of events in a story, the model predicts the next words and phrases that would be implemented to describe the story by using unigrams (to predict the next word), bigrams (to predict the next two words), and trigrams (the next three words).

### Running the Model

#### To run any model (Unigram, Bigram, or Trigram), you use the command:
- "python main.py -m Unigram" for running the Unigram model
  <img width="686" alt="Screenshot 2024-08-14 at 7 42 13 PM" src="https://github.com/user-attachments/assets/c79bab11-89fe-423e-b92e-16a7573c4795">
  
- "python main.py -m Bigram" for running the Bigram model
  <img width="658" alt="Screenshot 2024-08-14 at 7 42 53 PM" src="https://github.com/user-attachments/assets/1022067b-0298-4590-9ef1-603f53e51070">

- "python main.py -m Trigram" for running the Trigram model
  <img width="673" alt="Screenshot 2024-08-14 at 7 43 22 PM" src="https://github.com/user-attachments/assets/33958b41-c039-4d2c-9ab8-c5f386d22524">

#### To run the models using additive smoothing, you use the command:
- "python main.py -m Unigram -s [smoothing value] " for running the Unigram model
  <img width="706" alt="Screenshot 2024-08-14 at 7 43 58 PM" src="https://github.com/user-attachments/assets/a71134e9-a04b-4a3d-9261-a3850c2b5af9">

- "python main.py -m Bigram -s [smoothing value] " for running the Bigram model
  <img width="717" alt="Screenshot 2024-08-14 at 7 44 36 PM" src="https://github.com/user-attachments/assets/6fab5ce0-634a-450a-951c-1dc6af94af0b">

- "python main.py -m Tigram -s [smoothing value] " for running the Trigram model
<img width="727" alt="Screenshot 2024-08-14 at 7 45 05 PM" src="https://github.com/user-attachments/assets/34f391c1-0c13-4b65-b21a-5994e900845d">

- The smoothing values we implemented are 1, 0.5, and 0.1
- Hence, if we were to run the Bigram model using a smoothing value of 0.5, the command would look like: 
"python main.py -m Bigram -s 0.5"

#### Linear Interpolation
Here's the Linear Interpolation formula where the sum of the lambda hyperparameter variables equal the interpolation value and theta is the smoothed parameters after linear interpolation:
<img width="475" alt="Screenshot 2024-08-14 at 7 58 40 PM" src="https://github.com/user-attachments/assets/d8829cc2-8ff1-4a57-80b8-48780032e1b5">

- The interpolation value can only be 1 as the hyperparameter values should sum to 1. If any other interpolation value is put, then the program gives this error:
<img width="938" alt="Screenshot 2024-08-14 at 8 04 42 PM" src="https://github.com/user-attachments/assets/c7b769c2-65e1-4e27-9dd2-c8e905b1eb91">
  
- "python main.py -m Unigram -i 1" for running the Unigram model
  <img width="724" alt="Screenshot 2024-08-14 at 8 03 19 PM" src="https://github.com/user-attachments/assets/7c88846f-c1ee-4c69-8aa4-c6ca51a7b4a9">

- "python main.py -m Bigram -i 1" for running the Bigram model
  <img width="715" alt="Screenshot 2024-08-14 at 8 00 54 PM" src="https://github.com/user-attachments/assets/947225a1-216b-4b4c-b09f-c2cd9ad69fc1">

- "python main.py -m Tigram -i 1" for running the Trigram model
<img width="726" alt="Screenshot 2024-08-14 at 8 01 23 PM" src="https://github.com/user-attachments/assets/04c09e70-276f-4c4c-a87b-952f4789e88d">

- If we were to run the Bigram model using a smoothing value of 1 with the hyperparameters 0.6, 0.3, and 0.1 (the sum is 1), the command would look like: 
"python main.py -m Bigram -i 1"
  <img width="715" alt="Screenshot 2024-08-14 at 8 00 54 PM" src="https://github.com/user-attachments/assets/947225a1-216b-4b4c-b09f-c2cd9ad69fc1">


