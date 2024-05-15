from models import *
import argparse

def main():
    # Get command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='Unigram',
                        choices=['Unigram', 'Bigram', 'Trigram'])
    parser.add_argument('--smoothing', '-s', type=int, default=0)
    args = parser.parse_args()

    # Ensure smoothing argument is non-negative
    if args.smoothing < 0:
        print("Negative number passed to --smoothing; defaulting to 0")
        args.smoothing = 0

    # Load data
    with open('1b_benchmark.train.tokens', 'r', encoding='utf8') as train_file:
        train_data = train_file.read()
    with open("1b_benchmark.dev.tokens", 'r', encoding='utf8') as test_file:
        test_data = test_file.read()

    if args.model == "Unigram":
        model = UnigramModel(args.smoothing)
    elif args.model == "Bigram":
        model = BigramModel(args.smoothing)
    elif args.model == "Trigram":
        model = TrigramModel(args.smoothing)
    else:
        raise Exception("Pass Unigram, Bigram, or Trigram to --model")

    model.train(train_data)
    model.perplexity(test_data)

    '''model = UnigramModel()
    model.train(traindata)
    model.smoothperplexity(testdata, 1)
    model = BigramModel()
    model.train(train_data)
    model.perplexity(test_data)
    '''

if __name__ == '__main__':
    main()