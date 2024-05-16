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

    # set model type
    if args.model == "Unigram":
        model = UnigramModel(args.smoothing)
    elif args.model == "Bigram":
        model = BigramModel(args.smoothing)
    elif args.model == "Trigram":
        model = TrigramModel(args.smoothing)
    else:
        raise Exception("Pass Unigram, Bigram, or Trigram to --model")
    
    print(args)

    # load data and train model
    with open('data/1b_benchmark.train.tokens', 'r', encoding='utf8') as train_file:
        model.train(train_file)
    
    # load dev set and calculate perplexity
    with open("data/1b_benchmark.dev.tokens", 'r', encoding='utf8') as test_file:
        model.perplexity(test_file)

    # use debug set as frame of reference
    with open("data/debug.tokens", 'r', encoding='utf8') as debug_file:
        perplexity = model.perplexity(debug_file)

    print("Perplexity: ", perplexity)


if __name__ == '__main__':
    main()