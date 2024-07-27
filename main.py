from models import *
import argparse

def main():
    # Get command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='Unigram',
                        choices=['Unigram', 'Bigram', 'Trigram'])
    parser.add_argument('--smoothing', '-s', type=float, default=0)
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
            train_file.seek(0)
            print("train perplexity: " + str(model.perplexity(train_file)))
    
    # load dev set and calculate perplexity
    with open("data/1b_benchmark.dev.tokens", 'r', encoding='utf8') as dev_file:
            print("dev perplexity: " + str(model.perplexity(dev_file)))


    # use debug set as frame of reference
    with open("data/debug.tokens", 'r', encoding='utf8') as debug_file:
            print("debug perplexity: " + str(model.perplexity(debug_file)))


if __name__ == '__main__':
    main()
