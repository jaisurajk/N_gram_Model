from models import *

def main():
    traindata = open('1b_benchmark.train.tokens', 'r', encoding='utf8')
    testdata = open("1b_benchmark.dev.tokens", "r", encoding='utf8')
    '''model = UnigramModel()
    model.train(traindata)
    model.smoothperplexity(testdata, 1)'''
    model = BigramModel()
    model.train(traindata)
    model.perplexity(testdata)

if __name__ == '__main__':
    main()