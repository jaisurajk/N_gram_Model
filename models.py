from collections import defaultdict
import math

class UnigramModel:
    def __init__(self):
        self.initial = {}
        self.finalTokenCount = {}
        self.totalTokens = 0
    
    def train(self, file):
        stop = 0
        unk = 0
        for i in file:
            tokens = i.split()
            for j in tokens:
                if j in self.initial:
                    self.initial[j] += 1
                else:
                    self.initial[j] = 1
            stop += 1
        for i in self.initial:
            if self.initial[i] >= 3:
                self.finalTokenCount[i] = self.initial[i]
                self.totalTokens += self.initial[i]
            else:
                unk += self.initial[i]
        self.finalTokenCount.update({"<UNK>":unk})
        self.finalTokenCount.update({"<STOP>":stop})
        self.totalTokens = self.totalTokens + unk + stop

    def perplexity(self, file):
        prob = 0
        ct = 0
        numStops = 0
        for i in file:
            tokens = i.split()
            numStops += 1
            for j in tokens:
                ct += 1
                if j in self.finalTokenCount:
                    prob += math.log(self.finalTokenCount[j] / self.totalTokens)
                else:
                    prob += math.log(self.finalTokenCount["<UNK>"] / self.totalTokens)
        prob += math.log(self.finalTokenCount["<STOP>"]/self.totalTokens)
        print(math.exp(-prob/(ct+numStops)))
    
    def smoothperplexity(self, file, alpha):
        prob = 0
        ct = 0
        numStops = 0
        for i in file:
            tokens = i.split()
            numStops += 1
            for j in tokens:
                ct += 1
                if j in self.finalTokenCount:
                    adjustedCount = (self.finalTokenCount[j] + alpha) * (self.totalTokens)
                    prob += math.log(adjustedCount/ (self.totalTokens * len(self.finalTokenCount)))
                else:
                    adjustedCount = (self.finalTokenCount["<UNK>"] + alpha) * (self.totalTokens)
                    prob += math.log(adjustedCount/ (self.totalTokens * len(self.finalTokenCount)))
        prob += math.log(((self.finalTokenCount["<STOP>"]+alpha)*self.totalTokens)/(self.totalTokens * len(self.finalTokenCount)))
        print(math.exp(-prob/(ct+numStops)))

class BigramModel:
    def __init__(self):
        self.initial = {}
        self.finalTokenCount = defaultdict(int)
        self.totalTokens = 0
        self.finalBigramCount = defaultdict(lambda: defaultdict(int))

    def train(self, file):
        stop = 0
        unk = 0
        for i in file:
            tokens = i.split()
            for j in tokens:
                if j in self.initial:
                    self.initial[j] += 1
                else:
                    self.initial[j] = 1
            stop += 1
        for i in self.initial:
            if self.initial[i] >= 3:
                self.finalTokenCount[i] = self.initial[i]
                self.totalTokens += self.initial[i]
            else:
                unk += self.initial[i]
        self.finalTokenCount.update({"<UNK>":unk})
        self.finalTokenCount.update({"<STOP>":stop})
        file.seek(0)
        for i in file:
            tokens = i.split()
            prev = None
            for j in tokens:
                if prev == None:
                    if j in self.finalTokenCount:
                        self.finalBigramCount["<START>"][j] += 1
                        prev = j
                    else:
                        self.finalBigramCount["<START>"]["<UNK>"] += 1
                        prev = "<UNK>"
                else:
                    if j in self.finalTokenCount:
                        self.finalBigramCount[prev][j] += 1
                        prev = j
                    else:
                        self.finalBigramCount[prev]["<UNK>"] += 1
                        prev = "<UNK>"
            self.finalBigramCount[prev]["<STOP>"] += 1
        self.totalTokens = self.totalTokens + unk + stop

    def perplexity(self, file):
        prob = 0
        ct = 0
        numStops = 0
        for i in file:
            tokens = i.split()
            numStops += 1
            prev = None
            for j in tokens:
                ct += 1
                if prev == None:
                    if j in self.finalBigramCount["<START>"]:
                        prob += math.log(self.finalBigramCount["<START>"][j]/self.finalTokenCount["<STOP>"])
                else:
                    if prev in self.finalBigramCount and j in self.finalBigramCount[prev]:
                        prob += math.log(self.finalBigramCount[prev][j]/self.finalTokenCount[prev])
                prev = j
            if self.finalTokenCount[prev] != 0 and self.finalBigramCount[prev]["<STOP>"] != 0:
                prob += math.log(self.finalBigramCount[prev]["<STOP>"]/self.finalTokenCount[prev])
        print(math.exp(-prob/(ct+numStops)))