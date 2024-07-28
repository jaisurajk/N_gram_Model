from collections import Counter, defaultdict
import math

class UnigramModel:
    def __init__(self, alpha=0):
        self.token_counts = {}
        self.totalTokens = 0
        self.alpha = alpha
        self.vocab_size = 0
    
    def train(self, file_stream):
        raw_counts = defaultdict(int)
        #counts all the tokens, adds STOPs
        for line in file_stream:
            tokens = line.strip().split() + ["<STOP>"]
            for token in tokens:
                raw_counts[token] += 1
        #replace rare tokens with UNK
        unk_count = 0
        for token, count in raw_counts.items():
            if count < 3:
                unk_count += count
            else:
                self.token_counts[token] = count

        if unk_count > 0:
            self.token_counts["<UNK>"] = unk_count

        self.vocab_size = len(self.token_counts)
        self.totalTokens = sum(self.token_counts.values()) 

    #calculates the unigram log probability for a token
    def unigram_mle(self, token):
        count = self.token_counts.get(token, self.token_counts.get("<UNK>", 0)) + self.alpha
        return count/(self.totalTokens + self.alpha * self.vocab_size)
    
    def perplexity(self, file_stream):
        prob = 0
        token_count = 0
        for line in file_stream:
            tokens = line.split() + ["<STOP>"]
            for token in tokens:
                token_count += 1 
                prob += math.log(self.unigram_mle(token))
        return math.exp(-prob / token_count)
    
class BigramModel:
    def __init__(self, alpha=0):
        self.alpha = alpha
        self.initial = {}
        self.finalTokenCount = defaultdict(int)
        self.totalTokens = 0
        self.finalBigramCount = defaultdict(lambda: defaultdict(int))

    def train(self, file):
        stop = 0
        unk = 0
        #get initial token counts
        for i in file:
            tokens = i.split()
            for j in tokens:
                if j in self.initial:
                    self.initial[j] += 1
                else:
                    self.initial[j] = 1
            stop += 1
        #replace rare tokens with UNK
        for i in self.initial:
            if self.initial[i] >= 3:
                self.finalTokenCount[i] = self.initial[i]
                self.totalTokens += self.initial[i]
            else:
                unk += self.initial[i]
        self.finalTokenCount.update({"<UNK>":unk})
        self.finalTokenCount.update({"<STOP>":stop})
        file.seek(0)
        #get bigram counts
        for i in file:
            tokens = i.split() + ["<STOP>"]
            prev = "<START>"
            for j in tokens:
                if prev == "<START>":
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
        self.totalTokens = self.totalTokens + unk + stop

    def bigram_mle(self, token, prev, adjusted_total):
        #case for token at the start of the sentence
        if prev == "<START>":
            if token in self.finalBigramCount["<START>"]:
                return (self.finalBigramCount["<START>"][token] + self.alpha)/(self.finalTokenCount["<STOP>"] + adjusted_total)
            else: #if the bigram hasn't been seen before
                if self.alpha != 0: #smoothing if enabled
                    return self.alpha/(self.finalTokenCount["<STOP>"] + adjusted_total)
        #case for rest of the tokens
        else:
            if prev in self.finalBigramCount and token in self.finalBigramCount[prev]:
                return (self.finalBigramCount[prev][token] + self.alpha)/(self.finalTokenCount[prev] + adjusted_total)
            else: #if the bigram hasn't been seen before
                if self.alpha != 0: #smoothing if enabled
                    return self.alpha/(self.finalTokenCount[prev] + adjusted_total)
        return 0 
                
    def perplexity(self, file):
        prob = 0
        token_ct = 0
        adjusted_total = self.alpha * len(self.finalTokenCount)
        for i in file:
            tokens = i.split() + ["<STOP>"]
            token_ct += len(tokens) 
            prev = "<START>"
            for token in tokens:
                if token not in self.finalTokenCount:
                    token = "<UNK>"
                if self.bigram_mle(token, prev, adjusted_total) != 0:
                    prob += math.log(self.bigram_mle(token, prev, adjusted_total))
                prev = token
        return math.exp(-prob/(token_ct))

class TrigramModel:
    def __init__(self, alpha=0, interpolate=0):
        self.initial = Counter()
        self.finalTokenCount = defaultdict(int)
        self.finalBigramCount = defaultdict(lambda: defaultdict(int))
        self.finalTrigramCount = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.totalTokens = 0
        self.alpha = alpha
        self.interpolate = False
        if interpolate:
            self.Unigram = UnigramModel()
            self.Bigram = BigramModel()
            self.interpolate = True
            print("interpolated perplexity")

    def train(self, sentences):
        if self.interpolate:
            self.Unigram.train(sentences)
            sentences.seek(0)
            self.Bigram.train(sentences)
            sentences.seek(0)
        stop = 0
        unk = 0
        #get initial token counts
        for sentence in sentences:
            tokens = sentence.split()
            self.initial.update(tokens)
            stop += 1
        #replace rare tokens with UNK
        for token, count in self.initial.items():
            if count >= 3:
                self.finalTokenCount[token] = count
                self.totalTokens += count
            else:
                unk += count
        self.finalTokenCount["<UNK>"] = unk
        self.finalTokenCount["<STOP>"] = stop
        self.totalTokens += unk + stop
        sentences.seek(0)
        #get trigram and bigram counts
        for sentence in sentences:
            tokens = ["<START>"] + sentence.split() + ["<STOP>"]
            for i in range(len(tokens) - 2):
                first = tokens[i]
                second = tokens[i+1] if tokens[i+1] in self.finalTokenCount else "<UNK>"
                third = tokens[i+2] if tokens[i+2] in self.finalTokenCount else "<UNK>"
                if first == "<START>":
                    self.finalTrigramCount["<START>"][second][third] += 1
                    self.finalBigramCount["<START>"][second] += 1
                else:
                    first = tokens[i] if tokens[i] in self.finalTokenCount else "<UNK>"
                    self.finalTrigramCount[first][second][third] += 1
                    self.finalBigramCount[first][second] += 1

    def trigram_mle(self, tokens, i, adjusted_total):
        first, second = (tokens[i], tokens[i+1])
        third = tokens[i+2] if tokens[i+2] in self.finalTokenCount else "<UNK>"
        # for the probability of the token immediately following <START> in the trigram model, use the bigram probability
        if first == "<START>" and second == "<START>": #<START><START><X>
            if self.finalBigramCount["<START>"][third] != 0:
                return (self.finalBigramCount[second][third]+self.alpha) / (self.finalTokenCount["<STOP>"]+adjusted_total)
            else:
                if self.alpha != 0: #smoothing if enabled
                    return self.alpha/(self.finalTokenCount["<STOP>"]+adjusted_total)
        else:
            second = tokens[i+1] if tokens[i+1] in self.finalTokenCount else "<UNK>"
            if first == "<START>": #<START><X><Y>
                if self.finalTrigramCount[first][second][third] != 0 and self.finalBigramCount[first][second] != 0:
                    return (self.finalTrigramCount[first][second][third]+self.alpha) / (self.finalBigramCount[first][second]+adjusted_total)
                else:
                    if self.alpha != 0: #smoothing if enabled
                        return self.alpha/(self.finalBigramCount[first][second]+adjusted_total)
            else: #<X><Y><Z>
                first = tokens[i] if tokens[i] in self.finalTokenCount else "<UNK>"
                if self.finalTrigramCount[first][second][third] != 0 and self.finalBigramCount[first][second] != 0:
                    return (self.finalTrigramCount[first][second][third]+self.alpha) / (self.finalBigramCount[first][second]+adjusted_total)
                else:
                    if self.alpha != 0: #smoothing if enabled
                        return self.alpha/(self.finalBigramCount[first][second]+adjusted_total)
        return 0
                    
    def perplexity(self, sentences):
        prob = 0
        token_ct = 0
        adjusted_total = self.alpha * len(self.finalTokenCount)
        for sentence in sentences:
            tokens = sentence.split()
            tokens = ["<START>", "<START>"] + tokens + ["<STOP>"]
            token_ct += len(tokens) - 2 #minus the 2 start tokens
            for i in range(len(tokens) - 2):
                if tokens[i+2] not in self.finalTokenCount:
                    tokens[i+2] = "<UNK>"
                if self.interpolate:
                    unigramle = self.Unigram.unigram_mle(tokens[i+2])
                    bigramle = self.Bigram.bigram_mle(tokens[i+2], tokens[i+1], adjusted_total)
                    trigramle = self.trigram_mle(tokens, i, adjusted_total)
                    if self.trigram_mle(tokens, i, adjusted_total) != 0 and unigramle != 0 and bigramle != 0:
                        prob += math.log(0.01*(unigramle) + 0.4*bigramle + 0.59*trigramle)
                else:
                    trigramle = self.trigram_mle(tokens, i, adjusted_total)
                    if trigramle != 0:
                        prob += math.log(self.trigram_mle(tokens, i, adjusted_total))
        return math.exp(-prob/token_ct)
