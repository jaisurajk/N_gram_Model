from collections import Counter, defaultdict
import math

class UnigramModel:
    def __init__(self, alpha=0):
        self.token_counts = {}
        self.total_tokens = 0
        self.alpha = alpha
        self.vocab_size = 0
    
    def train(self, file_stream):
        # get token counts from training data
        raw_counts = defaultdict(int)
        for line in file_stream:
            tokens = line.strip().split() + ["<STOP>"]
            for token in tokens:
                raw_counts[token] += 1

        # save counts and convert low-frequency tokens to <UNK>
        unk_count = 0
        for token, count in raw_counts.items():
            if count < 3:
                unk_count += count
            else:
                self.token_counts[token] = count

        if unk_count > 0:
            self.token_counts["<UNK>"] = unk_count

        self.vocab_size = len(self.token_counts)
        self.total_tokens = sum(self.token_counts.values()) 

        # debug purposes
        print(self.total_tokens)
        print(self.vocab_size)
        print(len(self.token_counts))

    def perplexity(self, file_stream):
        prob = 0
        token_count = 0
        # recalculate total tokens accounting for smoothing (or lack thereof)
        adjusted_total = self.total_tokens + self.alpha * self.vocab_size
        for line in file_stream:
            tokens = line.strip().split() + ["<STOP>"]
            for token in tokens:
                token_count += 1 # keep track of total tokens in this eval set
                # get count of this token if present in train vocab, else it's <UNK>
                count = self.token_counts.get(token, self.token_counts.get("<UNK>", 0)) + self.alpha
                # add this tokens probability to total (via addtion since using logspace)
                prob += math.log(count / adjusted_total) 
        return math.exp(-prob / token_count)
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
    def __init__(self, alpha=0):
        self.alpha = alpha
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
                if j not in self.finalTokenCount:
                    j = "<UNK>"
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

    def smoothperplexity(self, file, alpha):
        log_prob = 0.0
        num_tokens = 0
        vocab_size = len(self.finalTokenCount)
    
        for line in file:
            tokens = ['<START>'] + line.split() + ['<STOP>']
            prev_token = '<START>'
            num_tokens += 1
    
            for curr_token in tokens[1:]:
                num_tokens += 1
                bigram_count = self.finalBigramCount[prev_token].get(curr_token, 0)
                unigram_count = self.finalTokenCount[prev_token]
    
                smoothed_prob = (bigram_count + alpha) / (unigram_count + alpha * vocab_size)
                log_prob += math.log(smoothed_prob)
    
                prev_token = curr_token
    
        perplexity = 2 ** (-log_prob / num_tokens)
        print(perplexity)

class TrigramModel:
    def __init__(self):
        self.initial = Counter()
        self.finalTokenCount = Counter()
        self.finalBigramCount = defaultdict(lambda: Counter())
        self.finalTrigramCount = defaultdict(lambda: defaultdict(lambda: Counter()))
        self.totalTokens = 0

    def train(self, sentences):
        """
        Train the model using a list of sentences.
        
        Args:
            sentences (list of str): List of sentences to train the model.
        """
        stop = 0
        for sentence in sentences:
            tokens = sentence.split()
            self.initial.update(tokens)
            stop += 1
        
        unk = 0
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
        for sentence in sentences:
            tokens = ["<START>", "<START>"] + sentence.split() + ["<STOP>"]
            for i in range(len(tokens) - 2):
                first = tokens[i]
                second = tokens[i+1]
                third = tokens[i+2] if tokens[i+2] in self.finalTokenCount else "<UNK>"
                if first == "<START>" and second == "<START>":
                    self.finalTrigramCount["<START>"]["<START>"][third] += 1
                    self.finalBigramCount[first][second] += 1
                elif first == "<START>" and second != "<START>":
                    second = tokens[i+1] if tokens[i+1] in self.finalTokenCount else "<UNK>"
                    self.finalTrigramCount["<START>"][second][third] += 1
                    self.finalBigramCount[first][second] += 1
                else:
                    first = tokens[i] if tokens[i] in self.finalTokenCount else "<UNK>"
                    second = tokens[i+1] if tokens[i+1] in self.finalTokenCount else "<UNK>"
                    self.finalTrigramCount[first][second][third] += 1
                    self.finalBigramCount[first][second] += 1

    def perplexity(self, sentences):
        """
        Calculate the perplexity of the model on a list of sentences.
        
        Args:
            sentences (list of str): List of sentences to calculate perplexity on.
        
        Returns:
            float: The perplexity value.
        """
        prob = 0
        totaltokens = 0
        for sentence in sentences:
            tokens = sentence.split()
            tokens = ["<START>", "<START>"] + tokens + ["<STOP>"]
            totaltokens += len(tokens) - 2 #minus the 2 start tokens

            for i in range(len(tokens) - 2):
                first, second = (tokens[i], tokens[i+1])
                third = tokens[i+2] if tokens[i+2] in self.finalTokenCount else "<UNK>"
                # for the probability of the token immediately following <START> in the trigram
                # model, you use the bigram probability. So in this example, you use the bigram p(HDTV| <START>) in the
                # trigram model for the probability of HDTV.
                if(first == "<START>" and second == "<START>"):
                    if self.finalBigramCount[second][third] != 0:
                        prob += math.log(self.finalBigramCount[second][third] / self.finalTokenCount["<STOP>"])
                else:
                    second = tokens[i+1] if tokens[i+1] in self.finalTokenCount else "<UNK>"
                    if first == "<START>":
                        if self.finalTrigramCount[first][second][third] != 0 and self.finalBigramCount[first][second] != 0:
                            prob += math.log(self.finalTrigramCount[first][second][third] / self.finalBigramCount[first][second])
                    else:
                        first = tokens[i] if tokens[i] in self.finalTokenCount else "<UNK>"
                        if self.finalTrigramCount[first][second][third] != 0 and self.finalBigramCount[first][second] != 0:
                            prob += math.log(self.finalTrigramCount[first][second][third] / self.finalBigramCount[first][second])
        print(math.exp(-prob/totaltokens))