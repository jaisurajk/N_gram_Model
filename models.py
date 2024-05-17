from collections import defaultdict
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

<<<<<<< HEAD

class TrigramModel():
    def __init__(self, alpha=0):
        return
=======
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

        for sentence in sentences:
            tokens = ["<START>", "<START>"] + sentence.split() + ["<STOP>"]
            for i in range(len(tokens) - 2):
                first = tokens[i] if tokens[i] in self.finalTokenCount else "<UNK>"
                second = tokens[i+1] if tokens[i+1] in self.finalTokenCount else "<UNK>"
                third = tokens[i+2] if tokens[i+2] in self.finalTokenCount else "<UNK>"
                self.finalBigramCount[first][second] += 1
                self.finalTrigramCount[first][second][third] += 1

    def perplexity(self, sentences):
        """
        Calculate the perplexity of the model on a list of sentences.
        
        Args:
            sentences (list of str): List of sentences to calculate perplexity on.
        
        Returns:
            float: The perplexity value.
        """
        prob = 0
        ct = 0
        numStops = 0

        for sentence in sentences:
            tokens = ["<START>", "<START>"] + sentence.split() + ["<STOP>"]
            numStops += 1
            for i in range(2, len(tokens)):
                ct += 1
                first = tokens[i-2] if tokens[i-2] in self.finalTokenCount else "<UNK>"
                second = tokens[i-1] if tokens[i-1] in self.finalTokenCount else "<UNK>"
                third = tokens[i] if tokens[i] in self.finalTokenCount else "<UNK>"
                
                if self.finalTrigramCount[first][second][third] > 0:
                    trigram_count = self.finalTrigramCount[first][second][third]
                    bigram_count = self.finalBigramCount[first][second]
                    prob += math.log(trigram_count / bigram_count)
                else:
                    prob += math.log(self.finalTrigramCount[first][second]["<UNK>"] / self.finalBigramCount[first][second])
        
        perplexity_value = math.exp(-prob / (ct + numStops))
        print(perplexity_value)
        return perplexity_value

>>>>>>> f6729c2 (Update models.py)
