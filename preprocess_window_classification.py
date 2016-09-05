import numpy as np

class Preprocess(object):
    def __init__(self, file_dir, window_size=2, word_dim=10):
        self.file_dir = file_dir
        self.window_size = window_size
        self.word_dim = word_dim
        self.sents = self.read_file()
        self.word_idx = self.word_indexer(self.sents)
        self.look_up = self.make_lookup_table(self.word_idx)
        
    def read_file(self):
        sents = []
        with open(self.file_dir, 'r', encoding='utf-8') as f:
            sent = []
            for line in f.readlines():
                if line == '\n':
                    sents.append(sent)
                    sent = []
                else:
                    sent.append(line[:-1].split())
        new_sents = []
        word, cls = [], None
        for sent in sents:
            new_sent = [['<S>', 'O'] for i in range(self.window_size)]
            for pair in sent:
                if pair[1] in ['O']:
                    if len(word) > 0:
                        new_sent.append(word)
                        word = []
                        cls = None
                    new_sent.append(pair)
                else:
                    if cls is None:
                        word = pair.copy()
                        cls = pair[1]
                    elif cls == pair[1]:
                        word[0] += ' %s' % pair[0]
                    else:
                        new_sent.append(word)
                        word = pair.copy()
                        cls = pair[1]
            new_sent.extend([['<E>', 'O'] for i in range(self.window_size)])
            if len(new_sent) > 2*self.window_size:
                new_sents.append(new_sent)
        return new_sents
    
    @staticmethod
    def word_indexer(sents):
        if len(sents) == 0:
            raise IndexError()
        words = ['<S>', '<E>']
        for sent in sents:
            for word in sent:
                words.append(word[0])
        word_idx = {}
        for idx, word in enumerate(list(set(words))):
            word_idx[word] = idx
        return word_idx
    
    def make_lookup_table(self, word_idx):
        return np.random.randn(self.word_dim, len(word_idx))