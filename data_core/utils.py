import unicodedata
import re
from data_core.data_processor import LangProcessor
from constants import(
    MAX_LENGTH,
    ENG_PREFIXES,
    EOS_TOKEN,
    DEVICE
)
import torch


class WordHelper:
    
    @staticmethod
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
        
    @staticmethod
    def normalizeString(s):
        s = WordHelper.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
        return s.strip()

    @staticmethod
    def readLangs(lang1, lang2, reverse=False):
        # TODO : rewrite logging info
        print("Reading lines...")

        # Read the file and split into lines
        lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
            read().strip().split('\n')

        # Split every line into pairs and normalize
        pairs = [[WordHelper.normalizeString(s) for s in l.split('\t')] for l in lines]

        # Reverse pairs, make Lang instances
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = LangProcessor(lang2)
            output_lang = LangProcessor(lang1)
        else:
            input_lang = LangProcessor(lang1)
            output_lang = LangProcessor(lang2)

        return input_lang, output_lang, pairs

    @staticmethod
    def filterPair(p):
        return len(p[0].split(' ')) < MAX_LENGTH and \
            len(p[1].split(' ')) < MAX_LENGTH and \
            p[1].startswith(ENG_PREFIXES)
    @staticmethod
    def filterPairs(pairs):
        return [pair for pair in pairs if WordHelper.filterPair(pair)]

    @staticmethod
    def indexesFromSentence(lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]

    @staticmethod
    def tensorFromSentence(lang, sentence):
        indexes = WordHelper.indexesFromSentence(lang, sentence)
        indexes.append(EOS_TOKEN)
        return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(1, -1)
    
    @staticmethod
    def tensorsFromPair(input_lang, output_lang, pair):
        input_tensor = WordHelper.tensorFromSentence(input_lang, pair[0])
        target_tensor = WordHelper.tensorFromSentence(output_lang, pair[1])
        return (input_tensor, target_tensor)
