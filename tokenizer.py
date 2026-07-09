import re


class WordTokenizer:
    def __init__(self, text):
        self.stoi, self.itos = self._build_vocab(text)
        self.vocab_size = len(self.stoi)

    def _split_words(self, text):
        text = text.lower()
        return re.findall(r"[a-zA-Z0-9]+", text)

    def _build_vocab(self, text):
        words = self._split_words(text)
        unique_words = sorted(set(words))
        stoi = {word: i for i, word in enumerate(unique_words)}
        itos = {i: word for i, word in enumerate(unique_words)}
        return stoi, itos

    def encode(self, text):
        words = self._split_words(text)
        return [self.stoi[word] for word in words]

    def decode(self, token_ids):
        words = [self.itos[i] for i in token_ids]
        return " ".join(words)

    @classmethod
    def from_vocab(cls, stoi, itos):
        """Rebuild a tokenizer from an already-built vocabulary (e.g. loaded
        from a saved checkpoint), instead of building a new one from text."""
        tokenizer = cls.__new__(cls)   # create an instance WITHOUT calling __init__
        tokenizer.stoi = stoi
        tokenizer.itos = itos
        tokenizer.vocab_size = len(stoi)
        return tokenizer