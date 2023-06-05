
class Tokenizer:

    def __init__(self, detokenizer_only=False):
        if not detokenizer_only:
            import spacy
            self.nlp = spacy.load('en', exclude=['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])
        
    def tokenize(self, text):
        
        doc = self.nlp(text)
        tokens = [ tok.text for tok in doc ]
        
        return tokens

    def detokenize(self, text):

        result_tokens = []
        tokens = text.split()

        QUOTES = ['"', "'"]
        counts = { char: 0 for char in QUOTES }
        total_squote_n = sum([ 1 for char in tokens if char == "'" ])

        for i, token in enumerate(tokens):
            prev_tok = result_tokens[-1] if i > 0 else None

            if prev_tok is not None and \
               ((prev_tok[-1] in "([{¿¡€$£#‘“") or \
                (token in ')]}.,:;?!’”-/%—') or \
                (token in ["n't"]) or \
                (prev_tok[-1] in ['-', '—', '/'] and i > 1) or \
                (token in ["'s", "’s", "'ve", "’ve", "'nt", "’nt", "'re", "’re", "'m", "’m", "'d", "’d", "'ll", "’ll"]) or \
                (token in "'’" and prev_tok[-1] == 's' and total_squote_n % 2 == 1) or \
                (token in QUOTES and counts[token] % 2 == 1) or \
                (prev_tok[-1] in QUOTES and counts[prev_tok[-1]] % 2 == 1 and len(prev_tok) <= 2) or \
                (prev_tok == 'can' and token == 'not') or \
                (prev_tok[-1] == 'º' and token == 'C')):
                result_tokens[-1] += token
            else:
                result_tokens.append(token)

            if token in QUOTES:
                counts[token] += 1

        text = ' '.join(result_tokens)
        return text

