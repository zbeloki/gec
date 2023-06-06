import m2

from datasets import Dataset
import numpy as np

from collections import Counter
import pdb

MAX_SEQ_LEN = 256
SEP_TOKEN = "｟SEP｠"
MASK_TOKEN = "｟MASK｠"

def m2_to_dataset(m2_fpath, use_simple_types=False, invert=False):
    
    sentences = []
    types = []
    spans = []
    target_spans = []
    
    with open(m2_fpath, 'r') as f:
        for entry in m2.read_m2(f):
            if invert:
                entry = entry.invert()
            for edit in entry.annotations(ignore_unk=True, ignore_empty=True):
                char_span = entry.to_char_span(edit, detokenized=True)
                error_type = edit.error_type
                if use_simple_types:
                    error_type = m2.simplify_type(error_type)
                sentences.append(entry.original())
                types.append(error_type)
                spans.append(char_span)
                target_spans.append(edit.corrections[0])

    return Dataset.from_dict({
        'sentence': sentences,
        'error_type': types,
        'span': spans,
        'target_span': target_spans,
    })


def infer_token_span(start_logits, end_logits, word_ids):

    N_BEST = 4
    MAX_ANSWER_LEN = 3
    
    pred_start_toks = np.argsort(-start_logits)[:N_BEST]
    pred_end_toks = np.argsort(-end_logits)[:N_BEST]

    spans = []
    for start_tok in pred_start_toks:
        for end_tok in pred_end_toks:
            if not (0 <= end_tok - start_tok <= MAX_ANSWER_LEN):
                continue
            spans.append( (start_tok, end_tok, start_logits[start_tok]+end_logits[end_tok]) )
    if len(spans) == 0:
        spans.append( (pred_start_toks[0], pred_start_toks[0]+1, 0.0) )
        
    spans.sort(key=lambda span: -span[2])
    from_tok = spans[0][0]
    to_tok = spans[0][1]
    score = spans[0][2]

    # avoid selecting only subwords, it must be a full word
    while to_tok > 0 and word_ids[to_tok] == word_ids[to_tok-1]:
        to_tok += 1

    return from_tok, to_tok, score


def infer_char_span(start_logits, end_logits, offset_mapping, word_ids):

    from_tok, to_tok, score = infer_token_span(start_logits, end_logits, word_ids)

    from_char = offset_mapping[from_tok][0]
    to_char = offset_mapping[to_tok][0]
    
    return from_char, to_char, score


def to_esg_input_format(sentence, error_type, char_span=None):
    if char_span is None:
        esg_input = error_type + SEP_TOKEN + sentence
    else:
        from_ch, to_ch = char_span
        orig_span_text = sentence[from_ch:to_ch].strip()
        masked_context = sentence[:from_ch] + MASK_TOKEN + sentence[to_ch:]
        esg_input = error_type + SEP_TOKEN + orig_span_text + SEP_TOKEN + masked_context
    return esg_input

def load_type_rates(m2_fpath, use_simple_types):
    types = []
    with open(m2_fpath, 'r') as f:
        for entry in m2.read_m2(f):
            edits = entry.annotations(ignore_unk=True, ignore_empty=True)
            for edit in edits:
                error_type = edit.error_type
                if use_simple_types:
                    error_type = m2.simplify_type(error_type)
                types.append(error_type)
    counts = Counter(types)
    rates = { t: counts[t]/len(types) for t in counts.keys() }
    return rates

