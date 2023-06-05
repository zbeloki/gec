from tokenizer import Tokenizer

from collections import namedtuple
import copy
import re
import pdb

CLASS_NAMES = [
    'R:ADJ', 'M:ADJ', 'U:ADJ',
    'R:ADJ:FORM',
    'R:ADV', 'M:ADV', 'U:ADV',
    'R:CONJ', 'M:CONJ', 'U:CONJ',
    'R:CONTR', 'M:CONTR', 'U:CONTR',
    'R:DET', 'M:DET', 'U:DET',
    'R:MORPH',
    'R:NOUN', 'M:NOUN', 'U:NOUN',
    'R:NOUN:INFL',
    'R:NOUN:NUM',
    'R:NOUN:POSS', 'M:NOUN:POSS', 'U:NOUN:POSS',
    'R:ORTH',
    'R:OTHER', 'M:OTHER', 'U:OTHER',
    'R:PART', 'M:PART', 'U:PART',
    'R:PREP', 'M:PREP', 'U:PREP',
    'R:PRON', 'M:PRON', 'U:PRON',
    'R:PUNCT', 'M:PUNCT', 'U:PUNCT',
    'R:SPELL',
    'R:VERB', 'M:VERB', 'U:VERB',
    'R:VERB:FORM', 'M:VERB:FORM', 'U:VERB:FORM',
    'R:VERB:INFL',
    'R:VERB:SVA',
    'R:VERB:TENSE', 'M:VERB:TENSE', 'U:VERB:TENSE',
    'R:WO',
]

def error_types(simplified=False):
    tags = CLASS_NAMES
    if simplified:
        tags = list(set([ simplify_type(tag) for tag in tags ]))
    return sorted(tags)

def simplify_type(tag):
    if tag[:2] in ['R:', 'U:', 'M:']:
        tag = tag[2:]
    return tag


class read_m2:

    def __init__(self, f):
        self.f = f

    def __iter__(self):
        return self

    def __next__(self):
        entry_lines = []
        ln = self.f.readline()
        while ln.strip() == "":
            if not ln:
                raise StopIteration
            ln = self.f.readline()
        while ln.strip() != "":
            entry_lines.append(ln)
            ln = self.f.readline()
        return self.parse_m2_entry(entry_lines)

    def parse_m2_entry(self, lines):
        lines = [ ln.strip() for ln in lines if ln.strip() != "" ]
        s_line = lines[0]
        a_lines = lines[1:]
        assert lines[0][0] == 'S'
        assert len([ _ for a in a_lines if a[0] != 'A' ]) == 0

        sentence = s_line[2:].strip()
        annotations = []
        for a_line in a_lines:
            a_elems = a_line[2:].strip().split('|||')
            assert len(a_elems) == 6
            offsets = a_elems[0].split()
            assert len(offsets) == 2
            start, end = int(offsets[0].strip()), int(offsets[1].strip())
            etype = a_elems[1].strip()
            corrections = a_elems[2].split('||')
            required = a_elems[3].strip()
            comment = a_elems[4].strip()
            aid = a_elems[5].strip()
            annotation = M2Entry.M2Annotation(start, end, etype, corrections, required, comment, aid)
            annotations.append(annotation)

        return M2Entry(sentence, annotations)


class M2Entry:

    M2Annotation = namedtuple('M2Annotation', ['start', 'end', 'error_type', 'corrections',
                                               'required', 'comment', 'annotator_id'])

    def __init__(self, sentence, annotations):
        self._sentence = sentence
        self._annotations = annotations
        self._detokenizer = Tokenizer(detokenizer_only=True)

    def invert(self, annotator=None):
        if annotator is None:
            annotator = self.annotators()[0]
        edits = [ edit._asdict() for edit in self.annotations(annotator=annotator) ]
        edits.sort(key=lambda e: e['start']*10000+e['end'])
        tokens = self._sentence.split()
        for i in range(len(edits)):
            edit = edits[i]
            orig_span_tokens = tokens[edit['start']:edit['end']]
            new_span_tokens = edit['corrections'][0].split() if len(edit['corrections']) > 0 else []
            tokens = tokens[:edit['start']] + new_span_tokens + tokens[edit['end']:]
            offset_diff = len(new_span_tokens) - (edit['end']-edit['start'])
            edit['corrections'] = [ ' '.join(orig_span_tokens) ]
            edit['end'] += offset_diff
            for other_edit in edits[i+1:]:
                other_edit['start'] += offset_diff
                other_edit['end'] += offset_diff
        new_edits = [ self.M2Annotation(**e) for e in edits ]
        return M2Entry(' '.join(tokens), new_edits)

    def original(self, detokenize=True):
        sent = self._sentence
        if detokenize:
            sent = self._detokenizer.detokenize(sent)
        return sent

    def corrected(self, detokenize=True, annotator=None):
        """Returns a list of corrected sentences considering all annotators by default"""
        annotators = self.annotators()
        if annotator is not None:
            if annotator not in annotators:
                return []
            annotators = [ annotator ]
        corrected = { aid: self._sentence.split() for aid in annotators }

        def apply_annotation(sent_tokens, annotation):
            if annotation.start < 0 and annotation.end < 0:
                return
            del sent_tokens[annotation.start:annotation.end]
            for tok in reversed(annotation.corrections[0].split()):
                sent_tokens.insert(annotation.start, tok)
            
        annotations = [ a for a in self._annotations if a.annotator_id in annotators ]
        for annotation in reversed(annotations):
            aid = annotation.annotator_id
            apply_annotation(corrected[aid], annotation)

        result = []
        for aid in annotators:
            sent = ' '.join(corrected[aid])
            if detokenize:
                sent = self._detokenizer.detokenize(sent)
            result.append(sent)
            
        if len(annotations) == 0:
            result.append(self.original())

        return result

    def is_correct(self):
        edits = self.annotations(ignore_unk=False, ignore_empty=True)
        # ignore empties (may not be necessary)
        edits = [ e for e in edits if e.start >= 0 ]
        return len(edits) == 0

    def annotators(self):
        annotators = []
        for annotation in self._annotations:
            if annotation.annotator_id not in annotators:
                annotators.append(annotation.annotator_id)
        return annotators

    def annotations(self, annotator=None, ignore_unk=False, ignore_empty=True):
        annotators = self.annotators()
        if annotator is not None:
            if annotator not in annotators:
                return []
            annotators = [ annotator ]
        result = [ a for a in self._annotations if a.annotator_id in annotators ]
        if ignore_unk:
            result = [ a for a in result if a.error_type != 'UNK' ]
        if ignore_empty:
            result = [ a for a in result if a.start >= 0 ]
        return result

    def apply_annotations(self, annotations):
        """Only applies the first alternative correction of each edit"""
        tokens = self._sentence.split()
        annotations_sorted = sorted(annotations, key=lambda ann: ann.start)
        for annotation in reversed(annotations_sorted):
            # apply annotation            
            if annotation.start < 0 and annotation.end < 0:
                # noop (-1, -1) edits
                return
            del tokens[annotation.start:annotation.end]
            for tok in reversed(annotation.corrections[0].split()):
                tokens.insert(annotation.start, tok)
        return self._detokenizer.detokenize(' '.join(tokens))

    def to_char_span(self, annotation, detokenized=True):
        tokens = self._sentence.split()
        ref_sent = self.original(detokenize=detokenized)
        i_tok = 0
        i_char = -1
        start_tok = annotation.start
        end_tok = annotation.end - 1 if annotation.end > start_tok else start_tok
        if start_tok >= len(tokens):
            # many UNK annotations point to the end of the sent to indicate that something is missing  
            return len(ref_sent), len(ref_sent)
        start_char, end_char = -1, -1
        while end_char < 0 and i_tok < len(tokens):
            i_char = ref_sent.find(tokens[i_tok], i_char+1)
            if i_char < 0:
                break
            if i_tok == start_tok:
                start_char = i_char
            if i_tok == end_tok:
                end_char = i_char
            i_tok += 1
        # ensure both offsets were found
        assert start_char >= 0 and end_char >= 0
        # end_char should point to the end character of the given token
        if annotation.end > annotation.start:
            end_char += len(tokens[end_tok])
        return start_char, end_char


    def mask_annotation(self, annotation_to_mask, annotations_to_apply, mask="｟MASK｠"):
        """ """
        def build_mask_annotation():
            mask_annotation = copy.deepcopy(annotation_to_mask)
            mask_annotation.corrections[0] = mask
            return mask_annotation
        mask_annotation = build_mask_annotation()
        annotations = annotations_to_apply + [mask_annotation]
        masked_sent = self.apply_annotations(annotations)
        # original content of the masked span
        start, end = annotation_to_mask.start, annotation_to_mask.end
        orig_span_tokens = self._sentence.split()[start:end]
        orig_span = self._detokenizer.detokenize(' '.join(orig_span_tokens))
        return masked_sent, orig_span
