import string, re, os, pdb, os
import html as ihtml
from bs4 import BeautifulSoup
from tqdm import tqdm
from box import Box
from gensim.models.phrases import Phrases, Phraser
from gensim.parsing import preprocessing as pp
from .utils import THREADS
from textacy.preprocessing import replace as treplace
from urllib.parse import urlparse
from typing import List
import numpy as np
import logging

logger = logging.getLogger(__name__)

from stanza.resources.common import DEFAULT_MODEL_DIR
SPACY_GPU = True
import spacy
# Putting spacy files in same folder as stanza files
# spacy.util.set_data_path(DEFAULT_MODEL_DIR)
if SPACY_GPU:
    spacy.prefer_gpu()

import lemminflect  # just import. Ties itself into spacy internals
# nlp_fast = spacy.load('en', disable=['parser', 'ner'])
nlp_fast = spacy.load('en')

nlp_accurate = None
def init_nlp_accurate():
    global nlp_accurate
    # CPU actually faster than cpu! (6it/s v 30it/s). Setting to True for now since I have one,
    # might as well save some CPU
    # a1eed8c3: lemminflect
    import stanza
    from spacy_stanza import StanzaLanguage
    proc = 'tokenize,pos,lemma,ner'
    if not os.path.exists(DEFAULT_MODEL_DIR):
        stanza.download('en')
    snlp = stanza.Pipeline(lang="en", processors=proc, use_gpu=SPACY_GPU)
    nlp_accurate = StanzaLanguage(snlp)


# See gensim.parsing.preprocessing.RE_PUNCT
# RE_PUNCT = re.compile(r'([%s])+' % re.escape(string.punctuation), re.UNICODE)
RE_PUNCT = r'[%s]' % re.escape(string.punctuation)


def mult_whitespace(s):
    return pp.strip_multiple_whitespaces(s)


def resub(pattern, replace_with, txt):
    return re.sub(pattern, replace_with, txt, flags=re.IGNORECASE)


from markdown2 import Markdown
md = Markdown(extras=["cuddled-lists"])
def md2txt(s):
    return html2txt(md.convert(s))

from markdownify import markdownify
def html2md(s):
    return markdownify(s)


def html2txt(s):
    # s = UnicodeDammit.detwingle(s.encode()).decode()
    s = ihtml.unescape(s)  # is this necessary?
    return BeautifulSoup(s, "html5lib").text


def one_or_many(batch=False, keep=None):
    """
    Wraps each cleantext method so you can pass in either a single string or list.
    :param batch: True if a method expects to process all docs together (eg, lemmatization)
    :param keep: A string key for saving away intermediate values for access later on `.data`. If used,
        method should return a tuple (result, thing_to_keep). Eg, for lemmatization, it returns the lemmatized text
        but also keeps the unique lemmas themselves, so you can debug later via cleantxt.data.lemmas
    """
    def decorator(fn):
        def wrapper(self, *args, **kwargs):
            txt = self.result
            single = type(txt) == str
            if single: txt = [txt]
            # most functions can't handle empty strings
            txt = [t or "empty document" for t in txt]
            txt = fn(self, txt, *args, **kwargs) if batch\
                else  [fn(self, s, *args, **kwargs) for s in txt]

            data = self.data
            if keep:
                data[keep] = txt[-1]
                txt = txt[0]
            return self.__class__(txt, fn.__name__, data)
        return wrapper
    return decorator


class CleanText:
    def __init__(self, txt, last_fn=None, data=Box()):
        self.result = txt
        self.last_fn = last_fn
        self.data = data

    def value(self):
        txt = self.result
        if self.last_fn == 'markdown_split_paragraphs':
            # ensure it stays wrapped, even if just one paragraph
            return txt
        return txt[0] if len(txt) == 1 else txt

    @one_or_many()
    def unmark(self, s):
        return md2txt(s)

    @one_or_many()
    def fix_punct(self, s):
        return re.sub(rf"({RE_PUNCT})([a-zA-Z])", r"\1 \2", s)

    @one_or_many()
    def only_ascii(self, s):
        return re.sub(r"[^\x00-\x7F\xA9]+", "", s)

    @one_or_many()
    def only_english(self, s):
        s = re.sub("[\uac00-\ud7a3]+", 'korean', s)
        s = re.sub("[\u3040-\u30ff]+", 'japanese', s)
        s = re.sub("[\u4e00-\u9FFF]+", 'chinese', s)
        return s

    @staticmethod
    def ensure_punct(s):
        s = s.strip()
        if not re.search(rf"{RE_PUNCT}$", s):
            s += "."
        return s

    @one_or_many()
    def strip_html(self, s):
        s = html2txt(s)
        s = treplace.replace_urls(s, 'url')
        s = treplace.replace_emails(s, 'email')
        s = treplace.replace_phone_numbers(s, 'phone')
        return s

    @one_or_many()
    def remove_apos(self, s):
        # call this before removing punctuation via gensim/spacy, since they're replaced with space
        return re.sub(r"'", "", s)

    @one_or_many()
    def multiple_whitespace(self, s):
        return mult_whitespace(s)

    @one_or_many()
    def normalize_numbers(self, s):
        """
        Goal is to cleanup numbers, not squash them. Spacy will catch SYM-NUM+ well enough that we can leave actual
        numbers (with decimals), amounts, sizes, etc. BERT will want to use such things. This method will be a
        constant work in progress, assess from time to time
        """

        # s = treplace.replace_currency_symbols(s)
        # s = re.sub(r"(\$|€|£|¥|usd|cny|eur|gbp|dollars)", "price ", s, flags=re.IGNORECASE)

        # misc
        # ordinals? shouldn't these be handled by spacy?
        # s = resub(r"\b([0-9]+)(st|rd|th|nd)\b", r"\1", s)  # handled in keywords()
        s = resub(r"\b([0-9]{4})[\-/]([0-9]{4})\b", r"\1, \2", s)  # break up dates
        s = s.replace("~", "")  # re.sub(r"\b\~$", "$", s, flags=re.IGNORECASE)

        # Sizes
        s = resub(r"(kilo|mega|giga)?(bytes?|hertz|hz)", " bytes ", s)
        s = resub(r"(\d|\b)(kb|mb|gb|kib|mib|gig|ghz|mhz)s?\b", r"\1 bytes ", s)

        # handled in keywords()
        # s = resub(r"(trillion|billion|million|thousand|hundred)s?", " amount ", s)
        # s = resub(r"(\d|\b)(k|m|b)\b", r"\1 amount ", s)

        # Remove over-added spaces during ^
        s = mult_whitespace(s)

        return s

    @one_or_many(batch=True)
    def markdown_split_paragraphs(self, docs):
        # Convert doc(s) into paragraphs.
        paras = []
        for doc in docs:
            soup = BeautifulSoup(md.convert(doc), "html5lib")\
                .find("body").findChildren(recursive=False)
            last_tag = ""
            for t in soup:
                tag, text = t.name, t.text
                if not text: continue
                text = ' '.join([
                    self.ensure_punct(line)
                    for line in text.split('\n')
                    if line
                ])
                start_new = not paras or\
                    tag.startswith('h') or\
                    (tag == 'p' and not last_tag.startswith('h'))
                if start_new:
                    paras.append(text)
                else:
                    paras[-1] += " " + text
                last_tag = tag
        return paras


    # You'll never really call this on a single doc, so one_or_many is pointless; just error-preventing
    @one_or_many(batch=True, keep='lemmas')
    def keywords(
        self,
        docs,
        postags=['NOUN', 'ADJ', 'VERB', 'PROPN'],
        mode='fast',
        silent=False,
        bigrams=True,

        # TODO automate these somehow, based on corpus size or something? I don't understand them
        bigram_min_count=5,
        bigram_threshold=10.
    ):
        """
        Extracs keywords from documents using spacy lemmatization.
        :param docs: documents
        :param postags: which POS_TAGS to include. Pretty important param, you might also want to add PROPN
        :param mode: (fast|accurate). Fast uses lemminflect, accurate uses Stanford NLP. Pretty substantial in their
            trade-off IMO. Could use some helping eyes on this code.
        :param silent: whether to print progress (it can take a while, so progress bar)
        :param bigrams: whether to include bigrams
        """
        # could ensure they call this before keywords, because spacy is brittle without it, but this is easier.
        docs = [mult_whitespace(d) for d in docs]

        if mode == 'fast':
            nlp = nlp_fast
            mode_ = "Spacy + Lemminflect"
        else:
            if nlp_accurate is None:
                init_nlp_accurate()
            nlp = nlp_accurate
            mode_ = "spacy-stanza (Stanford NLP)"

        pbar = None
        if not silent:
            pbar = tqdm(total=len(docs))
            logger.info(f"Lemmatizing keywords with {mode_}")

        clean = []
        # batch_size doesn't seem to matter; n_process doesn't work with GPU, locks in CPU. n_threads deprecated
        # Joblib approach https://spacy.io/usage/examples#multi-processing causes issues for threads|processes
        # I give up, non-parallel it is!
        for doc in nlp.pipe(docs):
            assert doc, "keywords() got empty document. This should be handled in @wrapper"
            if not silent: pbar.update(1)
            tokens = []
            for t in doc:
                # https://spacy.io/api/token
                # print(t.pos_, t.ent_type_, t.is_stop, t.is_punct)

                # ner https://spacy.io/api/annotation#named-entities
                # slow, but catches sequences of hard-to-catch numeric stuff
                if t.ent_type_ in 'DATE TIME PERCENT ORDINAL MONEY QUANTITY CARDINAL'.split():
                    # already accounted for, $5m becomes MONEY MONEY MONEY
                    if tokens and tokens[-1] == t.ent_type_:
                        continue
                    t = t.ent_type_
                elif t.is_stop or t.is_punct:
                    continue
                elif t.like_url:
                    # Might want simple urls (apple.com), sticking to url/email regex in strip_html
                    t = urlparse(t.text).netloc
                elif t.pos_ not in postags:
                    continue
                else:
                    t = t._.lemma() if mode == 'fast' else t.lemma_
                    t = t.lower()
                    if len(t) < 2: continue
                tokens.append(t)
            clean.append(tokens)
        docs = clean
        if pbar: pbar.close()

        set_ = lambda docs_: set(t for d in docs_ for t in d)
        if not silent:
            logger.info(f"Before bigrams {len(set_(docs))}")
        if bigrams:
            # phrases = Phrases(docs, scoring='npmi', threshold=10e-5)
            # fiddle with min_count, threshold
            phrases = Phrases(
                docs,
                min_count=bigram_min_count,
                threshold=bigram_threshold
            )
            bigram = Phraser(phrases)
            docs = [bigram[d] for d in docs]
        tokens = list(set_(docs))
        if not silent:
            logger.info(f"After bigrams {len(tokens)}")
        return docs, tokens

    @one_or_many(batch=True)
    def join(self, docs: List[List[str]]):
        """
        keywords() returns lists of tokens, this joins it back into strings
        """
        # meant to lists of terms, so ensure we're not getting `str` or `List[str]`
        while np.array(docs).ndim < 2: docs = [docs]
        return [' '.join(terms) for terms in docs]
