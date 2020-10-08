import string, re, os, pdb, os
import html as ihtml
from bs4 import BeautifulSoup
from tqdm import tqdm
from box import Box
from gensim.models.phrases import Phrases, Phraser
from gensim.parsing import preprocessing as pp
from .utils import THREADS
from textacy.preprocessing import replace as treplace
from typing import List
import logging

# Markdown helpers
from markdownify import markdownify  # html2md
import markdown  # md2txt

logger = logging.getLogger(__name__)

from stanza.resources.common import DEFAULT_MODEL_DIR
SPACY_GPU = True
import spacy
# Putting spacy files in same folder as stanza files
# spacy.util.set_data_path(DEFAULT_MODEL_DIR)
if SPACY_GPU:
    spacy.prefer_gpu()

nlp_fast = None
def init_nlp_fast():
    global nlp_fast
    import lemminflect  # just import. Ties itself into spacy internals
    m, kwargs = 'en_core_web_sm', dict(disable=['parser', 'ner'])
    try:
        # TODO put this on /storage (pretty small, but still)
        nlp_fast = spacy.load(m, **kwargs)
    except:
        spacy.cli.download(m)
        nlp_fast = spacy.load(m, **kwargs)


# Much slower, but maybe more robust (see t.ent_type_ below)
NER = False
nlp_accurate = None
def init_nlp_accurate():
    global nlp_accurate
    # CPU actually faster than cpu! (6it/s v 30it/s). Setting to True for now since I have one,
    # might as well save some CPU
    # a1eed8c3: lemminflect
    import stanza
    from spacy_stanza import StanzaLanguage
    proc = 'tokenize,pos,lemma'
    if NER: proc += ',ner'
    if not os.path.exists(DEFAULT_MODEL_DIR):
        stanza.download('en')
    snlp = stanza.Pipeline(lang="en", processors=proc, use_gpu=SPACY_GPU)
    nlp_accurate = StanzaLanguage(snlp)


# TODO use RE_PUNCT inserts for proper punctuation handling. See gensim.parsing.preprocessing.RE_PUNCT
# RE_PUNCT = re.compile(r'([%s])+' % re.escape(string.punctuation), re.UNICODE)
RE_PUNCT = "[.,!;?]"


def mult_whitespace(s):
    return pp.strip_multiple_whitespaces(s)


def resub(pattern, replace_with, txt):
    return re.sub(pattern, replace_with, txt, flags=re.IGNORECASE)


def md2txt(s):
    html_ = markdown.markdown(s)
    return html2txt(html_)


def html2md(s):
    return markdownify(s)


def html2txt(s):
    # s = UnicodeDammit.detwingle(s.encode()).decode()
    s = ihtml.unescape(s)  # is this necessary?
    return BeautifulSoup(s, "html5lib").text


def one_or_many(chain=True, batch=False, keep=None):
    """
    Wraps each cleantext method so you can pass in either a single string or list.
    :param chain: if True, hains calls, like CleanText(sentences).strip_html().keywords().value().
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
            if not chain:
                return txt

            data = self.data
            if keep:
                data[keep] = txt[-1]
                txt = txt[0]
            return CleanText(txt, data)
        return wrapper
    return decorator


class CleanText:
    def __init__(self, txt, data=Box()):
        self.result = txt
        self.data = data

    def value(self):
        txt = self.result
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
    def ends_w_punct(s):
        return re.search(rf"{RE_PUNCT}$", s)

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
        s = resub(r"\b([0-9]+)(st|rd|th|nd)\b", r"\1", s)
        s = resub(r"\b([0-9]{4})[\-/]([0-9]{4})\b", r"\1, \2", s)  # break up dates
        s = s.replace("~", "")  # re.sub(r"\b\~$", "$", s, flags=re.IGNORECASE)

        # Sizes
        s = resub(r"(kilo|mega|giga)?(bytes?|hertz|hz)", " bytes ", s)
        s = resub(r"(\d|\b)(kb|mb|gb|kib|mib|gig|ghz|mhz)s?\b", r"\1 bytes ", s)

        s = resub(r"(trillion|billion|million|thousand|hundred)s?", " amount ", s)
        s = resub(r"(\d|\b)(k|m|b)\b", r"\1 amount ", s)

        # Remove over-added spaces during ^
        s = mult_whitespace(s)

        return s

    def __is_markdown_block(self, i, lines):
        s = lines[i]
        s_next = lines[i+1] if i+1 < len(lines) else ''
        RE_LI =  r"^\s*([*\-+]|[0-9]+\.)"
        is_block = False

        # heading
        if re.search(r"^[#]+", s):
            is_block = True
            end_with = "."
        # li (come before UL for inline replacement)
        elif re.search(RE_LI, s):
            s = re.sub("^\s*", "", s)  # unmark doesn't like spaces before li's
            is_block = True
            end_with = ";"
        # ul
        elif re.search(RE_LI, s_next):
            is_block = True
            end_with = ":"

        if not is_block: return False, ""
        s = md2txt(s)
        s = s if self.ends_w_punct(s) else s + end_with
        return True, s

    @one_or_many(batch=True)
    def markdown_split_paragraphs(self, docs):
        # Convert doc(s) into paragraphs. Do some basic cleanup
        paras = []
        def clean_append(p):
            p = md2txt(p)

            if len(p) < 128: return
            p = CleanText(p).fix_punct().only_ascii().multiple_whitespace().value()
            if not self.ends_w_punct(p):
                p = p + "."
            paras.append(p)

        docs = "\n\n".join(docs)
        lines = re.split('\n+', docs)
        block_agg = []
        for i, line in enumerate(lines):
            # For consistent markdown blocks (title, list-header, list-items) group them all into one paragraph.
            # Once no more block detected, bust it.
            is_block, block_txt = self.__is_markdown_block(i, lines)
            if is_block:
                block_agg.append(block_txt)
                continue
            elif len(block_agg) > 0:
                block = " ".join(block_agg)
                block_agg.clear()
                clean_append(block)
            clean_append(line)
        return paras

    # You'll never really call this on a single doc, so one_or_many is pointless; just error-preventing
    @staticmethod
    def _keywords_fast(pbar, docs, postags):
        if nlp_fast is None: init_nlp_fast()

        clean = []
        for doc in nlp_fast.pipe(docs):
            assert doc, "keywords() got empty document. This should be handled in @wrapper"
            pbar.update(1)
            tokens = []
            for t in doc:
                if t.is_stop or t.is_punct: continue
                elif t.like_num: t = 'number'
                elif t.is_currency: t = 'currency'
                elif t.pos_ == 'SYM': t = 'symbol'
                elif t.pos_ not in postags: continue
                else:
                    t = t.lemma_.lower()
                    t = pp.strip_non_alphanum(t)
                    # token = only_ascii(t)
                    if len(t) < 2: continue
                tokens.append(t)
            clean.append(tokens)
        return clean


    @staticmethod
    def _keywords_accurate(pbar, docs, postags):
        if nlp_accurate is None: init_nlp_accurate()

        clean = []
        # batch_size doesn't seem to matter; n_process doesn't work with GPU, locks in CPU. n_threads deprecated
        # Joblib approach https://spacy.io/usage/examples#multi-processing causes issues for threads|processes
        # I give up, non-parallel it is!
        for doc in nlp_accurate.pipe(docs):
            assert doc, "keywords() got empty document. This should be handled in @wrapper"
            if pbar: pbar.update(1)
            tokens = []
            for t in doc:
                # https://spacy.io/api/token
                if t.is_stop or t.is_punct:
                    continue

                # ner https://spacy.io/api/annotation#named-entities
                # slow, but catches sequences of hard-to-catch numeric stuff. Only save
                # one in the sequence ($10m -> MONEY MONEY MONEY)
                elif NER and tokens and tokens[-1] != t.ent_type_ and \
                    t.ent_type_ in 'DATE TIME PERCENT ORDINAL MONEY QUANTITY'.split():
                    t = t.ent_type_

                # Might want simple urls (apple.com), sticking to url/email regex in strip_html
                # elif t.like_url: tokens.append('url')
                # elif t.like_email: t = 'email'

                # These ever used, after ent_type_ above?
                elif t.like_num:
                    t = 'number'
                elif t.is_currency:
                    t = 'currency'
                elif t.pos_ == 'SYM':
                    t = 'symbol'
                # save for last, since using NER/POS conditions above
                elif t.pos_ not in postags:
                    continue
                else:
                    t = t.lemma_
                    # Sometimes want certin symbols, reconsider
                    t = re.sub("[,!?\%'/]", "", t.lower())  # pp.strip_non_alphanum(t)
                    if len(t) < 2: continue
                    # just a number after removing punct
                    if treplace.replace_numbers(t) == '_NUM_': t = "number"
                tokens.append(t)
            clean.append(tokens)
        return clean

    @one_or_many(batch=True, keep='lemmas')
    def keywords(
        self,
        docs,
        postags=['NOUN', 'ADJ', 'VERB'],
        mode='fast',
        silent=False,
        bigrams=True,

        # TODO automate these somehow, based on corpus size or something? I don't understand them
        bigram_min_count=1,
        bigram_threshold=2
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
        pbar = None
        if not silent:
            pbar = tqdm(total=len(docs))
            mode_ = {"fast": "Spacy + Lemminflect", "accurate": "spacy-stanza (Stanford NLP)"}[mode]
            logger.info(f"Lemmatizing keywords with {mode_}")
        fn = self._keywords_fast if mode == 'fast' else self._keywords_accurate
        # could ensure they call this before keywords, because spacy is brittle without it, but this is easier.
        docs = [mult_whitespace(d) for d in docs]
        docs = fn(pbar, docs, postags)
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

        # self._save_lemmas_for_debug(tokens)
        # return [' '.join(d) for d in docs]
        return docs, tokens

    @one_or_many()
    def join(self, doc: List[str]):
        """
        keywords() returns lists of tokens, this joins it back into strings
        """
        return ' '.join(doc)
