import string, re, os, pdb
import html as ihtml
from bs4 import BeautifulSoup
from tqdm import tqdm
from box import Box
from gensim.models.phrases import Phrases, Phraser
from gensim.parsing import preprocessing as pp
from .utils import THREADS
from markdown import Markdown
from io import StringIO

# import spacy
# spacy.prefer_gpu()

# Much slower, but maybe more robust (see t.ent_type_ below)
NER = False
# CPU actually faster than cpu! (6it/s v 30it/s). Setting to True for now since I have one,
# might as well save some CPU
GPU = True
# a1eed8c3: lemminflect
import stanza
from stanza.resources.common import DEFAULT_MODEL_DIR
from spacy_stanza import StanzaLanguage
proc = 'tokenize,pos,lemma'
if NER: proc += ',ner'
if not os.path.exists(DEFAULT_MODEL_DIR):
    stanza.download('en')
snlp = stanza.Pipeline(lang="en", processors=proc, use_gpu=GPU)
nlp = StanzaLanguage(snlp)

from textacy.preprocessing import replace as treplace
import logging
logger = logging.getLogger(__name__)


def __unmark_element(element, stream=None):
    if stream is None:
        stream = StringIO()
    if element.text:
        stream.write(element.text)
    for sub in element:
        __unmark_element(sub, stream)
    if element.tail:
        stream.write(element.tail)
    return stream.getvalue()

# patching Markdown
Markdown.output_formats["plain"] = __unmark_element
__md = Markdown(output_format="plain")
__md.stripTopLevelTags = False
unmark = lambda s: __md.convert(s)


# TODO use RE_PUNCT inserts for proper punctuation handling. See gensim.parsing.preprocessing.RE_PUNCT
# RE_PUNCT = re.compile(r'([%s])+' % re.escape(string.punctuation), re.UNICODE)
RE_PUNCT = "[.,!;?]"


def resub(pattern, replace_with, txt):
    return re.sub(pattern, replace_with, txt, flags=re.IGNORECASE)


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
        return unmark(s)

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
        # s = UnicodeDammit.detwingle(s.encode()).decode()
        s = ihtml.unescape(s)  # is this necessary?
        s = BeautifulSoup(s, "html5lib").text
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
        return pp.strip_multiple_whitespaces(s)
        # return re.sub("\s+", "", s)  # temp: gensim slow download on tether

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
        s = resub(r"\s+", " ", s)

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
        s = unmark(s)
        s = s if self.ends_w_punct(s) else s + end_with
        return True, s

    @one_or_many(batch=True)
    def markdown_split_paragraphs(self, docs):
        # Convert doc(s) into paragraphs. Do some basic cleanup
        paras = []
        def clean_append(p):
            p = unmark(p)

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

    @one_or_many(batch=True, keep='lemmas')
    def keywords(
        self,
        docs,
        postags=['NOUN', 'ADJ', 'VERB'],
        silent=False,
        bigrams=True,

        # TODO automate these somehow, based on corpus size or something? I don't understand them
        bigram_min_count=1,
        bigram_threshold=2
    ):
        """
        Extracs keywords from documents using spacy lemmatization. Currently using Spacy Stanza (Stanford NLP),
        which is much more accurate than standard Spacy in my experience, but it's also much slower.
        See f87d5b26 for a faster version using lemminflect, still pretty accurate. TODO add that back in as
        an option, like .keywords(mode=(accurate|fast))
        :param docs: documents
        :param postags: which POS_TAGS to include. Pretty important param, you might also want to add PROPN
        :param silent: whether to print progress (it can take a while, so progress bar)
        :param bigrams: whether to include bigrams

        :return:
        """
        pbar = None
        if not silent:
            pbar = tqdm(total=len(docs))
            logger.info("Lemmatizing keywords with Stanford NLP")
        clean = []

        # batch_size doesn't seem to matter; n_process doesn't work with GPU, locks in CPU. n_threads deprecated
        # See https://spacy.io/usage/examples#multi-processing
        for doc in nlp.pipe(docs):
            if pbar: pbar.update(1)
            # if not doc: continue
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
        if pbar: pbar.close()

        set_ = lambda docs_: set(t for d in docs_ for t in d)
        logger.info(f"Before bigrams {len(set_(clean))}")
        if bigrams:
            # phrases = Phrases(docs, scoring='npmi', threshold=10e-5)
            #  fiddle with min_count, threshold
            phrases = Phrases(
                clean,
                min_count=bigram_min_count,
                threshold=bigram_threshold
            )
            bigram = Phraser(phrases)
            clean = [bigram[d] for d in clean]
        tokens = list(set_(clean))
        logger.info(f"After bigrams {len(tokens)}")

        # self._save_lemmas_for_debug(tokens)
        # return [' '.join(d) for d in docs]
        return clean, tokens
