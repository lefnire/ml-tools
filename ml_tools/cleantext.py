import string, re, os
from bs4 import BeautifulSoup
from tqdm import tqdm
from gensim.models.phrases import Phrases, Phraser
from gensim.parsing import preprocessing as pp
from multiprocessing import cpu_count
from markdown import Markdown
from io import StringIO
import spacy
#spacy.prefer_gpu()
import lemminflect
from textacy.preprocessing import replace as treplace


THREADS = cpu_count()
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


def one_or_many(batch=False):
    """
    Wraps each cleantext method so you can pass in either a single string or list
    of strings
    :param batch: True if a method expects to process all docs together (eg, lemmatization)
    """
    def decorator(fn):
        def wrapper(txt, **kwargs):
            single = type(txt) == str
            if single: txt = [txt]
            res = fn(txt, **kwargs) if batch else [fn(s, **kwargs) for s in txt]
            return res[0] if single else res
        return wrapper
    return decorator


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

# TODO use RE_PUNCT inserts for proper punctuation handling. See gensim.parsing.preprocessing.RE_PUNCT
# RE_PUNCT = re.compile(r'([%s])+' % re.escape(string.punctuation), re.UNICODE)
RE_PUNCT = "[.,!;?]"


@one_or_many()
def unmark(s):
    return __md.convert(s)


@one_or_many()
def fix_punct(s):
    return re.sub(rf"({RE_PUNCT})([a-zA-Z])", r"\1 \2", s)


@one_or_many()
def only_ascii(s):
    return re.sub(r"[^\x00-\x7F\xA9]+", "", s)


@one_or_many()
def ends_w_punct(s):
    return re.search(rf"{RE_PUNCT}$", s)


@one_or_many()
def strip_html(s):
    s = BeautifulSoup(s, "html5lib").text
    s = treplace.replace_urls(s, 'url')
    s = treplace.replace_emails(s, 'email')
    s = treplace.replace_phone_numbers(s, 'phone')
    return s


@one_or_many()
def remove_apos(s):
    # call this before removing punctuation via gensim/spacy, since they're replaced with space
    return re.sub(r"'", "", s)


@one_or_many()
def multiple_whitespace(s):
    return pp.strip_multiple_whitespaces(s)
    # return re.sub("\s+", "", s)  # temp: gensim slow download on tether


def __is_markdown_block(i, lines):
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
    s = s if ends_w_punct(s) else s + end_with
    return True, s


@one_or_many(batch=True)
def markdown_split_paragraphs(docs):
    # Convert doc(s) into paragraphs. Do some basic cleanup
    paras = []
    def clean_append(p):
        p = unmark(p)

        if len(p) < 128: return
        p = fix_punct(p)
        p = only_ascii(p)
        p = multiple_whitespace(p)
        if not ends_w_punct(p):
            p = p + "."
        paras.append(p)

    docs = "\n\n".join(docs)
    lines = re.split('\n+', docs)
    block_agg = []
    for i, line in enumerate(lines):
        # For consistent markdown blocks (title, list-header, list-items) group them all into one paragraph.
        # Once no more block detected, bust it.
        is_block, block_txt = __is_markdown_block(i, lines)
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
@one_or_many(batch=True)
def keywords(dirty, postags=['NOUN', 'ADJ', 'VERB']):
    """
    Exracts keywords from documents, using lemmatization. Currently lemminflect, soon StanfordNLP (Stanza)
    """
    pbar = tqdm(total=len(dirty))
    clean = []

    for doc in nlp.pipe(dirty):
        pbar.update(1)
        if not doc: continue
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
    pbar.close()

    # Bigrams
    thresh = 2  # 1000 # higher threshold fewer phrases.
    phrases = Phrases(clean, min_count=1, threshold=thresh)
    bigram = Phraser(phrases)
    clean = [bigram[doc] for doc in clean]
    return clean


def multiple(s, methods):
    for m in methods: s = m(s)
    return s
