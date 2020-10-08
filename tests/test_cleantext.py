import pytest, pdb
from ml_tools import Similars, CleanText
from ml_tools.fixtures import articles

def test_md_split_1():
    doc = articles()[0]
    paras = CleanText(doc) \
        .markdown_split_paragraphs() \
        .value()
    assert len(paras) > 1
    print(paras)

def test_md_split_all():
    docs = articles()
    paras = CleanText(docs)\
        .markdown_split_paragraphs()\
        .value()
    assert len(paras) > 0
    assert len(docs) < len(paras)
    print(paras)

# @pytest.mark.parametrize("group_by", [None, "article", "paragraph"])
@pytest.mark.parametrize("fmt", ["md", "txt"])
@pytest.mark.parametrize("coverage", ["basic", "full"])
@pytest.mark.parametrize("mode", ["fast", "accurate"])
def test_normalize(fmt, coverage, mode):
    chain = CleanText(articles(fmt=fmt))
    if coverage == "basic":
        chain = chain.keywords(mode=mode)
    else:
        # Revisit this list as cleantext.py grows
        chain = chain\
            .unmark()\
            .strip_html()\
            .normalize_numbers()\
            .fix_punct()\
            .only_english()\
            .only_ascii()\
            .remove_apos()\
            .multiple_whitespace()\
            .keywords(mode=mode)
    clean = chain.join().value()
    assert len(chain.data.lemmas) > 10
    print(chain.data.lemmas[:5])
    assert len(clean) > 10
    print(clean[0])
