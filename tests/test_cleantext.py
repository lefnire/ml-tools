import pytest, pdb
from ml_tools import Similars, CleanText
from ml_tools.fixtures import articles
import ml_tools.cleantext as ct

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


@pytest.mark.parametrize("content", [
    (4, """
# Day 1
Lorem Ipsum is simply dummy text of the printing and typesetting industry. 
Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. 
* LI One
  * LI One Sub
* LI Two
* LI Three

It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.

# Day 2
Lorem Ipsum is simply dummy text of the printing and typesetting industry.

  1. OL One
  1. OL Two
  1. OL Three

Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book.

"""),

    (1, "hello"),

    (1, "*hello*"),

    (1, "_hello_"),

    (1, "# test"),

    (3, ["sentence one", "sentence two\n\nmultiple lines"])
])
def test_md_split_specific(content):
    ct, md = content
    res = CleanText(md).markdown_split_paragraphs().value()
    print(res)
    assert len(res) == ct


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

@pytest.mark.parametrize("content", [
"hello",

"*hello*",

"_hello_",

"# test",

"""# Markdown Title
Here is a list of items
* list item 1
* list item 2

## Next section
This is a paragraph. Blah bla blah.
""",
])
def test_unmark(content):
    res = CleanText(content).unmark().value()
    print(res)
    assert type(res) == str
    assert "#" not in res
    assert "*" not in res


@pytest.mark.parametrize("content", [
"hello",

"<span>test</span>",

"""<html>
<body><p>Test string</p></body>
</html>"""
])
def test_strip_html(content):
    res = CleanText([content]).strip_html().value()
    print(res)
    assert type(res) == str
    assert "<" not in res
