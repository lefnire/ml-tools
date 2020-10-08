import pytest
from ml_tools import Similars, CleanText
from ml_tools.fixtures import articles

corpus = articles()

clean = CleanText(corpus)\
        .unmark()\
        .fix_punct()\
        .only_ascii()\
        .only_english()\
        .strip_html()\
        .remove_apos()\
        .multiple_whitespace()\
        .keywords()

print(clean.value()[0])

def test_cleantext():
    # Try all the functions (update this from time to time)
    assert len(clean.data.lemmas) > 0
    print(clean.data.lemmas[:10])