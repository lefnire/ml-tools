from ml_tools.fixtures import articles


def test_group_none():
    res = articles()
    assert len(res) > 10
    assert type(res[0]) == str
    print(res[0])


def test_group_article():
    res = articles(group_by='article')
    assert len(res.vr) > 10
    assert type(res.vr[0]) == str


def test_group_paragraph():
    res = articles(group_by='paragraph')
    assert len(res.keys()) > 10
    assert type(res.vr_0) == str
