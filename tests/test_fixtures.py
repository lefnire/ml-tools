from lefnire_ml_utils.fixtures import articles


def test_group_none():
    res = articles()
    assert type(res) == list
    assert type(res[0]) == str
    print(res[0])


def test_group_article():
    res = articles(group_by='article')
    assert type(res.vr) == list
    assert type(res.vr[0]) == str


def test_group_article():
    res = articles(group_by='paragraph')
    assert type(res.vr_0) == str
