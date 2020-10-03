"""
Downloads some Wikipedia articles for use in tests, also commited to git & exported in case other devs want to
play around with these fixtures in their own tests / project-bootstrapping
"""
import random, re, requests, os, pdb
from pprint import pprint
from box import Box
from pathlib import Path
from typing import List

# Pick some that are pretty different. Or... consider some degree of overlap (to see how well clustering works)?
# I just grab-bagged two topics I like
pages = [Box(x) for x in [
    {"url": "https://en.wikipedia.org/wiki/Cognitive_behavioral_therapy", "k": "cbt"},
    {"url": "https://en.wikipedia.org/wiki/Virtual_reality", "k": "vr"}
]]
root_ = Path(__file__).parent


def download():
    from bs4 import BeautifulSoup
    for page in pages:
        content = requests.get(page.url).content
        soup = BeautifulSoup(content, 'html5lib')
        content = soup.find("div", id="mw-content-text")

        ps = content.find_all("p")
        txt = "\n\n".join([p.text for p in ps])
        with open(root_ / f"{page.k}.txt", "w") as f:
            f.write(txt)


def articles(group_by:str = None):
    """
    Return generated wikipedia articles.
    :param group_by:
        None=a flat list of strings
        'article'={article_title: List[str]}.
        'paragraph'={article_title_0: List[str], article_title_1: List[str], ..}
    """
    res = Box() if group_by else []
    def add_to_res(k: str, i: int, paras_: List[str]):
        text = "\n\n".join(paras_)
        if group_by is None:
            res.append(text)
        elif group_by == 'article':
            if not res.get(k, None):
                res[k] = []
            res[k].append(text)
        elif group_by == 'paragraph':
            res[f"{k}_{i}"] = text

    for page in pages:
        fname = root_ / f"{page.k}.txt"
        if not os.path.exists(fname):
            download()
            return articles(group_by)  # try again
        with open(fname, "r") as f:
            content = f.read()

        ps = content.split('\n\n')

        # Export one tiny entry, and one giant entry. This to ensure downstream tests will handle cases where
        # nlp functions operate on too-small entries, or entries larger than tokenizer can handle
        small_entry = [ps[0]]  # article title, eg "Virtual Reality"
        big_entry = []  # build this up below

        i = 0
        while True:
            n_paras = random.randint(1, 7)
            paras = ps[:n_paras]
            if not paras: break  # done
            ps = ps[n_paras:]
            clean = []
            for p in paras:
                p = re.sub(r"\[[0-9]+\]", "", p)
                if not re.search("[a-zA-Z]+", p):
                    continue  # empty
                p = re.sub(r"\s+", " ", p)
                clean.append(p)
            if not clean: continue
            big_entry += clean
            add_to_res(page.k, i, clean)
            i += 1
        # 10 paras plenty for big-entry. Want to trigger out-of-bounds tests, but not bog GPU
        add_to_res(page.k, i+1, big_entry[:10])
        # add_to_res(page.k, i+2, small_entry)  # FIXME yep, blowing things up

    return res
