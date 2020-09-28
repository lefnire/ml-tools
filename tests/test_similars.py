import pytest
from lefnire_ml_utils import Similars

corpus = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'A man is eating pasta.',
          'The girl is carrying a baby.',
          'The baby is carried by the woman',
          'A man is riding a horse.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'Someone in a gorilla costume is playing a set of drums.',
          'A cheetah is running behind its prey.',
          'A cheetah chases prey on across a field.'
          ] * 5

def test_x():
    s = Similars()
    res = s.start(corpus).embed().normalize().agglomorative()
    print(res.value())
    res = s.start(corpus).embed().normalize().kmeans()
    print(res.value())

def test_x_y_agg():
    s = Similars()
    x, y = corpus[:6], corpus[6:]
    res = s.start(x, y).embed().normalize().agglomorative()
    print(res.value())
    res = s.start(x, y).embed().normalize().kmeans()
    print(res.value())
