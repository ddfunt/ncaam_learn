from bs4 import BeautifulSoup

class Stat:
    pass
class Team:

    name = None
    _url = None
    year = None
    games = None

    @classmethod
    def from_basic_stats(cls, data):
        _data = []
        for i, d in enumerate(data):
            df = d.find_all('span')
            if len(df) == 0 or i==1:
                _data.append(d)

        if len(_data) == 0:
            return None
        data = _data
        keys = ['rank',
         'name',
         'conf',
         'win_loss',
         'adjEM',
         'adjO',
               # '_',
         'adjD',
                #'_',
         'adjT',
                #'_',
         'luck',
                #'_',
         'adjEM',
                #'_',
         'oppO',
                #'_',
         'oppD',
                #'_',
         'noncon_adjEM'
         ]
        #print('len', len(keys), len(data), )
        instance = cls()
        for key, d in zip(keys, data):
            instance.seed = None
            if key == 'name':
                seed = d.find_all('span')
                if len(seed) > 0:
                    instance.seed = seed[0].text
                d = d.find_all('a')[0]
            setattr(instance, key, d.text)
        instance.url = data[1].find_all('a')[0]['href']
        return instance

    def __repr__(self):
        return f'Team({self.name})'

    def load_games(self, session):
        self.season = Season(session, self)

class Season:

    def __init__(self, session, team, base='https://kenpom.com/'):

        page = session.get(base + team.url)
        page_s = BeautifulSoup(page.text, 'lxml')
        sched = page_s.find_all('table', id='schedule-table')[0]
        games = sched.find_all('tbody')[0]
        print(len(games))
        for row in games:
            print(row)

class Game:
    opponet = None