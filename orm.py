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
        self.games = []
        page = session.get(base + team.url)
        page_s = BeautifulSoup(page.text, 'lxml')
        sched = page_s.find_all('table', id='schedule-table')[0]
        games = sched.find_all('tbody')[0]
        print(len(games))
        #print()
        tourn = 0

        for row in games:
            real = True
            #print(row)
            try:

                if 'label' in row.td['class']:
                    real = False
                    print(row.text)
                    if 'NCAA' in row.td.text:
                        tourn = 2
                    else:
                        tourn = 1
                else:
                    print('HEY IM HERE')
                    self.games.append(Game.from_web(row, tourn))

            except Exception as e:
                #print(e)

                pass
            finally:
                if real:
                    try:
                        self.games.append(Game.from_web(row, tourn))
                        print('Game Created')
                    except:
                        pass
class Game:
    opponet = None
    _score = None

    @classmethod
    def from_web(cls, data, tourn):
        #print('CREATING GAME')
        instance = cls()
        instance.tourn = tourn
        #print(data)
        rows = data.find_all('td')
        #print(rows)
        instance.date = rows[0].text
        instance.rank = rows[1].text
        instance.opponent = rows[3].text
        instance.outcome, instance.score = rows[4].text.split(',')
        if 'Home' in rows[6].text:
            instance.home = 2
        elif 'Neutral' in rows[6].text:
            instance.home = 1
        else:
            instance.home = 0
        return instance

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        self._score = value
        self.us_score, self.them_score = value.split('-')
