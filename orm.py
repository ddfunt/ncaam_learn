from bs4 import BeautifulSoup

class Stat:
    pass
class Team:
    keys = ['rank',
            'name',
            'conf',
            'win_loss',
            'adjEM',
            'adjO',
            'adjD',
            'adjT',
            'luck',
            'adjEM',
            'oppO',
            'oppD',
            'noncon_adjEM'
            ]

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

        #print('len', len(keys), len(data), )
        instance = cls()
        for key, d in zip(instance.keys, data):
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

    def headers(self, opp=False):
        if opp:
            return ', '.join([f'opp_{key}' for key in self.keys])
        else:
            return ','.join(self.keys)

    def csv_row(self, opp=False):
        if opp:
            d = [f'opp_{str(getattr(self, key))}' for key in self.keys]
        else:
            d = [str(getattr(self, key)) for key in self.keys]
        return ', '.join(d)

class Season:


    def __init__(self, session, team, base='https://kenpom.com/'):
        self.games = []
        page = session.get(base + team.url)
        page_s = BeautifulSoup(page.text, 'lxml')
        sched = page_s.find_all('table', id='schedule-table')[0]
        games = sched.find_all('tbody')[0]

        #print()
        tourn = 0

        for row in games:
            real = True
            #print(row)
            try:

                if 'label' in row.td['class']:
                    real = False

                    if 'NCAA' in row.td.text:
                        tourn = 2
                    else:
                        tourn = 1
                else:

                    self.games.append(Game.from_web(row, tourn))

            except Exception as e:
                #print(e)

                pass
            finally:
                if real:

                    try:
                        game = Game.from_web(row, tourn)
                        #print(game.date)
                        self.games.append(game)

                    except Exception as e:
                        pass#print(e)
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
        #print(len(rows))
        #print(rows)
        if len(rows) == 11:
            r = [0, 1, 3, 4,7]
        elif len(rows) == 10:
            r = [0, 1, 2,3, 6 ]
        #print(rows)
        instance.date = rows[r[0]].text
        instance.rank = rows[r[1]].text
        instance.opponent = rows[r[2]].text
        #print(rows[r[3]])
        instance.outcome, instance.score = rows[r[3]].text.split(',')
        if 'Home' in rows[r[4]].text:
            instance.home = 'H'
        elif 'Neutral' in rows[r[4]].text:
            instance.home = 'N'
        else:
            instance.home = 'A'
        return instance

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        self._score = value
        self.team1_score, self.team2_score = value.split('-')


    def headers(self):
        s = 'date, rank, opponent, outcome, team_score, opp_score, home/away'
        return s

    def csv_row(self):

        row = f'{self.date}, {self.rank}, {self.opponent}, {self.outcome}, {self.team1_score}, {self.team2_score}, {self.home}'
        return row
