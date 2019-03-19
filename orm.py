from bs4 import BeautifulSoup
import time

def get_page(url, session,  retry=0):

    try:
        page = session.get(url)
        time.sleep(1)
    except:
        if retry <= 4:
            time.sleep(2)
            page = get_page(url, session, retry +1)
        else:
            raise ConnectionError
    return page

key =  None
[-1, -1, -1, -1,]



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

    def load_advanced_stats(self, sess):
        page = self.url
        #print('https://kenpom.com/'+ self.url)
        page = get_page( 'https://kenpom.com/'+ self.url, sess)
        print('https://kenpom.com/'+ self.url, sess)
        page_s =BeautifulSoup(page.text, 'lxml')
        #print(url)
        report = page_s.find_all('div', id='report')[0]

        table = report.find_all('table', id='report-table')[0].find_all('tbody')[0]
        team_stats = table.find_all('tr')  # [0]
        #import ipdb; ipdb.set_trace()
        misc = []
        for stat in team_stats:
            if hasattr(stat, 'text'):
                try:
                    s = stat.text.split('\n')
                    print(s)
                    if len(s) == 3:
                        s = s[-2]
                    else:
                        s = float(s[-1])
                    #print(s, stat.text.split('\n'))
                    misc.append(s)
                    # print(s)
                except:
                    st = stat.find_all('td')[-1].text.replace('%', '').replace('"', "")
                    try:
                        s = float(st)
                        misc.append(s)
                    except Exception as e:
                        # print(e)
                        pass

        self.misc = '^'.join(map(str, misc))
        #print(self.misc)

    def __repr__(self):
        return f'Team({self.name})'

    def load_games(self, session):
        self.season = Season(session, self)
        #self.load_advanced_stats(session)


    def headers(self, opp=False):
        if opp:
            return ', '.join([f'opp_{key}' for key in self.keys]) +',opp_misc'
        else:
            return ','.join(self.keys) +',misc'

    def csv_row(self, opp=False):
        stats = '^'.join(self.stats)
        #print(stats)
        if opp:
            d = [f'opp_{str(getattr(self, key))}' for key in self.keys] + [stats]
        else:
            d = [str(getattr(self, key)) for key in self.keys] + [stats]
        return ', '.join(d)

class Season:


    def __init__(self, session, team, base='https://kenpom.com/'):
        self.games = []
        page = get_page(base + team.url, session)


        page_s = BeautifulSoup(page.text, 'lxml')
        self.page = page_s
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
        s = 'opponent, outcome, home/away'
        return s

    def csv_row(self):

        row = f'{self.opponent}, {self.outcome}, {self.home}'
        return row
