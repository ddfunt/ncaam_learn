import requests
from bs4 import BeautifulSoup
from orm import Team
import pandas as pd

from collections import OrderedDict
import time

base = 'https://kenpom.com/'
login_suffix = 'handlers/login_handler.php'
user = 'diracdeltafunct@gmail.com'
password = 'uvawahoowa'


stats_urls = ['https://kenpom.com/summary.php?s=RankAdjOE',
              'https://kenpom.com/stats.php?s=RankeFG_Pct',
              'https://kenpom.com/teamstats.php?s=RankFG3Pct',
              'https://kenpom.com/teamstats.php?s=RankF3GRate',
              'https://kenpom.com/pointdist.php?s=RankOff_3',
              'https://kenpom.com/index.php?s=RankSOSO',
              'https://kenpom.com/height.php?s=BenchRank'
       ]
def advanced_stats(session, url, year):
    if int(year) == 2019:

        page = get_page(url, session)
    else:
        #& y = 2017
        page = get_page(url +'&y=2017', session)
    table = page.find_all('table', id='ratings-table')[0]
    body = table.find_all('tbody')[0]
    rows = body.find_all('tr')
    result = []
    for row in rows:
        dl = []
        data = row.find_all('td')
        for r in data:
            # print(r)
            try:
                if 'text-align:' in r['style']:
                    l = r.find_all('a')
                    dl.append(l[0].text)
            except:
                pass
            try:
                if 'td-left' in r['class']:
                    dl.append(r.text)
            except:
                pass
        result.append(dl)
    return result

payload = {'email': user, 'password': password}

def open_session():
    sess = requests.Session()

    sess.post(base+login_suffix, payload)
    return sess

def get_page(url, session):
    resp = session.get(url).text

    return BeautifulSoup(resp, 'lxml')

def to_csv():
    pass

def get_all_teams(page, sess=None, stats=True, year=None):
    data_table = page.find_all('table', id='ratings-table')[0]
    body = data_table.find_all('tbody')[0]
    rows = body.find_all('tr')
    teams = OrderedDict()
    stat_list = []
    for stat in stats_urls:

        stat_list.extend(advanced_stats(sess, stat, year))

    for row in rows:
        d = row.find_all('td')

        t = Team.from_basic_stats(d)


        if t is not None:
            print('loading', t.name)
            #if stats:
             #   t.load_advanced_stats(sess)
             #   #time.sleep(1)
            teams[t.name] = t
            teams[t.name].stats = []

    for stat in stat_list:
        if len(stat) > 0:
            try:
                teams[stat[0]].stats.extend(stat[1:])
            except:
                pass

    print('DONE LOADING')
    return teams


def parse_current():
    sess = open_session()

    page = get_page(base, sess)

    teams = get_all_teams(page, sess=sess, stats=True)
    with open('team_data.csv', 'w') as f:
        f.write(list(teams.values())[0].headers(opp=False) +'\n')
        for _,t in teams.items():
            #print(t)
            f.write(t.csv_row(opp=False) + '\n')



def parse_all():
    sess = open_session()

    home = get_page(base, sess)
    years_search = home.find_all('div', id='years-container')
    years_raw = years_search[0].find_all('a')
    years_links = {year.text: year['href'] for year in years_raw}

    #for year, link in years_links.items():#years_links.items():
    for year in [ '2015', '2016', '2017', '2018']:
   #for year in ['2018']:

        if year == '2019':
            link = ''
        else:
            link = years_links[year]

        page = get_page(base + link, sess)
        teams = get_all_teams(page, stats=True, sess=sess, year=year)

        t = teams[list(teams.keys())[0]]

        full_table = []
        for team_name, team in list(teams.items()):
            print('LOADING', team_name, year)

            team.load_games(sess)
            team_csv = team.csv_row()
            for game in team.season.games:
                opp = game.opponent

                try:
                    opp_csv = teams[opp].csv_row()
                    game_csv = game.csv_row()
                    full_table.append(','.join([game_csv, team_name]))
                    header = ', '.join([game.headers(), 'team1'])

                    if header is not None:
                        _header = header
                except Exception as e:
                    pass #print(e)


            #time.sleep(1)
        #print(full_table)

        table = []
        for _, team in teams.items():
            #headers = team.headers()
            d_row = team.csv_row()
            table.append(d_row)
        _table = '\n'.join(table)

        with open(f'team_data_{year}.csv', 'w') as f:
            f.write(team.headers() + ',\n')
            f.write(_table)




        write_data = '\n'.join(full_table)
        with open(f'game_data_{year}.csv', 'w') as f:
            f.write(_header + '\n')
            f.write(write_data)




    #print(t)


if __name__ == '__main__':
    from orm import Team
    #parse_current()
    parse_all()