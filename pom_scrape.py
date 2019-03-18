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

def get_all_teams(page, sess=None, stats=True):
    data_table = page.find_all('table', id='ratings-table')[0]
    body = data_table.find_all('tbody')[0]
    rows = body.find_all('tr')
    teams = OrderedDict()
    for row in rows:
        d = row.find_all('td')

        t = Team.from_basic_stats(d)


        if t is not None:
            print('loading', t.name)
            if stats:
                t.load_advanced_stats(sess)
                #time.sleep(1)
            teams[t.name] = t
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
    for year in [ '2019']:
   #for year in ['2018']:

        if year == '2019':
            link = ''
        else:
            link = years_links[year]

        page = get_page(base + link, sess)
        teams = get_all_teams(page, stats=True, sess=sess)

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