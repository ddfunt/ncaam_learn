import requests
from bs4 import BeautifulSoup
import pandas as pd


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

def get_all_teams(page):
    data_table = page.find_all('table', id='ratings-table')[0]
    body = data_table.find_all('tbody')[0]
    rows = body.find_all('tr')
    teams = []
    for row in rows:
        d = row.find_all('td')

        t = Team.from_basic_stats(d)
        if t is not None:
            teams.append(t)
    return teams

if __name__ == '__main__':
    from orm import Team

    sess = open_session()

    home = get_page(base, sess)
    years_search = home.find_all('div', id='years-container')
    years_raw = years_search[0].find_all('a')
    years_links = {year.text: year['href'] for year in years_raw}

    page = get_page(base + years_links['2018'], sess)
    teams = get_all_teams(page)

    t = teams[0]
    t.load_games(sess)

    print(t)

