import requests

base = 'https://kenpom.com/'
login_suffix = 'handlers/login_handler.php'
user = 'diracdeltafunct@gmail.com'
password = 'uvawahoowa'



payload = {'email': user, 'password': password}
sess = requests.Session()

sess.post(base+login_suffix, payload)

d = sess.get('https://kenpom.com/team.php?team=Gonzaga')
print(d.text)
