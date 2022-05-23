import requests

url = 'http://127.0.0.1:8000/auth/login/'

r = requests.post(url, json={"passowrd":"password","username":"username"})

print(r)

# pip freeze  --> to show all packages that used in your project
# pip freeze >> requirements.txt  --> to store requiremtns in specific file
# pip install -r requirments.txt  --> to install all required packages for you project

