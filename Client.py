import requests
import json
from flask import jsonify
for i in range(60):
    print(i)
    with open('Input/example.json', 'r') as myfile:
        data=myfile.read()
    res = requests.post('http://34.68.37.211:8000/', json = data)
    if res.ok:
        print (res.content)