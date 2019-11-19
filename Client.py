import requests
import json
from flask import jsonify

with open('Input/example.json', 'r') as myfile:
    data=myfile.read()
res = requests.post('http://127.0.0.1:6000/', json = data)
if res.ok:
    print (res.content)