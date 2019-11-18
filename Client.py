import requests
import json
from flask import jsonify

with open('Input/example.json', 'r') as myfile:
    data=myfile.read()
res = requests.post('http://localhost:5000/', json = data)
if res.ok:
    print (res.content)