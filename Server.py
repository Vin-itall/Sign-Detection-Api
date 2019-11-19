import ast

import numpy as np
import os
import pandas
import tensorflow as tf
from flask import Flask, request, jsonify, make_response
import json
import csv
import Utils
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from sklearn import preprocessing
from sklearn.externals.joblib import dump, load

app = Flask(__name__)

labels = {0: 'book', 1: 'car', 2: 'gift', 3: 'movie', 4: 'sell', 5: 'total'}
ouptut = {}


def load_dirty_json(dirty_json):
    regex_replace = [(r"([ \{,:\[])(u)?'([^']+)'", r'\1"\3"'), (r" False([, \}\]])", r' false\1'), (r" True([, \}\]])", r' true\1')]
    for r, s in regex_replace:
        dirty_json = re.sub(r, s, dirty_json)
    clean_json = json.loads(dirty_json)
    return clean_json

def convert_to_csv(d):
    columns = ['score_overall', 'nose_score', 'nose_x', 'nose_y', 'leftEye_score', 'leftEye_x', 'leftEye_y',
               'rightEye_score', 'rightEye_x', 'rightEye_y', 'leftEar_score', 'leftEar_x', 'leftEar_y',
               'rightEar_score', 'rightEar_x', 'rightEar_y', 'leftShoulder_score', 'leftShoulder_x', 'leftShoulder_y',
               'rightShoulder_score', 'rightShoulder_x', 'rightShoulder_y', 'leftElbow_score', 'leftElbow_x',
               'leftElbow_y', 'rightElbow_score', 'rightElbow_x', 'rightElbow_y', 'leftWrist_score', 'leftWrist_x',
               'leftWrist_y', 'rightWrist_score', 'rightWrist_x', 'rightWrist_y', 'leftHip_score', 'leftHip_x',
               'leftHip_y', 'rightHip_score', 'rightHip_x', 'rightHip_y', 'leftKnee_score', 'leftKnee_x', 'leftKnee_y',
               'rightKnee_score', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 'leftAnkle_x', 'leftAnkle_y',
               'rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y']
    d = str(d)
    data = ast.literal_eval(d)
    csv_data = np.zeros((len(data), len(columns)))
    x = csv_data.shape[0]
    for i in range(0,x):
        one = []
        one.append(data[i]["score"])
        for obj in data[i]["keypoints"]:
            one.append(obj["score"])
            one.append(obj["position"]["x"])
            one.append(obj["position"]["y"])
        csv_data[i] = np.array(one)
    df = pd.DataFrame(csv_data, columns=columns)
    return df

def loadModels(test_data):
    model1 = tf.keras.models.load_model('Models/CNN_1.h5')
    model2 = tf.keras.models.load_model('Models/CNN_2.h5')
    model3 = tf.keras.models.load_model('Models/CNN_3.h5')
    model4 = tf.keras.models.load_model('Models/CNN_4.h5')
    predictions=[]
    prediction1 = np.argmax(model1.predict(np.array([test_data, ])))
    prediction2 = np.argmax(model2.predict(np.array([test_data, ])))
    prediction3 = np.argmax(model3.predict(np.array([test_data, ])))
    prediction4 = np.argmax(model4.predict(np.array([test_data, ])))
    predictions.append(labels[prediction1])
    predictions.append(labels[prediction2])
    predictions.append(labels[prediction3])
    predictions.append(labels[prediction4])
    return predictions

@app.route('/', methods=['GET', 'POST'])
def add_message():
    data = request.get_json()
    print(type(data))
    df = convert_to_csv(data)
    test_data = Utils.preprocess(df)
    predictions = loadModels(test_data)
    x = { "0" : predictions[0], "1" : predictions[1], "2" :predictions[3], "3" :predictions[3]}
    return send_response(x)

def send_response(x):
    jsonRes = jsonify(x)
    return jsonRes

if __name__ == '__main__':
    app.run(host= '172.31.30.146',port=5000)

