import numpy as np
import pandas
import tensorflow as tf
from flask import Flask, request, jsonify
import json
import csv
import Utils
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from sklearn import preprocessing
from sklearn.externals.joblib import dump, load

app = Flask(__name__)

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
    data = json.loads(d)
    csv_data = np.zeros((len(data), len(columns)))
    for i in range(csv_data.shape[0]):
        one = []
        one.append(data[i]['score'])
        for obj in data[i]['keypoints']:
            one.append(obj['score'])
            one.append(obj['position']['x'])
            one.append(obj['position']['y'])
        csv_data[i] = np.array(one)
    pd.DataFrame(csv_data, columns=columns).to_csv('Res/input.csv', index_label='Frames#')

def loadModels(test_data):
    model = tf.keras.models.load_model('Models/CNN.h5')
    prediction = np.argmax(model.predict(np.array([test_data, ])))
    print(prediction)
    return str(prediction)



output = [{ "CNN_Prediction" : 'Sell'},{ "NB_Prediction" : 'Total'}]
output = str(output)

@app.route('/', methods=['GET', 'POST'])
def add_message():
    result=''
    try:
        data = request.json
        convert_to_csv(data)
        test_data = Utils.preprocess()
        result = loadModels(test_data)
    except TypeError:
        pass
    return send_response(result)


def send_response(x):
    jsonRes = jsonify(x)
    return jsonRes

if __name__ == '__main__':
    app.debug = True
    app.run(host= '127.0.0.1')

