import os
from flask import Flask, render_template
from flask import request
import pickle
import numpy as np
import pandas as pd
import scipy
import sklearn

import skimage
import skimage.io
import skimage.color
import skimage.transform
import skimage.feature
import matplotlib.pyplot as plt

app = Flask(__name__)
BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload/')
MODEL_PATH = os.path.join(BASE_PATH, 'static/models/')

model_sgd_path = os.path.join(MODEL_PATH, 'model_after_pipeline1.pickle')
scaler_path = os.path.join(MODEL_PATH, 'scalar.pickle')
model_sgd = pickle.load(open(model_sgd_path, 'rb'))
scalar = pickle.load(open(scaler_path, 'rb'))


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        file_name = upload_file.filename
        print(file_name)

        ext = file_name.split('.')[-1]
        print(ext)
        if ext.lower() in ['png', 'jpg', 'jpeg']:
            path_save = os.path.join(UPLOAD_PATH, file_name)
            upload_file.save(path_save)
            print("File saved successfully")

            results = pipeline_model(path_save, scalar, model_sgd)
            print(results)
            heigh = getheight(path_save)
            return render_template('upload.html', fileupload=True, data=results, image=file_name, height=heigh)

        else:
            print("Please upload only .jpg, .png, .jpeg")
            return render_template('upload.html')

    else:
        return render_template('upload.html', fileupload=False)


@app.route('/about/')
def about():
    return render_template('about.html')


def getheight(path):
    img = skimage.io.imread(path)
    h, w, _ = img.shape
    aspect = h/w
    given_width = 200
    height = given_width*aspect
    return height


def pipeline_model(path, scaler_transform, model_sgd):
    # pipeline model
    image = skimage.io.imread(path)
    # transform image into 80 x 80
    image_resize = skimage.transform.resize(image, (80, 80))
    image_scale = 255*image_resize
    image_transform = image_scale.astype(np.uint8)
    # rgb to gray
    gray = skimage.color.rgb2gray(image_transform)
    # hog feature
    feature_vector = skimage.feature.hog(gray,
                                         orientations=10,
                                         pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    # scaling

    scalex = scaler_transform.transform(feature_vector.reshape(1, -1))
    result = model_sgd.predict(scalex)
    # decision function # confidence
    decision_value = model_sgd.decision_function(scalex).flatten()
    labels = model_sgd.classes_
    # probability
    z = scipy.stats.zscore(decision_value)
    prob_value = scipy.special.softmax(z)

    # top 5
    top_5_prob_ind = prob_value.argsort()[::-1][:5]
    top_labels = labels[top_5_prob_ind]
    top_prob = prob_value[top_5_prob_ind]
    # put in dictornary
    top_dict = dict()
    for key, val in zip(top_labels, top_prob):
        top_dict.update({key: np.round(val, 3)})

    return top_dict


if __name__ == '__main__':
    app.run()
