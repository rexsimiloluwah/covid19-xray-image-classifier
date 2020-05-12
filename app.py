from flask import Flask, render_template, jsonify, request
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
# import warnings
# warnings.filterwarnings('ignore')
import numpy as np
import cv2
import matplotlib.image as mpimg 
from tensorflow.python.keras.utils import CustomObjectScope
from tensorflow.python.keras.initializers import glorot_uniform
from tensorflow.keras import optimizers
import tensorflow as tf
from tensorflow.python.keras.models import load_model, model_from_json
from tensorflow.python.framework import ops
ops.reset_default_graph()

import os


app = Flask(__name__)

import logging

logger = logging.getLogger('root')


ops.reset_default_graph()

import os

class Model:
    def __init__(self):
        self.session = tf.Session()
        self.graph = tf.get_default_graph()
        # the folder in which the model and weights are stored
        self.model_folder = './saved_models'
        self.loaded_model = None
        # for some reason in a flask app the graph/session needs to be used in the init else it hangs on other threads
        with self.graph.as_default():
            with self.session.as_default():
                logging.info("neural network initialised")
            
    def load(self, file_name=None):
        with self.graph.as_default():
            with self.session.as_default():
                try:
                    model_name = file_name[0]
                    weights_name = file_name[1]

                    if model_name is not None:
                        # load the model
                        json_file_path = os.path.join(self.model_folder, model_name)
                        json_file = open(json_file_path, 'r')
                        loaded_model_json = json_file.read()
                        json_file.close()
                        self.loaded_model = model_from_json(loaded_model_json)
                    if weights_name is not None:
                        # load the weights
                        weights_path = os.path.join(self.model_folder, weights_name)
                        self.loaded_model.load_weights(weights_path)
                        logging.info("Neural Network loaded: ")
                        logging.info('\t' + "Neural Network model: " + model_name)
                        logging.info('\t' + "Neural Network weights: " + weights_name)
                        return True
                except Exception as e:
                    logging.exception(e)
                    return False

    def predict(self, path_name):
        with self.graph.as_default():
            with self.session.as_default():
                img = cv2.cvtColor(cv2.imread(path_name), cv2.COLOR_BGR2RGB)
                resized_image = cv2.resize(img, (450,450), interpolation = cv2.INTER_CUBIC)
                image = resized_image.reshape(1,450,450,3)
                image_scaled = image/255
                prediction = self.loaded_model.predict(image_scaled)
                prediction = (prediction.reshape(1)[0])*100
                
                if prediction <= 50:
                    return "Your COVID19 Status is -VE with a probability of {} %".format(prediction)
                else:
                    return "Unfortunately, Your COVID19 Status is +VE with a probability of {} %".format(prediction)


@app.route("/")
def home():
    
    return render_template('index.html')

@app.route("/predict", methods = ['GET', 'POST'])
def predict():

    if request.method == "POST":

        #Get the file uploaded from the request 
        uploaded_file = request.files['file']


        #Save the file to ./uploaded_images
        basedir = os.path.dirname(__file__)

        uploads_path = os.path.join(basedir, 'uploaded_images', secure_filename(uploaded_file.filename))
        uploaded_file.save(uploads_path)

        # Prediction 
        try:
            model = Model()
            model.load(['model.json', 'model_weights.h5'])


            prediction = model.predict(uploads_path)

            return render_template('index.html', prediction = prediction)
        except:
            return render_templates('index.html', prediction = "Please Upload a Valid Image")

        
    return ""

       



if __name__ == '__main__':
    
    app.run(debug = True)

