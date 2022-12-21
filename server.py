import json
import numpy as np
from topics import get_metrics
from load_model import get_models
import os.path
from pydantic import BaseModel
from typing import List, Union
from flask import Flask, request,Response, jsonify
from flask_cors import cross_origin

class Source(BaseModel):
    text: str
    dataset: str = "climate_change"

class Answer(BaseModel):
    fluency: float
    flexibility: float
    originality: float
    topics: List[int]

app = Flask(__name__)


origins = [
    "*"
]

@app.route("/test", methods=['GET'])
def home():
    return "Hello flask"

@app.route('/creativity', methods=['POST'])
@cross_origin()
def creativity():
    if request.method == 'POST':
        text = request.json['text']
        fluency, flexibility, originality, topics = get_metrics(text, dataset="climate_change")


        response= {}
        response['fluency'] = fluency
        response['flexability'] = flexibility
        response['originality'] = originality
        response['topics'] = str(topics)
        #answer = Answer(fluency=fluency,
         #               flexibility=flexibility,
          #              originality=originality,
           #             topics=list(topics))
        response = json.dumps(response)
        return Response(response,status=200)

@app.route('/models', methods=['GET'])
@cross_origin()
def load_models():
    response = "Error"
    # check if file exists
    if os.path.isfile('./models/climate_change_model'):
        response = "Files exists"
    else:
        get_models()
        response = "Downloaded files"
    return response
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
