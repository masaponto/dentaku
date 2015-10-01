#!/usr/bin/env python
# -*- coding: utf-8 -*-

# reference https://github.com/ginrou/handwritten_classifier/blob/master/app.py

from flask import Flask, render_template, request, jsonify

import numpy
from elm.elm import ELM

app = Flask(__name__)

elm = ELM(1000)
elm.load_weights()

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/estimate", methods = ["POST"])
def estimate():
    try:
        x = numpy.array(request.json["input"]) / 255.0
        y = int(elm.one_predict(x))
        print(y)
        return jsonify({"estimated":y})
    except Exception as e:
        print(e)
        return jsonify({"error":e})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 8080)
