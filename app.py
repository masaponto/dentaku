#!/usr/bin/env python
# -*- coding: utf-8 -*-

# reference https://github.com/ginrou/handwritten_classifier/blob/master/app.py

import tensorflow as tf
import numpy as np

import sys
sys.path.append('mnist')
import model

x = tf.placeholder("float", [None, 784])
sess = tf.Session()


with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2, variables = model.convolutional(x, keep_prob)

saver = tf.train.Saver(variables)
#saver.restore(sess, "mnist/data/convolutional_plus.ckpt")
saver.restore(sess, "mnist/data/convolutional_tmp.ckpt")

mark_dict = {10: '+',
             11: '-',
             12: '*',
             13: '/'}


def convolutional(input):
    y = sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()
    return y.index(max(y))


from flask import Flask, render_template, request, jsonify
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/estimate", methods=["POST"])
def estimate():
    try:
        input = (np.array(request.json["input"],
                          dtype=np.uint8) / 255.0).reshape(1, 784)
        output = convolutional(input)
        if output in mark_dict:
            output = mark_dict[output]
        return jsonify({"estimated": output})

    except Exception as e:
        print e
        return jsonify({"error": e})


@app.route("/generate", methods=["POST"])
def array2csv():
    try:
        input_data = (np.array(request.json["input"],
                               dtype=np.uint8)).reshape(1, 784)
        lst = input_data.tolist()
        return jsonify({"vec": lst})

    except Exception as e:
        print e
        return jsonify({"error": e})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
