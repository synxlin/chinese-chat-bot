#/usr/bin/env python3
#-*- coding: utf-8 -*-

import os
import time
import json
import random
from threading import Thread
from flask import Flask, render_template
from .utils import Controller


app_name = 'Voice Assistant'
app = Flask(app_name, template_folder='ui/templates')

#remote_addr = '166.111.224.152'
#remote_port = 5141
#remote_user = 'linyy14'
#remote_path = '~/Documents/speech_recognition/demo/server/audio/'

local_file_root = 'audio/'

controller = Controller()
controller.recorder.auto_set_threshold()
controller.recorder.set_save_root(local_file_root)

@app.route('/')
def index():
    global controller
    # controller.stop()
    controller.clear_texts()
    print('ok')
    page = 'online.html'
    return render_template(page)

@app.route('/online/startstop', methods=['POST', 'GET'])
def online_start_stop():
    global controller

    if time.time() - controller.timer > 0.5:  # 0.5 sec
        controller.timer = time.time()

        if controller._status is None:
            controller.clear_texts()
            controller.online()
        elif controller._status == 'online':
            controller.stop()
        else:
            assert False

    if controller._status is None:
        return 'off'
    if controller._status == 'online':
        return 'on'

@app.route('/online/text1', methods=['GET'])
def online_text():
    global controller
    texts = controller.get_texts()
    return json.dumps([' '.join(texts)])

@app.route('/online/text2', methods=['GET'])
def online_response():
    global controller
    texts = controller.get_response()
    return json.dumps([' '.join(texts)])
