#/usr/bin/env python3
#-*- coding: utf-8 -*-

from queue import Queue, Empty

from .xiaobing import xiaobing
from .preprocess import preprocess


class Jarvis(object):
    def __init__(self):
        self._question_queue = Queue()
        self._response_queue = Queue()
        xiaobing(self._question_queue, self._response_queue)
    
    def put_question(self, question):
        self._question_queue.put(preprocess(question))

    def get_response(self):
        return self._response_queue.get()

    def get_nowait_response(self):
        return self._response_queue.get_nowait()

    def off(self):
        try:
            while True:
                self._question_queue.get_nowait()
        except Empty:
            pass
        try:
            while True:
                self._response_queue.get_nowait()
        except Empty:
            pass
