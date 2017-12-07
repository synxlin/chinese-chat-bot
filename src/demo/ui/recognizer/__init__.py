#/usr/bin/env python3
#-*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
from threading import Thread
from queue import Queue, Empty
from time import sleep

from .data_loader import SpectrogramParser
from .model import DeepSpeech
from .decoder import *

checkpoint = torch.load('ui/recognizer/resource/best_model.pth.tar')
model = DeepSpeech.load_model_checkpoint(checkpoint)
labels = DeepSpeech.get_labels(model)
audio_conf = DeepSpeech.get_audio_conf(model)
audio_conf['noise_dir'] = None

parser = SpectrogramParser(audio_conf, normalize=True)
#decoder = BeamDecoder(labels, lm_path='resource/ngram.arpa', alpha=0.89,
#                      beam_width=20)
decoder = GreedyDecoder(labels)
model = model.cuda()


def recognize(audio_path):
    spect = parser.parse_audio(audio_path).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1)).cuda()
    print('recognizing')
    out = model(Variable(spect, volatile=True))
    out = out.transpose(0, 1)
    torch.cuda.synchronize()
    decoded_output, _ = decoder.decode(out.data.cpu())
    del out, spect
    return decoded_output


class Recognizer(object):
    """ Speech Recognition Interface """

    def __init__(self):
        self.status = 'off'

        self._speech_queue = Queue()
        self._result_queue = Queue()

        self._interval = 0.1  # sec
        self._recv_buffer_size = 4096

        self._root = 'audio/'

    def on(self):
        assert self.status == 'off'
        self.status = 'on'
        Thread(target=self._process_loop).start()

    def off(self):
        assert self.status == 'on'
        self.status = 'off'

    def put_speech(self, speech):
        assert self.status == 'on'
        self._speech_queue.put(speech)

    def get_result_nowait(self):
        assert self.status == 'on'
        return self._result_queue.get_nowait()

    def _process_loop(self):
        while self.status =='on':
            try:
                while True:
                    speech = self._speech_queue.get_nowait()
                    result = recognize(speech)
                    print(result)
                    result = result[0][0]
                    self._result_queue.put(result)
            except Empty:
                sleep(self._interval)

