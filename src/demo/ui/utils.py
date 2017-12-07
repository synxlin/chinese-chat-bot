#/usr/bin/env python3
#-*- coding: utf-8 -*-

import gc
import pyaudio
import wave
import numpy as np
from os import path
import subprocess
from queue import Queue, Empty
from threading import Thread, Lock
from time import sleep

from .recognizer import Recognizer
from .jarvis import Jarvis


class VoiceRecorder(object):
    """ Realtime Recorder """
    def __init__(self):
        self.status = 'off'

        self._pyaudio = pyaudio.PyAudio()
        self._stream = None
        self._speech_queue = Queue()
        self._frame_queue = Queue()
        self._save_root = 'audio/'

        # voice format
        self._format = pyaudio.paInt16
        self._threshold = 500
        self._rate = 16000
        self._frame_size = 1024  # 1024 / 16000 = 0.064s
        self._channels = 1
        self._frame_length = float(self._frame_size) / float(self._rate)

        # speech
        self._min_sentence_length = 0.5  # sec
        self._min_sentence_frame_num = int(self._min_sentence_length / self._frame_length)
        self._min_pause_length = 0.5 # pause between sentences, sec
        self._min_pause_frame_num = int(self._min_pause_length / self._frame_length)
        # self._max_buffer_length = 2
        # self._max_buffer_frame_num = self._max_buffer_length / self._frame_length

        self._power_threshold = 0.0002
        self._zcr_threshold = 0.05
        self._auto_threshold_length = 2  # sec
        self._auto_threshold_frame_num = int(self._auto_threshold_length / self._frame_length)
        self._auto_threshold_dropout = 0.5
        self._auto_threshold_power_mult = 3
        self._auto_threshold_zcr_mult = 3

        self._noise = []
        self._noise_frame_num = 10

        # stream lock
        self.lock = Lock()

    def save(self, frame, filename):
        path = self._save_root + filename
        with wave.open(path, 'wb') as fout:
            fout.setparams((self._channels, 2, self._rate, 0, 'NONE', 'not compressed'))
            fout.writeframes(frame)
        return path

    def on(self, frame_preprocess=True):
        assert self.status == 'off'

        # start audio stream
        self._stream = self._pyaudio.open(format=self._format, \
            channels=self._channels, rate=self._rate, input=True, \
            output=False, frames_per_buffer=self._frame_size)

        # start recording
        self.status = 'on'

        Thread(target=self._record).start()

        if frame_preprocess:
            Thread(target=self._frame_preprocess).start()

    def off(self):
        # assert self.status == 'on'

        self.status = 'off'
        #if self._stream is not None:
        #    self._stream.close()
        #    self._stream = None

        # clear queue
        try:
            while True:
                self._frame_queue.get_nowait()
        except Empty:
            pass

        try:
            while True:
                self._speech_queue.get_nowait()
        except Empty:
            pass

    def auto_set_threshold(self):
        assert self.status == 'off'

        print('auto setting threshold.')

        self.on(frame_preprocess=False)

        powers = []
        zcrs = []
        for i in range(self._auto_threshold_frame_num):
            frame = self._frame_queue.get()
            power, zcr = self._frame_power_zcr(frame)
            powers.append(power)
            zcrs.append(zcr)

        self.off()

        powers.sort()
        zcrs.sort()

        dropout = self._auto_threshold_dropout
        dropout_st = int(len(powers)*dropout*0.5)
        dropout_ed = int(len(powers)*(1 - dropout*0.5))

        powers = powers[dropout_st:dropout_ed]
        zcrs = zcrs[dropout_st:dropout_ed]

        self._power_threshold = self._auto_threshold_power_mult * sum(powers) / len(powers)
        self._zcr_threshold = self._auto_threshold_zcr_mult * sum(zcrs) / len(zcrs)

        print('power threshold:', self._power_threshold)
        print('zcr threshold:', self._zcr_threshold)

    def get_speech_nowait(self):
        return self._speech_queue.get_nowait()

    def set_save_root(self, root):
        self._save_root = root

    def _record(self):
        while self.status == 'on':  # read only, thread safe
            assert self._stream is not None
            frame = self._stream.read(self._frame_size)
            self._frame_queue.put(frame)
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def _frame_preprocess(self):  # frame -> sentences
        speech_frames = []
        background_frames = []
        while self.status == 'on':
            try:
                while True:
                    frame = self._frame_queue.get_nowait()
                    is_speech = self._is_speech(frame)
                    if is_speech:
                        if len(speech_frames) == 0 or len(background_frames) == 0:
                            speech_frames.append(frame)
                            background_frames.clear()
                        elif len(speech_frames) > 0 and len(background_frames) > 0:
                            speech_frames.extend(background_frames)
                            speech_frames.append(frame)
                            background_frames.clear()
                        else:
                            assert False  # impossible

                    if not is_speech:
                        if len(self._noise) == self._noise_frame_num:
                            self._noise = self._noise[1:]
                        self._noise.append(frame)  # modeling background noise

                        if len(speech_frames) == 0:
                            pass  # Do nothing
                        elif len(speech_frames) > 0:
                            background_frames.append(frame)

                    if len(background_frames) > self._min_pause_frame_num:
                        if len(speech_frames) > self._min_sentence_frame_num:
                            sentence = self._concat_frames(speech_frames)
                            # denoise
                            if self._noise:
                               sentence = self._denoise(sentence)
                            self._speech_queue.put(sentence)
                            self.status = 'off'
                        background_frames.clear()
                        speech_frames.clear()
            except Empty:
                sleep(self._frame_length)


    def _frame_power_zcr(self, frame):
        numdata = self._frame_to_nparray(frame)
        power = self._power(numdata)
        zcr = self._zcr(numdata)
        return power, zcr

    def _frame_to_nparray(self, frame):
        assert self._format == pyaudio.paInt16
        numdata = np.fromstring(frame, dtype=np.int16)
        numdata = numdata / 2**15  # max val of int16 = 2**15-1
        return numdata

    def _nparray_to_frame(self, numdata):
        numdata = numdata * 2**15
        numdata = numdata.astype(np.int16)
        frame = numdata.tobytes()
        return frame
        

    def _power(self, numdata):
        return np.mean(numdata**2)

    def _zcr(self, numdata):
        zc = numdata[1:] * numdata[:-1] < 0
        zcr = sum(zc) / len(zc)
        return zcr

    def _is_speech(self, frame):
        power, zcr = self._frame_power_zcr(frame)
        voiced_sound = power > self._power_threshold
        unvoiced_sound =  zcr > self._zcr_threshold
        return voiced_sound or unvoiced_sound

    def _concat_frames(self, frames):
        return b''.join(frames)


    def _denoise(self, speech):
        # Spectral Subtraction
        speech_val = self._frame_to_nparray(speech)
        noise_val = self._frame_to_nparray(b''.join(self._noise))

        speech_fft_mag = np.abs(np.fft.fft(speech_val))
        noise_fft_mag = np.abs(np.fft.fft(noise_val))

        speech_freq = np.linspace(0, self._rate, len(speech_val))
        noise_freq = np.linspace(0, self._rate, len(noise_val))

        noise_fft_interp = np.interp(speech_freq, noise_freq, noise_fft_mag)

        denoised_fft_mag = np.maximum(speech_fft_mag - noise_fft_interp, np.zeros(speech_fft_mag.shape))

        denoised_fft = np.fft.fft(speech_val) * denoised_fft_mag / speech_fft_mag

        denoised_val = np.real(np.fft.ifft(denoised_fft))

        denoised = self._nparray_to_frame(denoised_val)
        return denoised


class Controller(object):
    def __init__(self):
        self.recognizer = Recognizer()
        self.recorder = VoiceRecorder()

        self.jarvis = Jarvis()

        self.timer = 0
        self._status = None  # None, 'online'
        self._texts = []
        self._responses = []
        self._cnt = 0

        self._interval = 0.1

    def get_texts(self):
        return self._texts[:]

    def get_response(self):
        return self._responses[:]

    def clear_texts(self):
        assert self._status == None  # or use a mutex
        self._texts = []
        self._responses = []

    def online(self):
        assert self._status == None
        self._cnt = 0
        self._status = 'online'
        self.recorder.on()
        self.recognizer.on()
        Thread(target=self._online_loop).start()

    def stop(self):
        status = self._status
        self._status = None
        if status == 'online':
            self.recorder.off()
            self.recognizer.off()
            self.jarvis.off()

    def _online_loop(self):
        while self._status == 'online':
            result = None
            speech = None
            try:
                result = self.recognizer.get_result_nowait()
            except Empty:
                pass

            try:
                speech = self.recorder.get_speech_nowait()
            except Empty:
                pass

            if result:
                self._texts.append(result)
                self.jarvis.off()
                self.jarvis.put_question(result)
                response = self.jarvis.get_response()
                self._responses.append(response + '\n')
                gc.collect()
                subprocess.run(['ekho','\" %s\"' % response])
                sleep(self._interval)
                # self.stop()
            if speech and self._cnt == 0:
                filename = 'data.wav'
                filepath = self.recorder.save(speech, filename)
                print('saving to ', filename)
                self.recognizer.put_speech(filepath)
                self._cnt += 1
            if not result and not speech:
                sleep(self._interval)
