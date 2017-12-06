#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
# Modified to support pytorch Tensors

import Levenshtein as Lev
import torch
from six.moves import xrange


class Decoder(object):
    """
    Basic decoder class from which all other decoders inherit. Implements several
    helper functions. Subclasses should implement the decode() method.

    Arguments:
        labels (string): mapping from integers to characters.
        blank_index (int, optional): index for the blank '_' character. Defaults to 0.
        space_index (int, optional): index for the space ' ' character. Defaults to 28.
    """

    def __init__(self, labels, blank_index=0):
        # e.g. labels = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ#"
        self.labels = labels
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
        self.blank_index = blank_index
        space_index = len(labels)  # To prevent errors in decode, we add an out of bounds index for the space
        if ' ' in labels:
            space_index = labels.index(' ')
        self.space_index = space_index

    def wer(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        return Lev.distance(''.join(w1), ''.join(w2))

    def cer(self, s1, s2):
        """
        Computes the Character Error Rate, defined as the edit distance.

        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
        return Lev.distance(s1, s2)

    def decode(self, probs, sizes=None):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription

        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            string: sequence of the model's best guess for the transcription
        """
        raise NotImplementedError


class BeamCTCDecoder(Decoder):
    def __init__(self, labels, lm_path=None, alpha=0, beta=0, cutoff_top_n=40, cutoff_prob=1.0, beam_width=100,
                 num_processes=4, blank_index=0):
        super(BeamCTCDecoder, self).__init__(labels)
        try:
            from ctcdecode import CTCBeamDecoder
        except ImportError:
            raise ImportError("BeamCTCDecoder requires paddledecoder package.")
        self._decoder = CTCBeamDecoder(labels, lm_path, alpha, beta, cutoff_top_n, cutoff_prob, beam_width,
                                       num_processes, blank_index)

    def convert_to_strings(self, out, seq_len):
        results = []
        for b, batch in enumerate(out):
            utterances = []
            for p, utt in enumerate(batch):
                size = seq_len[b][p]
                if size > 0:
                    transcript = ''.join(map(lambda x: self.int_to_char[x], utt[0:size]))
                else:
                    transcript = ''
                utterances.append(transcript)
            results.append(utterances)
        return results

    def convert_tensor(self, offsets, sizes):
        results = []
        for b, batch in enumerate(offsets):
            utterances = []
            for p, utt in enumerate(batch):
                size = sizes[b][p]
                if sizes[b][p] > 0:
                    utterances.append(utt[0:size])
                else:
                    utterances.append(torch.IntTensor())
            results.append(utterances)
        return results

    def decode(self, probs, sizes=None):
        """
        Decodes probability output using ctcdecode package.
        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes: Size of each sequence in the mini-batch
        Returns:
            string: sequences of the model's best guess for the transcription
        """
        probs = probs.cpu().transpose(0, 1).contiguous()
        out, scores, offsets, seq_lens = self._decoder.decode(probs)

        strings = self.convert_to_strings(out, seq_lens)
        offsets = self.convert_tensor(offsets, seq_lens)
        return strings, offsets


class BeamDecoder(Decoder):
    """docstring for BeamDecoder"""
    def __init__(self, labels, lm_path=None, alpha=0, beam_width=100, blank_index=0):
        super(BeamDecoder, self).__init__(labels)
        try:
            import kenlm
            import math
            import copy
        except ImportError:
            raise ImportError("BeamDecoder requires kenlm package.")
        self.lm = kenlm.Model(lm_path)
        self.alpha, self.beam_width = alpha * math.log(10), beam_width

    def process_string(self, seq_probs, seq_ids, size, remove_repetitions=False):
        beam_width, alpha = self.beam_width, self.alpha
        strings = [''] * beam_width
        offsets = [[] for _ in range(beam_width)]
        prev_idx = [0] * beam_width
        ctc_scores = [0] * beam_width
        for i in range(size):  # seq

            tmp_strings, tmp_offsets, tmp_prev_idx, tmp_ctc_scores, tmp_scores = [], [], [], [], []
            for j in range(beam_width):  # prev
                for k in range(beam_width):  # now
                    string = strings[j]
                    offset = copy.copy(offsets[j])
                    ctc_score = ctc_scores[j]
                    char_idx = seq_ids[i][k]  # seq_ids shape of seq_length x output_dim(beam_width)
                    if char_idx != self.blank_index:
                        # if this char_idx is a repetition and remove_repetitions=true, then skip
                        if remove_repetitions and i != 0 and char_idx == prev_idx[j]:
                            pass
                        elif char_idx == self.space_index:
                            string += ' '
                            offset.append(i)
                        else:
                            string += self.int_to_char[char_idx]
                            offset.append(i)
                    ctc_score += seq_probs[i][k]
                    score = ctc_score + self.lm.score(string, bos=False, eos=False) * alpha
                    tmp_strings.append(string)
                    tmp_offsets.append(offset)
                    tmp_prev_idx.append(char_idx)
                    tmp_ctc_scores.append(ctc_score)
                    tmp_scores.append(score)
            beam_ids = sorted(range(len(tmp_scores)), key=lambda k: tmp_scores[k], reverse=True)
            beam_ids = beam_ids[:beam_width]
            strings = [tmp_strings[k] for k in beam_ids]
            offsets = [tmp_offsets[k] for k in beam_ids]
            prev_idx = [tmp_prev_idx[k] for k in beam_ids]
            ctc_scores = [tmp_ctc_scores[k] for k in beam_ids]
        string = strings[0]
        offset = offsets[0]
        return string, torch.IntTensor(offset)

    def decode(self, probs, sizes=None):
        """
        Returns the argmax decoding given the probability matrix. Removes
        repeated elements in the sequence, as well as blanks.

        Arguments:
            probs: Tensor of character probabilities from the network. Expected shape of seq_length x batch x output_dim
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            strings: sequences of the model's best guess for the transcription on inputs
            offsets: time step per character predicted
        """
        topk_probs, topk_ids = torch.topk(probs.transpose(0,1), self.beam_width, dim=2, largest=True, sorted=True)
        topk_probs, topk_ids = topk_probs.cpu(), topk_ids.cpu()
        batch_size = topk_probs.size(0)
        strings, offsets = [], []
        for x in xrange(batch_size):
        # score = self.lm.score(data, bos=False, eos=False) * math.log(10)
            seq_len = sizes[x] if sizes is not None else len(topk_probs[x])
            string, string_offsets = self.process_string(topk_probs[x], topk_ids[x], seq_len, remove_repetitions=True)
            strings.append([string])
            offsets.append([string_offsets])
        return strings, offsets


class GreedyDecoder(Decoder):
    def __init__(self, labels, blank_index=0):
        super(GreedyDecoder, self).__init__(labels, blank_index)

    def convert_to_strings(self, sequences, sizes=None, remove_repetitions=False, return_offsets=False):
        """Given a list of numeric sequences, returns the corresponding strings"""
        strings = []
        offsets = [] if return_offsets else None
        for x in xrange(len(sequences)):
            seq_len = sizes[x] if sizes is not None else len(sequences[x])
            string, string_offsets = self.process_string(sequences[x], seq_len, remove_repetitions)
            strings.append([string])  # We only return one path
            if return_offsets:
                offsets.append([string_offsets])
        if return_offsets:
            return strings, offsets
        else:
            return strings

    def process_string(self, sequence, size, remove_repetitions=False):
        string = ''
        offsets = []
        for i in range(size):
            char_idx = sequence[i]
            if char_idx != self.blank_index:
                # if this char_idx is a repetition and remove_repetitions=true, then skip
                if remove_repetitions and i != 0 and char_idx == sequence[i - 1]:
                    pass
                elif char_idx == self.space_index:
                    string += ' '
                    offsets.append(i)
                else:
                    string += self.int_to_char[char_idx]
                    offsets.append(i)
        return string, torch.IntTensor(offsets)

    def decode(self, probs, sizes=None):
        """
        Returns the argmax decoding given the probability matrix. Removes
        repeated elements in the sequence, as well as blanks.

        Arguments:
            probs: Tensor of character probabilities from the network. Expected shape of seq_length x batch x output_dim
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            strings: sequences of the model's best guess for the transcription on inputs
            offsets: time step per character predicted
        """
        _, max_probs = torch.max(probs.transpose(0, 1), 2)
        strings, offsets = self.convert_to_strings(max_probs.view(max_probs.size(0), max_probs.size(1)), sizes,
                                                   remove_repetitions=True, return_offsets=True)
        return strings, offsets
