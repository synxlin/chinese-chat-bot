import datetime
import os
import argparse
import math
from collections import OrderedDict


def add_model_options(parser):
    parser.add_argument('--rnn-type', default='lstm', choices=['rnn', 'lstm', 'gru'],
                        help='Type of the RNN. rnn|gru|lstm are supported')
    parser.add_argument('--hidden-size', type=int, help='Hidden size of RNNs')
    parser.add_argument('--hidden-layers', type=int, help='Number of RNN layers')
    parser.add_argument('--look-ahead', dest='look_ahead', action='store_true',
                        default=False, help='Turn off bi-directional RNNs,'
                        'introduces lookahead convolution')


def add_optimization_options(parser):
    parser.add_argument('--lr', '--learning-rate', dest='lr', type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--mt', '--momentum', dest='momentum',
                        type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', dest='weight_decay',
                        type=float, metavar='WD', help='weight decay')
    parser.add_argument('--lr-anneal', type=float, metavar='LR_ANNEAL',
                        help='every epoch, learning rate decays by lr_anneal')
    parser.add_argument('--max-norm', type=int,
                        help='Norm cutoff to prevent explosion of gradients')


def add_data_options(parser):
    parser.add_argument('--train-manifest', metavar='DIR',
                        help='path to train manifest csv',
                        default='data/datasets/train_manifest.csv')
    parser.add_argument('--val-manifest', metavar='DIR',
                        help='path to validation manifest csv',
                        default='data/datasets/val_manifest.csv')
    parser.add_argument('--labels-path', metavar='PATH',
                        default='data/datasets/labels.json',
                        help='Contains all characters for prediction')
    parser.add_argument('--in-order', default=False, action='store_true',
                        help='Turn off shuffling and sample from dataset based '
                       'on sequence length (smallest to largest)')
    parser.add_argument('--sample-rate', default=16000, type=int,
                        help='Audio Sample rate')
    parser.add_argument('--window-size', default=.02, type=float,
                        help='Window size for spectrogram in seconds')
    parser.add_argument('--window-stride', default=.01, type=float,
                        help='Window stride for spectrogram in seconds')
    parser.add_argument('--window', default='hamming',
                        help='Window type for spectrogram generation')
    parser.add_argument('--augment', default=False, action='store_true',
                        help='Use random tempo and gain perturbations.')
    parser.add_argument('--noise-dir', default=None,
                        help='Directory to inject noise into audio.'
                        'If default, noise Inject not added')
    parser.add_argument('--noise-prob', default=0.4,
                        help='Probability of noise being added per sample')
    parser.add_argument('--noise-min', default=0.0, type=float,
                        help='Minimum noise level to sample from.'
                        '(1.0 means all noise, not original signal)')
    parser.add_argument('--noise-max', default=0.5,
                        help='Maximum noise levels to sample from. Maximum 1.0', type=float)


def add_train_options(parser):
    parser.add_argument('-j', '--num-workers', default=12, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('-n', '--nGPU', dest='nGPU', default=4, type=int,
                        metavar='N', help='number of GPUs to use')
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int,
                        metavar='BS', help='mini-batch size')
    parser.add_argument('--epochs', type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('--pf', '--print-freq', dest='print_freq', default=10,
                        type=int, metavar='T', help='print frequency (default: 10)')
    parser.add_argument('--visdom', default=False, action='store_true',
                        help='Turn on visdom graphing')


def add_test_options(parser):
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--decoder', choices=['greedy', 'beam', 'my'],
                        type=str, help='Decoder to use')
    parser.add_argument('--beam_width', default=10, type=int, help='Beam width to use')
    parser.add_argument('--lm_path', type=str,
                        help='Path to an kenlm language model for use with beam search')
    parser.add_argument('--lm_alpha', type=float, help='Language model weight')
    # parser.add_argument('--lm_beta', default=1, type=float,
    #                     help='Language model word bonus (all words)')
    # parser.add_argument('--label_size', default=0, type=int,
    #                     help='Label selection size controls how many items in '
    #                          'each beam are passed through to the beam scorer')
    # parser.add_argument('--label_margin', default=-1, type=float,
    #                     help='Controls difference between minimal input score '
    #                          'for an item to be passed to the beam scorer.')


def create_config(**options):
    config = OrderedDict()
    config['arch'] = options['rnn_type'] + str(options['hidden_layers'])
    config['hidden_size'] = options['hidden_size'] if 'hidden_size' in options else 1024
    config['look_ahead'] = options['look_ahead'] if 'look_ahead' in options else False
    config['lr'] = options['lr']
    config['momentum'] = options['momentum'] if 'momentum' in options else 0.0
    config['weight_decay'] = options['weight_decay'] if 'weight_decay' in options else 0.0
    config['lr_anneal'] = options['lr_anneal'] if 'lr_anneal' in options else 1.1
    config['decoder'] = options['decoder'] if 'decoder' in options else 'greedy'
    config['beam_width'] = options['beam_width'] if 'beam_width' in options else 10
    config['lm_path'] = options['lm_path'] if 'lm_path' in options else None
    config['lm_alpha'] = options['lm_alpha'] if 'lm_alpha' in options else 0.8
    config['resume'] = options['resume'] if 'resume' in options else ''
    config['pretrained'] = options['pretrained'] if 'pretrained' in options else ''
    config['evaluate'] = options['evaluate'] if 'evaluate' in options else False

    config_str = 'Training Config:\n'
    for key, val in config.items():
        config_str += '{}: {}\n'.format(key, val)

    print('='*89)
    print(config_str)
    alg_id = config['arch'] + '_' + str(config['hidden_size']) + '_' + \
             datetime.datetime.now().strftime('%m%d_%H%M')
    print('saving to %s' % alg_id)
    print('='*89)

    if config['evaluate']:
        config['log_dir'] = os.path.join('evaluate', alg_id)
    else:
        config['log_dir'] = os.path.join('logs', alg_id)
        config['checkpoint_dir'] = os.path.join('checkpoints', alg_id)
    config['experiment_id'] = alg_id
    config['config_str'] = config_str

    return config


class Options:
    def __init__(self, description):
        self.parser = argparse.ArgumentParser(description=description)
        add_model_options(self.parser)
        add_optimization_options(self.parser)
        add_data_options(self.parser)
        add_train_options(self.parser)
        add_test_options(self.parser)

    def set_defaults(self, **kwargs):
        self.parser.set_defaults(**kwargs)

    def parse_args(self, *args):
        args = self.parser.parse_args(*args)
        for k, v in vars(args).items():
            self.__setattr__(k, v)
        config = create_config(**vars(self))
        for k, v in config.items():
            self.__setattr__(k, v)
        return self
