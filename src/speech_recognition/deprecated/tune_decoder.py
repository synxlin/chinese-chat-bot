import argparse
import json
import sys
from multiprocessing import Pool

import numpy as np
import torch

from data.data_loader import SpectrogramDataset, AudioDataLoader
from decoder import GreedyDecoder, BeamCTCDecoder
from model import DeepSpeech

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--logits', default="", type=str, help='Path to logits from test.py')
parser.add_argument('--test_manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--num_workers', default=16, type=int, help='Number of parallel decodes to run')
parser.add_argument('--output_path', default="tune_results.json", help="Where to save tuning results")
beam_args = parser.add_argument_group("Beam Decode Options", "Configurations options for the CTC Beam Search decoder")
beam_args.add_argument('--beam_width', default=10, type=int, help='Beam width to use')
beam_args.add_argument('--lm_path', default=None, type=str,
                       help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)')
beam_args.add_argument('--lm_alpha_from', default=1, type=float, help='Language model weight start tuning')
beam_args.add_argument('--lm_alpha_to', default=3.2, type=float, help='Language model weight end tuning')
beam_args.add_argument('--lm_beta_from', default=0.0, type=float,
                       help='Language model word bonus (all words) start tuning')
beam_args.add_argument('--lm_beta_to', default=0.45, type=float,
                       help='Language model word bonus (all words) end tuning')
beam_args.add_argument('--lm_num_alphas', default=45, type=float, help='Number of alpha candidates for tuning')
beam_args.add_argument('--lm_num_betas', default=8, type=float, help='Number of beta candidates for tuning')
beam_args.add_argument('--cutoff_top_n', default=40, type=int,
                       help='Cutoff number in pruning, only top cutoff_top_n characters with highest probs in '
                            'vocabulary will be used in beam search, default 40.')
beam_args.add_argument('--cutoff_prob', default=1.0, type=float,
                       help='Cutoff probability in pruning,default 1.0, no pruning.')

args = parser.parse_args()


def decode_dataset(logits, test_dataset, batch_size, lm_alpha, lm_beta, mesh_x, mesh_y, labels):
    print("Beginning decode for {}, {}".format(lm_alpha, lm_beta))
    test_loader = AudioDataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    target_decoder = GreedyDecoder(labels, blank_index=labels.index('_'))
    decoder = BeamCTCDecoder(labels, beam_width=args.beam_width, cutoff_top_n=args.cutoff_top_n,
                             blank_index=labels.index('_'), lm_path=args.lm_path,
                             alpha=lm_alpha, beta=lm_beta, num_processes=1)
    total_cer, total_wer = 0, 0
    for i, (data) in enumerate(test_loader):
        inputs, targets, input_percentages, target_sizes = data

        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        out = torch.from_numpy(logits[i][0])
        sizes = torch.from_numpy(logits[i][1])

        decoded_output, _, _, _ = decoder.decode(out, sizes)
        target_strings = target_decoder.convert_to_strings(split_targets)
        wer, cer = 0, 0
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            wer_inst = decoder.wer(transcript, reference) / float(len(reference.split()))
            cer_inst = decoder.cer(transcript, reference) / float(len(reference))
            wer += wer_inst
            cer += cer_inst
        total_cer += cer
        total_wer += wer

    wer = total_wer / len(test_loader.dataset)
    cer = total_cer / len(test_loader.dataset)

    return [mesh_x, mesh_y, lm_alpha, lm_beta, wer, cer]


if __name__ == '__main__':
    if args.lm_path is None:
        print("error: LM must be provided for tuning")
        sys.exit(1)

    model = DeepSpeech.load_model(args.model_path, cuda=False)
    model.eval()

    labels = DeepSpeech.get_labels(model)
    audio_conf = DeepSpeech.get_audio_conf(model)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.test_manifest, labels=labels,
                                      normalize=True)

    logits = np.load(args.logits)
    batch_size = logits[0][0].shape[1]

    results = []


    def result_callback(result):
        results.append(result)


    p = Pool(args.num_workers)

    cand_alphas = np.linspace(args.lm_alpha_from, args.lm_alpha_to, args.lm_num_alphas)
    cand_betas = np.linspace(args.lm_beta_from, args.lm_beta_to, args.lm_num_betas)
    params_grid = []
    for x, alpha in enumerate(cand_alphas):
        for y, beta in enumerate(cand_betas):
            params_grid.append((alpha, beta, x, y))

    futures = []
    for index, (alpha, beta, x, y) in enumerate(params_grid):
        print("Scheduling decode for a={}, b={} ({},{}).".format(alpha, beta, x, y))
        f = p.apply_async(decode_dataset, (logits, test_dataset, batch_size, alpha, beta, x, y, labels),
                          callback=result_callback)
        futures.append(f)
    for f in futures:
        f.wait()
        print("Result calculated:", f.get())
    print("Saving tuning results to: {}".format(args.output_path))
    with open(args.output_path, "w") as fh:
        json.dump(results, fh)
