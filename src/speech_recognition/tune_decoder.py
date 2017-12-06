import shutil
import json
import os
import time
import numpy as np
from multiprocessing import Pool

import torch
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss
from tqdm import tqdm

from data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler
from decoder import *
from model import DeepSpeech, supported_rnns
from utils import *


options = Options(description='Pytorch DeepSpeech2 Tune Decoder')
options.add_argument('--logits', type=str, 
                     help='Path to logits from test.py')
options.add_argument('--lm-alpha-from', default=0.4, type=float,
                     help='Language model weight start tuning')
options.add_argument('--lm-alpha-to', default=1.3, type=float,
                     help='Language model weight end tuning')
options.add_argument('--lm-num-alphas', default=45, type=int,
                     help='Number of alpha candidates for tuning')
options.add_argument('--lm-bw-from', default=10, type=int,
                     help='Beam width start tuning')
options.add_argument('--lm-bw-to', default=100, type=int,
                     help='Beam width end tuning')
options.add_argument('--lm-num-bws', default=18, type=int,
                     help='Number of beam width candidates for tuning')
options.set_defaults(batch_size=5)


def main():
    global args, test_logger
    args = options.parse_args()
    os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir, 'config.log'), 'w') as f:
        f.write(args.config_str)

    if args.resume:
        print('Loading checkpoint model %s' % args.resume)
        checkpoint = torch.load(args.resume)
        model = DeepSpeech.load_model_checkpoint(checkpoint)
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(args.nGPU)]).cuda()
        labels = DeepSpeech.get_labels(model)
        audio_conf = DeepSpeech.get_audio_conf(model)
        audio_conf['noise_dir'] = None
    else:
        return

    # define dataloader
    val_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest,
                                      labels=labels, normalize=True, augment=False)
    val_loader = AudioDataLoader(val_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    print(model)
    print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    if args.logits is None:
        out_list, sizes_list = validate(val_loader, model)
        torch.save({'out': out_list, 'sizes': sizes_list},
                   os.path.join(args.log_dir, 'test.logits'))
    else:
        checkpoint = torch.load(args.logits)
        out_list, sizes_list = checkpoint['out'], checkpoint['sizes']

    results = []


    def result_callback(result):
        results.append(result)


    p = Pool(args.num_workers)

    cand_bws = np.linspace(args.lm_bw_from, args.lm_bw_to, args.lm_num_bws)
    cand_alphas = np.linspace(args.lm_alpha_from, args.lm_alpha_to, args.lm_num_alphas)
    params_grid = []
    for x, bw in enumerate(cand_bws):
        for y, alpha in enumerate(cand_alphas):
            params_grid.append((bw, alpha, x, y))

    futures = []
    for _, (bw, alpha, x, y) in enumerate(params_grid):
        print("Scheduling decode for bw={}, alpha={} ({},{}).".format(bw, alpha, x, y))
        f = p.apply_async(decode_dataset, (val_loader, out_list, sizes_list, bw, alpha, x, y, labels),
                          callback=result_callback)
        futures.append(f)
    for f in futures:
        f.wait()
        print("Result calculated:", f.get())

    output_path = os.path.join(args.log_dir, 'test.log')
    print("Saving tuning results to: {}".format(output_path))
    with open(output_path, "w") as fh:
        json.dump(results, fh)


def validate(val_loader, model):
    print('DeepSpeech Forward')
    out_list, sizes_list = [], []
    model.eval()
    for i, (data) in tqdm(enumerate(val_loader), total=len(val_loader)):
        inputs, targets, input_percentages, target_sizes = data
        inputs = Variable(inputs, volatile=True).cuda()
        out = model(inputs)
        out = out.transpose(0, 1)  # TxNxH
        seq_length = out.size(0)
        sizes = input_percentages.mul_(int(seq_length)).int()
        out_list.append(out.data.cpu())
        sizes_list.append(sizes.cpu())
        torch.cuda.synchronize()
        del out, sizes
    return out_list, sizes_list


def decode_dataset(val_loader, out_list, sizes_list, beam_width, lm_alpha, mesh_x, mesh_y, labels):
    total_cer = 0
    decoder = BeamDecoder(labels, lm_path=args.lm_path, alpha=lm_alpha,
                          beam_width=beam_width)
    model.eval()
    for i, (data) in enumerate(val_loader):
        inputs, targets, input_percentages, target_sizes = data

        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        out = out_list[i]
        sizes = sizes_list[i]

        decoded_output, _ = decoder.decode(out, sizes)
        target_strings = decoder.convert_to_strings(split_targets)
        cer = 0
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            cer += decoder.cer(transcript, reference) / float(len(reference))
        total_cer += cer
    cer = total_cer / len(val_loader.dataset) * 100
    print('[{0}, {1}]\t[bw={2}, alpha={3}]\tAverage CER {cer:.3f}\t'.format(
           mesh_x, mesh_y, beam_width, lm_alpha, cer=cer))
    return [mesh_x, mesh_y, beam_width, lm_alpha, cer]


if __name__ == '__main__':
    main()