import shutil
import json
import os
import time

import torch
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss
from tqdm import tqdm

from data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler
from decoder import *
from model import DeepSpeech, supported_rnns
from utils import *

options = Options(description='Pytorch DeepSpeech2 Training')
options.set_defaults(epochs=100, batch_size=5, lr=3e-4, momentum=0.9, 
                     max_norm=400, lr_anneal=1.1, rnn_type='lstm',
                     hidden_layers=5, hidden_size=1024, decoder='greedy')


def main():
    global args, train_logger, test_logger
    args = options.parse_args()
    os.makedirs(args.log_dir)
    test_logger = Logger(os.path.join(args.log_dir, 'test.log'))
    with open(os.path.join(args.log_dir, 'config.log'), 'w') as f:
        f.write(args.config_str)
    if not args.evaluate:
        os.makedirs(args.checkpoint_dir)
        train_logger = Logger(os.path.join(args.log_dir, 'train.log'))
    loss_results, cer_results = torch.FloatTensor(args.epochs), torch.FloatTensor(args.epochs)

    if args.visdom:
        from visdom import Visdom
        viz = Visdom()
        opts = dict(title=args.experiment_id, ylabel='', xlabel='Epoch', 
                    legend=['Loss', 'CER'])
        viz_windows = None
        epochs = torch.arange(0, args.epochs)

    if args.resume:
        print('Loading checkpoint model %s' % args.resume)
        checkpoint = torch.load(args.resume)
        model = DeepSpeech.load_model_checkpoint(checkpoint)
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(args.nGPU)]).cuda()
        labels = DeepSpeech.get_labels(model)
        audio_conf = DeepSpeech.get_audio_conf(model)
        parameters = model.parameters()
        optimizer = torch.optim.SGD(parameters, lr=args.lr,
                                    momentum=args.momentum, nesterov=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = int(checkpoint.get('epoch', 0))  # Index start at 0 for training
        loss_results, cer_results = checkpoint['loss_results'], checkpoint['cer_results']
        if args.epochs > loss_results.numel():
            loss_results.resize_(args.epochs)
            cer_results.resize_(args.epochs)
            loss_results[start_epoch:].zero_()
            cer_results[start_epoch:].zero_()
        # Add previous scores to visdom graph
        if args.visdom and loss_results is not None:
            x_axis = epochs[0:start_epoch]
            y_axis = torch.stack(
                (loss_results[0:start_epoch], cer_results[0:start_epoch]),
                dim=1)
            viz_window = viz.line(
                X=x_axis,
                Y=y_axis,
                opts=opts,
            )
    else:
        start_epoch = args.start_epoch
        with open(args.labels_path) as label_file:
            labels = str(''.join(json.load(label_file)))

        audio_conf = dict(sample_rate=args.sample_rate,
                          window_size=args.window_size,
                          window_stride=args.window_stride,
                          window=args.window,
                          noise_dir=args.noise_dir,
                          noise_prob=args.noise_prob,
                          noise_levels=(args.noise_min, args.noise_max))
        model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                           nb_layers=args.hidden_layers,
                           labels=labels,
                           rnn_type=supported_rnns[args.rnn_type],
                           audio_conf=audio_conf,
                           bidirectional=not args.look_ahead)
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(args.nGPU)]).cuda()
        parameters = model.parameters()
        optimizer = torch.optim.SGD(parameters, lr=args.lr,
                                    momentum=args.momentum, nesterov=True)

    # define loss function (criterion) and decoder
    best_cer = None
    criterion = CTCLoss()
    decoder = GreedyDecoder(labels)

    # define dataloader
    if not args.evaluate:
        train_dataset = SpectrogramDataset(audio_conf=audio_conf,
                                           manifest_filepath=args.train_manifest,
                                           labels=labels, normalize=True, augment=args.augment)
        train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)
        train_loader = AudioDataLoader(train_dataset,
                                       num_workers=args.num_workers, batch_sampler=train_sampler)
        if not args.in_order and start_epoch != 0:
            print("Shuffling batches for the following epochs")
            train_sampler.shuffle()
    val_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest,
                                      labels=labels, normalize=True, augment=False)
    val_loader = AudioDataLoader(val_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    print(model)
    print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    if args.evaluate:
        validate(val_loader, model, decoder, 0)
        return

    for epoch in range(start_epoch, args.epochs):
        avg_loss = train(train_loader, train_sampler, model, criterion, optimizer, epoch)
        cer = validate(val_loader, model, decoder, epoch)

        loss_results[epoch] = avg_loss
        cer_results[epoch] = cer

        adjust_learning_rate(optimizer)

        is_best = False
        if best_cer is None or best_cer > cer:
            print('Found better validated model')
            best_cer = cer
            is_best = True
        save_checkpoint(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch,
                                             loss_results=loss_results, cer_results=cer_results),
                        is_best, epoch)

        if not args.in_order:
            print("Shuffling batches...")
            train_sampler.shuffle()

        if args.visdom:
            x_axis = epochs[0:epoch + 1]
            y_axis = torch.stack((loss_results[0:epoch + 1], cer_results[0:epoch + 1]), dim=1)
            if viz_window is None:
                viz_window = viz.line(
                    X=x_axis,
                    Y=y_axis,
                    opts=opts,
                )
            else:
                viz.line(
                    X=x_axis.unsqueeze(0).expand(y_axis.size(1), x_axis.size(0)).transpose(0, 1),  # Visdom fix
                    Y=y_axis,
                    win=viz_window,
                    update='replace',
                )


def train(train_loader, train_sampler, model, criterion, optimizer, epoch):
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
    avg_loss = 0
    model.train()
    end = time.time()
    for i, (data) in enumerate(train_loader):
        if i == len(train_sampler):
            break
        inputs, targets, input_percentages, target_sizes = data
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = Variable(inputs, requires_grad=False).cuda()
        target_sizes = Variable(target_sizes, requires_grad=False)
        targets = Variable(targets, requires_grad=False)

        out = model(inputs)
        out = out.transpose(0, 1)  # TxNxH
        seq_length = out.size(0)
        sizes = Variable(input_percentages.mul_(int(seq_length)).int(), requires_grad=False)

        loss = criterion(out, targets, sizes, target_sizes)
        loss = loss / inputs.size(0)  # average the loss by minibatch

        loss_sum = loss.data.sum()
        inf = float("inf")
        if loss_sum == inf or loss_sum == -inf:
            print("WARNING: received an inf loss, setting loss value to 0")
            loss_value = 0
        else:
            loss_value = loss.data[0]
        losses.update(loss_value, inputs.size(0))
        avg_loss += loss_value

        # compute gradient
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.max_norm)
        # SGD step
        optimizer.step()

        torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i, len(train_sampler), batch_time=batch_time,
                    data_time=data_time, loss=losses))
        del loss
        del out
    avg_loss /= len(train_sampler)

    print('Epoch: [{0}]\t Average Loss {loss:.3f}\t'
          'Time {batch_time.avg:.3f}\tData {data_time.avg:.3f}\t'.format(
            epoch, loss=avg_loss, batch_time=batch_time,
            data_time=data_time))
    train_logger.write('{0}\t{1}\t{2}\t{3}\n'.format(
        0, avg_loss, batch_time.avg, data_time.avg))
    return avg_loss


def validate(val_loader, model, decoder, epoch):
    total_cer = 0
    model.eval()
    for i, (data) in tqdm(enumerate(val_loader), total=len(val_loader)):
        inputs, targets, input_percentages, target_sizes = data

        inputs = Variable(inputs, volatile=True).cuda()
        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        out = model(inputs)
        out = out.transpose(0, 1)  # TxNxH
        seq_length = out.size(0)
        sizes = input_percentages.mul_(int(seq_length)).int()

        decoded_output, _ = decoder.decode(out.data, sizes)
        target_strings = decoder.convert_to_strings(split_targets)
        cer = 0
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            cer += decoder.cer(transcript, reference) / float(len(reference))
        total_cer += cer

        torch.cuda.synchronize()
        del out
    cer = total_cer / len(val_loader.dataset) * 100
    print('Epoch: [{0}]\tAverage CER {cer:.3f}\t'.format(epoch, cer=cer))
    test_logger.write('{0}\n'.format(cer))
    return cer


def adjust_learning_rate(optimizer):
    """Sets the learning rate to the initial LR decayed"""
    lr = optimizer.param_groups[0]['lr']
    lr = lr / args.lr_anneal
    print('Set learning rate to {}'.format(lr))
    for group in optimizer.param_groups:
        group['lr'] = lr


def save_checkpoint(state, is_best, epoch):
    filename = os.path.join(args.checkpoint_dir, 'checkpoint_%d.pth.tar' % epoch)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.checkpoint_dir, 'model_best.pth.tar'))
    if epoch >= 1:
        filename = os.path.join(args.checkpoint_dir, 'checkpoint_%d.pth.tar' % (epoch-1))
        if os.path.exists(filename):
            os.remove(filename)


if __name__ == '__main__':
    main()