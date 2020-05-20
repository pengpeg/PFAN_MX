# -*- coding: utf-8 -*-
# @Time    : 2020/3/9 14:53
# @Author  : Chen
# @File    : main.py
# @Software: PyCharm
import argparse, logging, os, sys
from factory import Factory

def parse_args():
    parser = argparse.ArgumentParser(description='Domain adaptation base on pseudo-labeled target domain samples.')
    parser.add_argument('--record-dir', type=str, default='record', help='The log file directory.')
    parser.add_argument('--data-dir', type=str, default='E:\\datasets\\Office-31', help='The root directory of data.')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoint2', help='The model directory.')
    parser.add_argument('--source', type=str, default='amazon', help='The source dataset.')
    parser.add_argument('--target', type=str, default='webcam', help='The Target dataset.')
    parser.add_argument('--base-net', type=str, default='AlexNet', choices=['AlexNet', 'ResNet50'],
                        help='The backbone network framework.')
    parser.add_argument('--num-classes', type=int, default=31, help='The number of categories.')
    parser.add_argument('--lamba-triplet', type=float, default=0.1, help='')
    parser.add_argument('--lamba-globalda', type=float, default=0.1, help='')
    parser.add_argument('--lamba-classda', type=float, default=0.1, help='')
    parser.add_argument('--u', type=float, default='0.6', help='The threshold parameter.')
    parser.add_argument('--base-lr', type=float, default=0.01, help='The learning rate.')
    parser.add_argument('--alpha', type=float, default=10, help='The learning rate scheduler parameter.')
    parser.add_argument('--beta', type=float, default=0.05, help='The learning rate scheduler parameter.')
    parser.add_argument('--batch-size', type=int, default=100, help='')
    parser.add_argument('--max-iter', type=int, default=200, help='The iteration number of each apa process.')
    parser.add_argument('--max-step', type=int, default=20, help='The maximum step')
    parser.add_argument('--interval', type=int, default=5, help='The frequency of displaying training process.')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.exists(args.record_dir):
        os.mkdir(args.record_dir)
    num_logfile = 0
    logfile_detail = '%s/PFAN_%s_%s_to_%s_%d_detail.log' % (
        args.record_dir, args.base_net, args.source, args.target, num_logfile)
    logfile_brief = '%s/PFAN_%s_%s_to_%s_%d_brief.log' % (
        args.record_dir, args.base_net, args.source, args.target, num_logfile)
    while os.path.exists(logfile_detail):
        num_logfile += 1
        logfile_detail = '%s/PFAN_%s_%s_to_%s_%d_detail.log' % (
        args.record_dir, args.base_net, args.source, args.target, num_logfile)
        logfile_brief = '%s/PFAN_%s_%s_to_%s_%d_brief.log' % (
        args.record_dir, args.base_net, args.source, args.target, num_logfile)
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('record')
    # sh = logging.StreamHandler(sys.stdout)
    fh_detail = logging.FileHandler(filename=logfile_detail)
    fh_brief = logging.FileHandler(filename=logfile_brief)
    # sh.setLevel(logging.DEBUG)
    fh_detail.setLevel(logging.DEBUG)
    fh_brief.setLevel(logging.INFO)
    # logger.addHandler(sh)
    logger.addHandler(fh_detail)
    logger.addHandler(fh_brief)
    logger.debug('log test')

    logger.info(args)

    factory = Factory(logger, args.data_dir, args.checkpoint_dir, base_net=args.base_net, batch_size=args.batch_size,
                      source=args.source, target=args.target, num_classes=args.num_classes, u=args.u,
                      base_lr=args.base_lr, alpha=args.alpha, beta=args.beta, max_iter=args.max_iter,
                      max_step=args.max_step, interval=args.interval)

    factory.transfer_learning()

if __name__ == '__main__':
    main()





