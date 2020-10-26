#-*- coding=utf-8 -*-
import argparse

def parse_option():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--TRAIN_DATA_SPLITS', type=str, default='train')
    parser.add_argument('--DROPOUT_RATIO', type=float, default=0.1)
    parser.add_argument('--CLASSIFIER_CHANNEL', type=str, default=2048)
    parser.add_argument('--IMAGE_CHANNEL', type=int, default=20480)
    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = parse_option()
    print(args)