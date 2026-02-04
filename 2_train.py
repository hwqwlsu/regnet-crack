# -*- coding:utf-8 -*-
from ultralytics import YOLO
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description='Train or validate YOLO model.')
# 想用哪个loss直接把对应的变量改为True即可，全部填写False这里默认使用ciouloss进行训练
parser.add_argument('--impiou', type=str, default=True, help='Whether to use wiou or not (True/False)')
# train用于训练原始模型  val 用于得到精度指标
parser.add_argument('--mode', type=str, default='train', help='Mode of operation.')

# 训练时此处填写yaml文件路径 推理时此处填写网络结构.yaml文件路径
parser.add_argument('--weights', type=str, default=r'yaml/yolo11n_repghost_imcbam.yaml', help='Path to model file.')
# 数据集存放路径
parser.add_argument('--data', type=str, default='data.yaml', help='Path to data file.')

parser.add_argument('--epoch', type=int, default=150, help='Number of epochs.')
parser.add_argument('--batch', type=int, default=64, help='Batch size.')
parser.add_argument('--workers', type=int, default=8, help='Number of workers.')
parser.add_argument('--device', type=str, default='0', help='Device to use.')
parser.add_argument('--name', type=str, default='', help='Name data file.')
parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer.')
parser.add_argument('--seed', type=str, default='0', help='seed.')
parser.add_argument('--single_cls', type=bool, default=False, help='single_cls.')

args = parser.parse_args()

def train(model, data, epoch, batch, workers, device, name):
    model.train(data=data, epochs=epoch, batch=batch, workers=workers, device=device, name=name,
                optimizer=args.optimizer, seed=int(args.seed), single_cls=args.single_cls, piou=args.impiou)


def validate(model, data, batch, workers, device, name):
    model.val(data=data, batch=batch, workers=workers, device=device, name=name)


def main():
    model = YOLO(args.weights)
    if args.mode == 'train':
        train(model, args.data, args.epoch, args.batch, args.workers, args.device, args.name)
    else:
        validate(model, args.data, args.batch, args.workers, args.device, args.name)


if __name__ == '__main__':
    main()
