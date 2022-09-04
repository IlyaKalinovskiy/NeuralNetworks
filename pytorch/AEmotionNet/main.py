import sys

sys.path.append("../../")
import os
import pprint
import logging
import argparse

from shutil import copyfile

import yaml
import numpy as np
import torch
import torch.optim as optim

from accuracy import Accuracy, Accuracy3D
from models.stp_net import STPNet

from pytorch.common.audio_processor import LibrosaAudioProcessor
from pytorch.common.batcher.batch_processor import SimpleBatchProcessor
from pytorch.common.datasets_parsers.av_parser import AVDBParser
from pytorch.common.group_random_sampler import GroupRandomSampler
from pytorch.common.images_batcher import ImagesBatcher
from pytorch.common.logging.logging_control import create_logger
from pytorch.common.losses import *
from pytorch.common.lr_schedulers import *
from pytorch.common.net_trainer import NetTrainer


def get_net_params(net, lr, weight_decay):
    # large lr for last fc parameters
    params = []
    for name, value in net.named_parameters():
        if not value.requires_grad:
            continue
        if "bias" in name:
            params += [{"name": name, "params": value, "lr": lr, "weight_decay": 0}]
        else:
            params += [
                {"name": name, "params": value, "lr": lr, "weight_decay": weight_decay}
            ]
    return params


def train():
    arguments_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arguments_parser.add_argument(
        "--config", help="yml config path", type=str, required=True
    )
    arguments_parser.add_argument(
        "--host", help="name of the host server", type=str, required=True
    )

    args = arguments_parser.parse_args()

    # get host name for brevity
    host = args.host

    # read configuration
    with open(args.config) as yml_file:
        cfg = yaml.load(yml_file)
        params = {
            key: cfg[key]
            for key in cfg
            if not key in ["dataset", "ini_net", "logging", "test"]
        }
        dataset = {key: cfg["dataset"][key][host] for key in cfg["dataset"]}
        log = {key: cfg["logging"][key][host] for key in cfg["logging"]}
        ini_net = cfg["ini_net"][host]

    # init logging
    experiment_name = params["train"]["experiment_name"] + str(
        params["train"]["cuda_device"]
    )
    create_logger(
        log["log_dir"],
        experiment_name + ".log",
        console_level=logging.CRITICAL,
        file_level=logging.NOTSET,
    )

    # log configuration
    pp = pprint.PrettyPrinter(indent=4)
    logging.info("Configuration is:\n%s" % pp.pformat(cfg))
    experiment_path = os.path.join(log["log_dir"], params["train"]["experiment_name"])
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    copyfile(args.config, os.path.join(experiment_path, "config.yml"))

    device = torch.device(
        "cuda:" + str(params["train"]["cuda_device"])
        if torch.cuda.is_available()
        else "cpu"
    )
    torch.manual_seed(params["seed"])
    torch.cuda.manual_seed_all(params["seed"])

    train_dataset_parser = AVDBParser(
        dataset_root=dataset["train"]["data_root"],
        file_list=dataset["train"]["file_list"],
        max_num_clips=params["parser"]["max_num_clips"],
        max_num_samples=params["parser"]["max_num_samples"],
        ungroup=False,
        normalize="AFEW-VA" in dataset["train"]["data_root"],
    )
    valid_dataset_parser = AVDBParser(
        dataset_root=dataset["valid"]["data_root"],
        file_list=dataset["valid"]["file_list"],
        max_num_clips=params["parser"]["max_num_clips"],
        max_num_samples=params["parser"]["max_num_samples"],
        ungroup=False,
        normalize="AFEW-VA" in dataset["train"]["data_root"],
    )

    softmax_size = (
        params["net"]["softmax_size"]
        if params["net"]["softmax_size"] > 0
        else train_dataset_parser.get_class_num()
    )
    print("\rsoftmax_size = %d" % softmax_size)

    # create train data sampler
    train_data_sampler = GroupRandomSampler(
        data=train_dataset_parser.get_data(),
        num_sample_per_classes=1,
        samples_is_randomize=True,
        step_size_for_samples=1,
    )
    valid_data_sampler = GroupRandomSampler(
        data=valid_dataset_parser.get_data(),
        num_sample_per_classes=1,
        samples_is_randomize=True,
        step_size_for_samples=1,
        is_shuffle=False,
    )

    # create train audio processor
    audio_processor = LibrosaAudioProcessor(
        type=params["preproc"]["audio"]["type"],
        sampling_rate=params["preproc"]["audio"]["sampling_rate"],
        max_wave_value=params["preproc"]["audio"]["max_wave_value"],
        sfft=params["preproc"]["audio"]["sfft"],
        mel=params["preproc"]["audio"]["mel"],
    )

    # create train image batcher
    train_images_batcher = ImagesBatcher(
        queue_size=params["train_batcher"]["queue_size"],
        batch_size=params["train_batcher"]["batch"],
        data_sampler=train_data_sampler,
        audio_processor=audio_processor,
        single_epoch=False,
        cache_data=False,
        disk_reader_process_num=params["train_batcher"]["disk_reader_process_num"],
    )
    valid_images_batcher = ImagesBatcher(
        queue_size=params["valid_batcher"]["queue_size"],
        batch_size=params["valid_batcher"]["batch"],
        data_sampler=valid_data_sampler,
        audio_processor=audio_processor,
        single_epoch=True,
        cache_data=False,
        disk_reader_process_num=params["valid_batcher"]["disk_reader_process_num"],
    )

    # create batch processor
    batch_processor = SimpleBatchProcessor(
        use_pin_memory=params["batch_proc"]["use_pin_memory"],
        cuda_id=params["train"]["cuda_device"],
        use_async=params["batch_proc"]["use_async"],
    )

    net_type = params["net"]["type"]
    if net_type == "STPNet":
        net = STPNet(softmax_size, depth=params["net"]["depth"])
    else:
        print("Type net is not supported!")
        exit(0)

    # net.apply(weights_init)
    loss = TotalLoss(params["losses"], 1, params["train"]["cuda_device"])

    net.to(device)
    loss.to(device)

    if params["net"]["fine_tune"]:
        net.load_state_dict(torch.load(ini_net))

    lr = params["opt"]["lr"]
    momentum = params["opt"]["momentum"]
    weight_decay = params["opt"]["weight_decay"]

    opt_type = params["opt"]["type"]
    if opt_type == "SGD":
        optimizer = optim.SGD(
            [{"params": net.parameters()}, {"params": loss.parameters()}],
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )  # , caffe_like=True)
    if opt_type == "ASGD":
        optimizer = optim.ASGD(
            [{"params": net.parameters()}, {"params": loss.parameters()}],
            lr=lr,
            weight_decay=weight_decay,
        )  # , caffe_like=True)
    if opt_type == "Adagrad":
        optimizer = optim.Adagrad(
            [{"params": net.parameters()}, {"params": loss.parameters()}],
            lr=lr,
            weight_decay=weight_decay,
        )
    if opt_type == "Adadelta":
        optimizer = optim.Adadelta(
            [{"params": net.parameters()}, {"params": loss.parameters()}],
            lr=lr,
            weight_decay=weight_decay,
        )
    if opt_type == "Adam":
        optimizer = optim.Adam(
            [{"params": net.parameters()}, {"params": loss.parameters()}],
            lr=lr,
            weight_decay=weight_decay,
        )
    if opt_type == "RMSprop":
        optimizer = optim.RMSprop(
            [{"params": net.parameters()}, {"params": loss.parameters()}],
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    if opt_type == "LBFGS":
        optimizer = optim.LBFGS(
            net.parameters(),
            lr=lr,
            max_iter=5,
            max_eval=None,
            tolerance_grad=1e-05,
            tolerance_change=1e-09,
            history_size=1,
            line_search_fn=None,
        )

    # if params['net']['init'] == 'fine_tune':
    #    optimizer = cp['optimizer']

    lr_scheduler_type = params["lr_scheduler"]["type"]
    gamma = float(params["lr_scheduler"]["gamma"])
    if lr_scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma)
    if lr_scheduler_type == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[15000, 30000, 60000], gamma=gamma
        )
    if lr_scheduler_type == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    if lr_scheduler_type == "SGDR":
        scheduler = SGDR_scheduler(
            optimizer,
            lr_start=lr,
            lr_end=gamma * lr,
            lr_period=params["train"]["epoch_size"] // params["train"]["step_size"],
            scale_lr=params["lr_scheduler"]["scale_lr"],
            scale_lr_fc=params["lr_scheduler"]["scale_lr_fc"],
        )

    net_trainer = NetTrainer(
        logs_dir=log["tb_log_dir"],
        cuda_id=params["train"]["cuda_device"],
        experiment_name=experiment_name,
        snapshot_dir=log["snapshot_dir"],
        config_data=cfg,
    )

    accuracy_fn = Accuracy3D(
        valid_dataset_parser.get_data(),
    )

    # lets try to train
    torch.set_num_threads(2)
    net_trainer.train(
        model=net,
        optimizer=optimizer,
        scheduler=scheduler,
        train_data_batcher=train_images_batcher,
        val_data_batcher=valid_images_batcher,
        val_iter=params["train"]["validate_iter"],
        batch_processor=batch_processor,
        loss_function=loss,
        max_iter=params["train"]["max_iter"],
        step_size=params["train"]["step_size"],
        snapshot_iter=params["train"]["snapshot_iter"],
        step_print=params["train"]["step_print"],
        accuracy_function=accuracy_fn,
    )

    # finalize data loader queues
    train_images_batcher.finish()
    valid_images_batcher.finish()


def test():
    arguments_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arguments_parser.add_argument(
        "--config", help="yml config path", type=str, required=True
    )
    arguments_parser.add_argument(
        "--host", help="name of the host server", type=str, required=True
    )

    args = arguments_parser.parse_args()
    host = args.host

    with open(args.config) as yml_file:
        cfg = yaml.load(yml_file)
        train_params = {
            key: cfg[key]
            for key in cfg
            if not key in ["dataset", "ini_net", "logging", "test"]
        }
        test_params = cfg["test"]
        dataset = {
            key: test_params["dataset"][key][host] for key in test_params["dataset"]
        }

    device = torch.device(
        "cuda:" + str(test_params["cuda_device"]) if torch.cuda.is_available() else "cpu"
    )
    torch.manual_seed(train_params["seed"])
    torch.cuda.manual_seed_all(train_params["seed"])
    cp = torch.load(test_params["file_model"])

    test_dataset_parser = AVDBParser(
        dataset_root=dataset["data_root"],
        file_list=dataset["test_file_list"],
        ungroup=train_params["preproc"]["data_frame"]["depth"] == 1,
    )

    test_data_sampler = GroupRandomSampler(
        data=test_dataset_parser.get_data(),
        num_sample_per_classes=train_params["preproc"]["data_frame"]["depth"],
        samples_is_randomize=train_params["sampler"]["samples_is_randomize"],
        step_size_for_samples=train_params["sampler"]["step_size_for_samples"],
        is_shuffle=False,
    )

    test_audio_processor = LibrosaAudioProcessor(
        type=train_params["preproc"]["audio"]["type"],
        sampling_rate=train_params["preproc"]["audio"]["sampling_rate"],
        max_wave_value=train_params["preproc"]["audio"]["max_wave_value"],
        sfft=train_params["preproc"]["audio"]["sfft"],
        mel=train_params["preproc"]["audio"]["mel"],
    )

    test_images_batcher = ImagesBatcher(
        queue_size=train_params["valid_batcher"]["queue_size"],
        batch_size=train_params["valid_batcher"]["batch"],
        data_sampler=test_data_sampler,
        audio_processor=test_audio_processor,
        single_epoch=True,
        cache_data=False,
        disk_reader_process_num=train_params["valid_batcher"]["disk_reader_process_num"],
    )

    batch_processor = SimpleBatchProcessor(
        use_pin_memory=train_params["batch_proc"]["use_pin_memory"],
        cuda_id=train_params["train"]["cuda_device"],
        use_async=train_params["batch_proc"]["use_async"],
    )

    model = cp["model"]
    model.to(device)
    model.eval()

    test_images_batcher.start()
    predict = np.zeros((0, 2), dtype=np.float32)
    while True:
        batch = test_images_batcher.next_batch()
        if batch is None:
            break

        data, _, _ = batch_processor.pre_processing(batch)
        logits = model(data)
        predict = np.concatenate((predict, logits.cpu().data.numpy()), axis=0)

    accuracy_fn = Accuracy(test_dataset_parser.get_data())
    accuracy_fn.by_frames(predict)
    accuracy_fn.by_clips(predict)


if __name__ == "__main__":
    train()
    # test()
