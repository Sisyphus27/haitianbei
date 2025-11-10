"""
Author: zy
Date: 2025-10-22 20:05:06
LastEditTime: 2025-10-22 20:06:31
LastEditors: zy
Description:
FilePath: /haitianbei/data_provider/data_factory.py

"""

from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_KG
import logging as _logging

data_dict = {
    "KG": Dataset_KG,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != "timeF" else 1
    train_only = args.train_only

    if flag == "test":
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    elif flag == "pred":
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        # Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
    )
    if not _logging.getLogger().handlers:
        _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _logging.info(f"{flag} {len(data_set)}")
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader
