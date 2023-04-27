import os

from torch.utils.data import DataLoader, dataloader, distributed
import torch

from custom_dataset import CustomDataset

from ultralytics.yolo.v8.detect import DetectionTrainer
from ultralytics.yolo.utils import LOGGER, RANK, colorstr
from ultralytics.yolo.utils.torch_utils import torch_distributed_zero_first
from ultralytics.yolo.data.build import InfiniteDataLoader, seed_worker
from ultralytics.yolo.data.utils import PIN_MEMORY
from ultralytics.yolo.data.dataloaders.v5loader import create_dataloader

from ultralytics.yolo.utils.torch_utils import de_parallel

def build_dataloader(cfg, batch, img_path, data_info, stride=32, rect=False, rank=-1, mode='train'):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    assert mode in ['train', 'val']
    shuffle = mode == 'train'
    if cfg.rect and shuffle:
        LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = CustomDataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == 'train',  # augmentation
            hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
            rect=cfg.rect or rect,  # rectangular batches
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=int(stride),
            pad=0.0 if mode == 'train' else 0.5,
            prefix=colorstr(f'{mode}: '),
            use_segments=cfg.task == 'segment',
            use_keypoints=cfg.task == 'pose',
            classes=cfg.classes,
            data=data_info)

    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    workers = cfg.workers if mode == 'train' else cfg.workers * 2
    nw = min([os.cpu_count() // max(nd, 1), batch if batch > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if cfg.image_weights or cfg.close_mosaic else InfiniteDataLoader  # allow attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return loader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, 'collate_fn', None),
        worker_init_fn=seed_worker,
        persistent_workers=(nw > 0) and (loader == DataLoader),  # persist workers if using default PyTorch DataLoader
        generator=generator), dataset

class CustomTrainer(DetectionTrainer):
    def get_dataloader(self, dataset_path, batch_size, rank=0, mode='train'):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return create_dataloader(path=dataset_path,
                                 imgsz=self.args.imgsz,
                                 batch_size=batch_size,
                                 stride=gs,
                                 hyp=vars(self.args),
                                 augment=mode == 'train',
                                 cache=self.args.cache,
                                 pad=0 if mode == 'train' else 0.5,
                                 rect=self.args.rect or mode == 'val',
                                 rank=rank,
                                 workers=self.args.workers,
                                 close_mosaic=self.args.close_mosaic != 0,
                                 prefix=colorstr(f'{mode}: '),
                                 shuffle=mode == 'train',
                                 seed=self.args.seed)[0] if self.args.v5loader else \
            build_dataloader(self.args, batch_size, img_path=dataset_path, stride=gs, rank=rank, mode=mode,
                             rect=mode == 'val', data_info=self.data)[0]



