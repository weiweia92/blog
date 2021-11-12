# engine/default.py
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
class DefaultTrainer(SimpleTrainer):
    def __init__(self, cfg):
        # Assume these objects must be constructed in this order.
        data_loader = self.build_train_loader(cfg)
        ...    
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        """
        return build_detection_train_loader(cfg)
函数调用关系如下图：

detectron2在构建model,optimizer和data_loader的时候都是在对应的build.py文件里实现的。我们看一下build_detection_train_loader是如何定义的(对应上图中紫色方框内的部分(自下往上的顺序)）：

def build_detection_train_loader(cfg, mapper=None):
    """
    A data loader is created by the following steps:

    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Start workers to work on the dicts. Each worker will:
      * Map each metadata dict into another format to be consumed by the model.
      * Batch them by simply putting dicts into a list.
    The batched ``list[mapped_dict]`` is what this dataloader will return.

    Args:
        cfg (CfgNode): the config
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            By default it will be `DatasetMapper(cfg, True)`.

    Returns:
        a torch DataLoader object
    """
    # 获得dataset_dicts
    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=True,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )

    # 将dataset_dicts转化成torch.utils.data.Dataset
    dataset = DatasetFromList(dataset_dicts, copy=False)

    # 进一步转化成MapDataset，每次读取数据时都会调用mapper来对dict进行解析
    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    # 采样器
    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    if sampler_name == "TrainingSampler":
        sampler = samplers.TrainingSampler(len(dataset))
        ...
    batch_sampler = build_batch_data_sampler(
        sampler, images_per_worker, group_bin_edges, aspect_ratios
    )

    # 数据迭代器 data_loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
    )
    return data_loader
