from ultralytics.yolo.v8.detect import DetectionTrainer


class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        ...
    def get_dataloader(self, dataset_path, batch_size, rank=0, mode='train'):
        return super().get_dataloader(dataset_path, batch_size, rank, mode)


trainer = CustomTrainer(overrides={...})
trainer.train()