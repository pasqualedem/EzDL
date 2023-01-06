import os
import torch

import numpy as np

from typing import Union, Callable, Mapping, Any, List

from super_gradients.training.utils.callbacks import *
from super_gradients.training.utils.early_stopping import EarlyStop
from super_gradients.training.utils.utils import AverageMeter
from PIL import ImageColor, Image


def callback_factory(name, params, **kwargs):
    params = params or {}
    if name in ['early_stop', 'early_stopping', 'EarlyStop']:
        if "phase" in params:
            params.pop("phase")
        return EarlyStop(Phase.VALIDATION_EPOCH_END, **params)
    if name == "SegmentationVisualizationCallback":
        seg_trainer = kwargs['seg_trainer']
        loader = kwargs['loader']
        dataset = kwargs['dataset']
        params['freq'] = params.get('freq', 1)
        params['phase'] = Phase.VALIDATION_BATCH_END \
            if params['phase'] == 'validation' \
            else Phase.TEST_BATCH_END
        return SegmentationVisualizationCallback(logger=seg_trainer.sg_logger,
                                                 batch_idxs=[0, len(loader) - 1],
                                                 last_img_idx_in_batch=4,
                                                 num_classes=dataset.trainset.CLASS_LABELS,
                                                 undo_preprocessing=dataset.undo_preprocess,
                                                 **params)
    if params.get("phase"):
        params['phase'] = Phase.__dict__[params.get("phase")]
    return globals()[name](**params)


class SegmentationVisualizationCallback(PhaseCallback):
    test_sequence_name = 'test_seg'
    """
    A callback that adds a visualization of a batch of segmentation predictions to context.sg_logger
    Attributes:
        freq: frequency (in epochs) to perform this callback.
        batch_idx: batch index to perform visualization for.
        last_img_idx_in_batch: Last image index to add to log. (default=-1, will take entire batch).
    """

    def __init__(self, logger, phase: Phase, freq: int, num_classes, batch_idxs=None, last_img_idx_in_batch: int = None,
                 undo_preprocessing=None):
        super(SegmentationVisualizationCallback, self).__init__(phase)
        if batch_idxs is None:
            batch_idxs = [0]
        self.freq = freq
        self.num_classes = num_classes
        self.batch_idxs = batch_idxs
        self.last_img_idx_in_batch = last_img_idx_in_batch
        self.undo_preprocesing = undo_preprocessing
        self.prefix = 'train' if phase == Phase.TRAIN_EPOCH_END else 'val' \
            if phase == Phase.VALIDATION_BATCH_END else 'test'
        if phase == Phase.TEST_BATCH_END:
            logger.create_image_mask_sequence(f'{self.prefix}_seg')

    def __call__(self, context: PhaseContext):
        epoch = context.epoch if context.epoch is not None else 0
        if epoch % self.freq == 0 and context.batch_idx in self.batch_idxs:
            preds = context.preds.clone()
            SegmentationVisualization.visualize_batch(logger=context.sg_logger,
                                                      image_tensor=context.inputs,
                                                      pred_mask=preds,
                                                      target_mask=context.target,
                                                      num_classes=self.num_classes,
                                                      batch_name=context.batch_idx,
                                                      undo_preprocessing_func=self.undo_preprocesing,
                                                      prefix=self.prefix,
                                                      names=context.input_name,
                                                      iteration=context.epoch)
            if self.prefix == 'test' and context.batch_idx == self.batch_idxs[-1]:
                context.sg_logger.add_image_mask_sequence(f'{self.prefix}_seg')


class SegmentationVisualization:

    @staticmethod
    def _visualize_image(image_np: np.ndarray, pred_mask: torch.Tensor, target_mask: torch.Tensor, classes):
        """

        :param image_np: numpy image
        :param pred_mask: (C, H, W) tensor of classes in one hot encoding
        :param target_mask: (H, W) tensor of classes
        :param classes:
        :return:
        """
        pred_mask = torch.tensor(pred_mask.copy())
        target_mask = torch.tensor(target_mask.copy())

        pred_mask = pred_mask.argmax(dim=0)

        if image_np.shape[0] < 3:
            image_np = torch.vstack([image_np,
                                     torch.zeros((3 - image_np.shape[0], *image_np.shape[1:]), dtype=torch.uint8)]
                                    )
        image_np = image_np[:3, :, :]  # Take only 3 bands if there are more
        image_np = np.moveaxis(image_np.numpy(), 0, -1)

        return image_np, {
            "predictions": {
                "mask_data": pred_mask.numpy(),
                "class_labels": classes,
            },
            "ground_truth": {
                "mask_data": target_mask.numpy(),
                "class_labels": classes,
            },
        }

    @staticmethod
    def visualize_batch(logger, image_tensor: torch.Tensor, pred_mask: torch.Tensor, target_mask: torch.Tensor,
                        num_classes,
                        batch_name: Union[int, str],
                        undo_preprocessing_func: Callable[[torch.Tensor], np.ndarray] = lambda x: x,
                        image_scale: float = 1.,
                        prefix: str = '',
                        names: List[str] = None,
                        iteration: int = 0):
        """
        A helper function to visualize detections predicted by a network:
        saves images into a given path with a name that is {batch_name}_{imade_idx_in_the_batch}.jpg, one batch per call.
        Colors are generated on the fly: uniformly sampled from color wheel to support all given classes.

        :param iteration:
        :param names:
        :param prefix:
        :param image_tensor:            rgb images, (B, H, W, 3)
        :param batch_name:              id of the current batch to use for image naming
        :param undo_preprocessing_func: a function to convert preprocessed images tensor into a batch of cv2-like images
        :param image_scale:             scale factor for output image
        """
        image_np = undo_preprocessing_func(image_tensor.detach()).type(dtype=torch.uint8).cpu()

        if names is None:
            names = ['_'.join([prefix, 'seg', str(batch_name), str(i)]) if prefix == 'val' else \
                         '_'.join([prefix, 'seg', str(batch_name * image_np.shape[0] + i)]) for i in
                     range(image_np.shape[0])]
        else:
            names = [f"{prefix}_seg_{name}" for name in names]

        for i in range(image_np.shape[0]):
            preds = pred_mask[i].detach().cpu().numpy()
            targets = target_mask[i].detach().cpu().numpy()

            img, mask_dict = SegmentationVisualization._visualize_image(image_np[i], preds, targets, num_classes)
            if prefix == 'val':
                logger.add_mask(names[i], img, mask_dict, global_step=iteration)
            else:
                logger.add_image_mask_to_sequence(SegmentationVisualizationCallback.test_sequence_name,
                                                  names[i], img, mask_dict)


# class MlflowCallback(PhaseCallback):
#     """
#     A callback that logs metrics to MLFlow.
#     """
#
#     def __init__(self, phase: Phase, freq: int,
#                  client: MLRun,
#                  params: Mapping = None
#                  ):
#         """
#         param phase: phase to log metrics for
#         param freq: frequency of logging
#         param client: MLFlow client
#         """
#
#         if phase == Phase.TRAIN_EPOCH_END:
#             self.prefix = 'train_'
#         elif phase == Phase.VALIDATION_EPOCH_END:
#             self.prefix = 'val_'
#         else:
#             raise NotImplementedError('Unrecognized Phase')
#
#         super(MlflowCallback, self).__init__(phase)
#         self.freq = freq
#         self.client = client
#
#         if params:
#             self.client.log_params(params)
#
#     def __call__(self, context: PhaseContext):
#         """
#         Logs metrics to MLFlow.
#             param context: context of the current phase
#         """
#         if context.epoch % self.freq == 0:
#             self.client.log_metrics({self.prefix + k: v for k, v in context.metrics_dict.items()})


class MetricsLogCallback(PhaseCallback):
    """
    A callback that logs metrics to MLFlow.
    """

    def __init__(self, phase: Phase, freq: int
                 ):
        """
        param phase: phase to log metrics for
        param freq: frequency of logging
        param client: MLFlow client
        """

        if phase == Phase.TRAIN_EPOCH_END:
            self.prefix = 'train_'
        elif phase == Phase.VALIDATION_EPOCH_END:
            self.prefix = 'val_'
        else:
            raise NotImplementedError('Unrecognized Phase')

        super(MetricsLogCallback, self).__init__(phase)
        self.freq = freq

    def __call__(self, context: PhaseContext):
        """
        Logs metrics to MLFlow.
            param context: context of the current phase
        """
        if self.phase == Phase.TRAIN_EPOCH_END:
            context.sg_logger.add_scalar('epoch', context.epoch)
        if context.epoch % self.freq == 0:
            context.sg_logger.add_scalars({self.prefix + k: v for k, v in context.metrics_dict.items()})


class AverageMeterCallback(PhaseCallback):
    def __init__(self):
        super(AverageMeterCallback, self).__init__(Phase.TEST_BATCH_END)
        self.meters = {}

    def __call__(self, context: PhaseContext):
        """
        Logs metrics to MLFlow.
            param context: context of the current phase
        """
        context.metrics_compute_fn.update(context.preds, context.target)
        metrics_dict = context.metrics_compute_fn.compute()
        for k, v in metrics_dict.items():
            if not self.meters.get(k):
                self.meters[k] = AverageMeter()
            self.meters[k].update(v, 1)


class SaveSegmentationPredictionsCallback(PhaseCallback):
    def __init__(self, phase, path, num_classes):
        super(SaveSegmentationPredictionsCallback, self).__init__(phase)
        self.path = path
        self.num_classes = num_classes

        os.makedirs(self.path, exist_ok=True)
        colors = ['blue', 'green', 'red']
        self.colors = []
        for color in colors:
            if isinstance(color, str):
                color = ImageColor.getrgb(color)
            self.colors.append(torch.tensor(color, dtype=torch.uint8))

    def __call__(self, context: PhaseContext):
        for prediction, input_name in zip(context.preds, context.input_name):
            path = os.path.join(self.path, input_name)
            prediction = prediction.detach().cpu()
            masks = torch.concat([
                (prediction.argmax(0) == cls).unsqueeze(0)
                for cls in range(self.num_classes)
            ])

            img_to_draw = torch.zeros(*prediction.shape[-2:], 3, dtype=torch.uint8)
            # TODO: There might be a way to vectorize this
            for mask, color in zip(masks, self.colors):
                img_to_draw[mask] = color

            Image.fromarray(img_to_draw.numpy()).save(path)
