import mmcv
import torch
import numbers
import os.path as osp
import torch.nn as nn

from mmcv.runner import load_checkpoint
from mmedit.core import L1Evaluation, psnr, ssim, tensor2img
from mmedit.utils import get_root_logger

from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS


@MODELS.register_module()
class UIEC(BaseModel):


    def __init__(self, generator, loss_percep=None, loss_l1=None, loss_tv=None,
                 train_cfg=None, test_cfg=None, pretrained=None):
        super(UIEC, self).__init__()
        self.with_percep_loss = loss_percep is not None
        self.with_l1_loss = loss_l1 is not None
        self.with_tv_loss = loss_tv is not None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # model
        self.generator = build_backbone(generator)

        # build loss modules
        if self.with_percep_loss:
            self.loss_percep = build_loss(loss_percep)

        if self.with_l1_loss:
            self.loss_l1 = build_loss(loss_l1)

        if self.with_tv_loss:
            self.loss_tv = build_loss(loss_tv)

        self.init_weights(pretrained=pretrained)

    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        """Forward function.

        Args:
            lq (Tensor): Input lq images.
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """

        if test_mode:
            return self.forward_test(lq, gt=gt, **kwargs)

        return self.generator.forward(lq)

    def train_step(self, data_batch, optimizer):
        # data
        lq = data_batch['lq']
        gt = data_batch['gt']

        output = self(**data_batch, test_mode=False)

        # loss
        losses = dict()
        log_vars = dict()

        if self.loss_percep:
            loss_per, _ = self.loss_percep(output, gt)
            losses['loss_percep'] = loss_per

        if self.loss_l1:
            loss2 = self.loss_l1(output, gt)
            losses['loss_l1'] = loss2

        if self.loss_tv:
            loss3 = self.loss_tv(output, gt)
            losses['loss_tv'] = loss3

        loss, log_vars = self.parse_losses(losses)

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        log_vars.pop('loss')  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu()))

        return outputs

    def init_weights(self, pretrained=None, strict=True):
        self.generator.init_weights(pretrained, strict)

    def forward_test(self, lq, gt=None, meta=None,
                     save_image=False, save_path=None, iteration=None):

        # generator
        with torch.no_grad():
            output = self.generator.forward(lq)

        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            if 'gt_path' in meta[0]:
                pred_path = meta[0]['gt_path']
            else:
                pred_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(pred_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(output), save_path)

        return results