import numpy as np
import torch

from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_fns, optimizer, config,
                 data_loader, weight_loss_0, weight_loss_1, weight_loss_pos, valid_data_loader=None,
                 test_data_loader=None,
                 lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_fns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader

        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.weight_loss = [weight_loss_0, weight_loss_1, weight_loss_pos]
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_fns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_fns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        print(f"the current training epoch is{epoch}")
        self.model.train()
        device_id = self.device
        # device_id = self.model.device_ids[0]

        self.train_metrics.reset()

        for batch_idx, (path_feature, type_feature, onehot_tensor, lengths, mask, target,
                        stroke_feature, stroke_type_feature, stroke_lengths_feature,
                        stroke_mask_feature, _, Sex_age_tensor) in enumerate(self.data_loader):

            path_feature, type_feature = path_feature.to(device_id), type_feature.to(device_id)
            onehot_tensor, Sex_age_tensor = onehot_tensor.to(device_id), Sex_age_tensor.to(device_id)
            stroke_feature, stroke_type_feature = stroke_feature.to(device_id), stroke_type_feature.to(device_id)
            mask, target, stroke_mask_feature = mask.to(device_id), target.to(device_id), stroke_mask_feature.to(
                device_id)

            gcn = True if batch_idx == 0 else False
            self.optimizer.zero_grad()
            output, _, _, _, _, _, _, _ = self.model(path_feature, type_feature, lengths, mask, stroke_feature,
                                                     stroke_type_feature, stroke_lengths_feature, stroke_mask_feature,
                                                     onehot_tensor, Sex_age_tensor, gcn)

            loss = self.criterion(output=output, target=target, class_weight=self.weight_loss)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            with torch.no_grad():
                y_pred = torch.sigmoid(output)
                y_pred = y_pred.cpu().detach().numpy()
                y_true = target.cpu().detach().numpy()
                for met in self.metric_fns:
                    self.train_metrics.update(met.__name__, met(y_pred, y_true))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()
        log['train'] = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            log['validation'] = {'val_' + k: v for k, v in val_log.items()}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        print(f"valid began, current epoch is{epoch}")
        self.model.eval()
        device_id = self.device
        # device_id = self.model.device_ids[0]

        self.valid_metrics.reset()
        with torch.no_grad():

            for batch_idx, (path_feature, type_feature, onehot_tensor, lengths, mask, target, stroke_feature
                            , stroke_type_feature, stroke_lengths_feature
                            , stroke_mask_feature, _, Sex_age_tensor) in enumerate(self.valid_data_loader):

                path_feature, type_feature = path_feature.to(device_id), type_feature.to(device_id)
                onehot_tensor, Sex_age_tensor = onehot_tensor.to(device_id), Sex_age_tensor.to(device_id)
                stroke_feature, stroke_type_feature = stroke_feature.to(device_id), stroke_type_feature.to(
                    device_id)
                mask, target, stroke_mask_feature = mask.to(device_id), target.to(
                    device_id), stroke_mask_feature.to(device_id)
                output, _, _, _, _, _, _, _ = self.model(path_feature, type_feature, lengths, mask, stroke_feature,
                                                         stroke_type_feature, stroke_lengths_feature,
                                                         stroke_mask_feature, onehot_tensor, Sex_age_tensor, gcn=False)

                loss = self.criterion(output=output, target=target, class_weight=self.weight_loss)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                y_pred = torch.sigmoid(output)
                y_pred = y_pred.cpu().detach().numpy()
                y_true = target.cpu().detach().numpy()
                for met in self.metric_fns:
                    self.valid_metrics.update(met.__name__, met(y_pred, y_true))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def test(self):
        self.model.eval()
        device_id = self.device
        # device_id = self.model.device_ids[0]
        total_loss = 0.0
        total_metrics = torch.zeros(len(self.metric_fns))
        print("begin test now")
        with torch.no_grad():

            for batch_idx, (path_feature, type_feature, onehot_tensor
                            , lengths, mask, target, stroke_feature, stroke_type_feature, stroke_lengths_feature,
                            stroke_mask_feature, _, Sex_age_tensor) in enumerate(self.test_data_loader):

                path_feature, type_feature = path_feature.to(device_id), type_feature.to(device_id)
                onehot_tensor, Sex_age_tensor = onehot_tensor.to(device_id), Sex_age_tensor.to(device_id)
                stroke_feature, stroke_type_feature = stroke_feature.to(device_id), stroke_type_feature.to(
                    device_id)
                mask, target, stroke_mask_feature = mask.to(device_id), target.to(
                    device_id), stroke_mask_feature.to(device_id)
                output, _, _, _, _, _, _, _ = self.model(path_feature, type_feature, lengths, mask, stroke_feature,
                                                         stroke_type_feature, stroke_lengths_feature,
                                                         stroke_mask_feature, onehot_tensor, Sex_age_tensor,
                                                         gcn=False)

                loss = self.criterion(output=output, target=target, class_weight=self.weight_loss)

                batch_size = path_feature.shape[0]
                total_loss += loss.item() * batch_size
                y_pred = torch.sigmoid(output)
                y_pred = y_pred.cpu().detach().numpy()
                y_true = target.cpu().detach().numpy()
                for i, metric in enumerate(self.metric_fns):
                    total_metrics[i] += metric(y_pred, y_true) * batch_size

        test_output = {'n_samples': len(self.test_data_loader.sampler),
                       'total_loss': total_loss,
                       'total_metrics': total_metrics}
        return test_output

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
