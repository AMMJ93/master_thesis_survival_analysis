import torchtuples.callbacks as cb
from torchtuples.base import Model
from torchtuples.optim import OptimWrap
from typing import Dict
import torch
from pycox.preprocessing import label_transforms
from pycox.models.interpolation import InterpolateLogisticHazard
import torchtuples as tt
from torchtuples.tupletree import tuplefy
from pycox import models
import pandas as pd


class Ensemble(Model):
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        if callable(self._optimizer):
            self._optimizer = self._optimizer(params=self.net.parameters())
        if not isinstance(self._optimizer, OptimWrap):
            self._optimizer = OptimWrap(self._optimizer)

    def compute_metrics(self, data, metrics=None) -> Dict[str, torch.Tensor]:
        """Function for computing the loss and other metrics.

        Arguments:
            data {tensor or tuple} -- A batch of data. Typically the tuple `(input, target)`.
        Keyword Arguments:
            metrics {dict} -- A dictionary with metrics. If `None` use `self.metrics`. (default: {None})
        """
        return_metrics = {}
        if metrics is None:
            metrics = self.metrics
        if (self.loss is None) and (self.loss in metrics.values()):
            raise RuntimeError(f"Need to set `self.loss`.")

        input, target = data
        input = self._to_device(input)
        target = self._to_device(target)
        #         out1, out2 = self.net(input)
        combined = self.net(input)
        #         print(combined.shape)
        combined = tuplefy(combined)
        #         out1 = tuplefy(out1)
        #         out2 = tuplefy(out2)

        #         resnet_loss = self.loss(*out1, *target)
        #         clinical_loss = self.loss(*out2, *target)
        loss = self.loss(*combined, *target)
        #         loss = resnet_loss + clinical_loss

        return {'loss': loss}

    #         return {'loss': loss,
    #                 'loss_resnet': resnet_loss,
    #                 "loss_clinical": clinical_loss}

    def _setup_metrics(self, metrics=None):
        all_metrics = {'loss': self.loss}
        if metrics is not None:
            if not hasattr(metrics, 'items'):
                if not hasattr(metrics, '__iter__'):
                    metrics = [metrics]
                metrics = {met.__name__: met for met in metrics}
            if 'loss' in metrics:
                raise ValueError("The 'loss' keyword is reserved for the loss function.")
            all_metrics.update(metrics)
        return all_metrics

    def fit_dataloader(self, dataloader, epochs=1, callbacks=None, verbose=True, metrics=None,
                       val_dataloader=None):
        """Fit a dataloader object.
        See 'fit' for tensors and np.arrays.

        Arguments:
            dataloader {dataloader} -- A dataloader that gives (input, target).

        Keyword Arguments:
            epochs {int} -- Number of epochs (default: {1})
            callbacks {list} -- list of callbacks (default: {None})
            verbose {bool} -- Print progress (default: {True})

        Returns:
            TrainingLogger -- Training log
        """
        self._setup_train_info(dataloader)
        self.metrics = self._setup_metrics(metrics)
        self.log.verbose = verbose
        self.val_metrics.dataloader = val_dataloader
        if callbacks is None:
            callbacks = []
        self.callbacks = cb.TrainingCallbackHandler(self.optimizer, self.train_metrics, self.log,
                                                    self.val_metrics, callbacks)
        self.callbacks.give_model(self)

        stop = self.callbacks.on_fit_start()
        for _ in range(epochs):
            if stop: break
            stop = self.callbacks.on_epoch_start()
            if stop: break
            for data in dataloader:
                stop = self.callbacks.on_batch_start()
                if stop: break
                self.optimizer.zero_grad()
                self.batch_metrics = self.compute_metrics(data, self.metrics)
                self.batch_loss = self.batch_metrics['loss']
                self.batch_loss.backward()
                stop = self.callbacks.before_step()
                if stop: break
                self.optimizer.step()
                stop = self.callbacks.on_batch_end()
                if stop: break
            else:
                stop = self.callbacks.on_epoch_end()
        self.callbacks.on_fit_end()
        return self.log
    
    def _predict_func_dl(self, func, dataloader, numpy=False, eval_=True, grads=False, to_cpu=False):
        """Get predictions from `dataloader`.
        `func` can be anything and is not concatenated to `self.net` or `self.net.predict`.
        This is different from `predict` and `predict_net` which both use call `self.net`.
        """
        if hasattr(self, 'fit_info') and (self.make_dataloader is self.make_dataloader_predict):
            data = _get_element_in_dataloader(dataloader)
            if data is not None:
                input = tuplefy(data)
                input_train = self.fit_info['input']
                if input.to_levels() != input_train['levels']:
                    warnings.warn("""The input from the dataloader is different from
                    the 'input' during trainig. Make sure to remove 'target' from dataloader.
                    Can be done with 'torchtuples.data.dataloader_input_only'.""")
                if input.shapes().apply(lambda x: x[1:]) != input_train['shapes']:
                    warnings.warn("""The input from the dataloader is different from
                    the 'input' during trainig. The shapes are different.""")

        if eval_:
            self.net.eval()
        with torch.set_grad_enabled(grads):
            preds = []
            for input in dataloader:
                input = tuplefy(input).to_device(self.device)
                preds_batch = tuplefy(func(input))
                if numpy or to_cpu:
                    preds_batch = preds_batch.to_device('cpu')
                preds.append(preds_batch)
        if eval_:
            self.net.train()
        preds = tuplefy(preds).cat()
        if numpy:
            preds = preds.to_numpy()
        if len(preds) == 1:
            preds = preds[0]
        return preds    


class Base(Ensemble):
    """Base class for survival models.
    Essentially same as torchtuples.Model,
    """
    def predict_surv(self, input, batch_size=8224, numpy=None, eval_=True,
                     to_cpu=False, num_workers=0):
        raise NotImplementedError

    def predict_surv_df(self, input, batch_size=8224, eval_=True, num_workers=0):
        raise NotImplementedError

    def predict_hazard(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False,
                       num_workers=0):
        raise NotImplementedError

    def predict_pmf(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False,
                    num_workers=0):
        raise NotImplementedError


class LogisticAltered(Base):
    label_transform = label_transforms.LabTransDiscreteTime

    def __init__(self, net, optimizer=None, device=None, duration_index=None, loss=None):
        self.duration_index = duration_index
        if loss is None:
            loss = models.loss.NLLLogistiHazardLoss()
        super().__init__(net, loss, optimizer, device)

    @property
    def duration_index(self):
        """
        Array of durations that defines the discrete times. This is used to set the index
        of the DataFrame in `predict_surv_df`.

        Returns:
            np.array -- Duration index.
        """
        return self._duration_index

    @duration_index.setter
    def duration_index(self, val):
        self._duration_index = val

    def predict_surv_df(self, input, batch_size=8224, eval_=True, num_workers=0):
        surv = self.predict_surv(input, batch_size, True, eval_, True, num_workers)
        return pd.DataFrame(surv.transpose(), self.duration_index)

    def predict_surv(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False,
                     num_workers=0, epsilon=1e-7):
        hazard = self.predict_hazard(input, batch_size, False, eval_, to_cpu, num_workers)
        surv = (1 - hazard).add(epsilon).log().cumsum(1).exp()
        return tt.utils.array_or_tensor(surv, numpy, input)

    def predict_hazard(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False,
                       num_workers=0):
        hazard = self.predict(input, batch_size, False, eval_, False, to_cpu, num_workers).sigmoid()
        return tt.utils.array_or_tensor(hazard, numpy, input)

    def interpolate(self, sub=10, scheme='const_pdf', duration_index=None):
        if duration_index is None:
            duration_index = self.duration_index
        return InterpolateLogisticHazard(self, scheme, duration_index, sub)

    
def _get_element_in_dataloader(dataloader):
    dataset = dataloader.dataset
    try:
        return dataset[:2]
    except:
        pass
    try:
        return dataset[[0, 1]]
    except:
        pass
    try:
        return dataloader.collate_fn([dataset[0], dataset[1]])
    except:
        pass
    return None

def wrapfunc(outer, inner):
    """Essentially returns the function `lambda x: outer(inner(x))`
    If `outer` is None, return `inner`.
    """
    if outer is None:
        return inner
    def newfun(*args, **kwargs):
        return outer(inner(*args, **kwargs))
    return newfun