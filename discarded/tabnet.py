

# Extending tabnet (https://github.com/dreamquark-ai/tabnet) v3.1.1 to
# - support data points available in loss functions
# - include a JaneStreetTabNetSampler that fetch trade opportunities in chronological order
# - include a JaneStreetTabNetDataset that updates two rolling features that depend on the output of the model (take o don't take the trade):
#       + taken weighted qty / available weighted qty (since the beginning of the day)
#       + taken weighted buy side qty / taken weighted qty (since the beginning of the day)

# The idea was to use this model with the utility function as loss function
# The idea was discarded: too slow for the time constrain in the competition and besides tabnet works better with categorical variables


from pytorch_tabnet.tab_model import TabNetClassifier, TabModel
from pytorch_tabnet.metrics import Metric, check_metrics
  
import torch
import numpy as np

from scipy.special import softmax
from pytorch_tabnet.utils import PredictDataset, filter_weights, validate_eval_set, create_sampler
from pytorch_tabnet.abstract_model import TabModel
from pytorch_tabnet.multiclass_utils import infer_output_dim, check_output_dim
from pytorch_tabnet.callbacks import Callback
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

#from pytorch_tabnet.metrics import Metric
#from pytorch_tabnet.tab_model import TabNetRegressor


from sklearn.utils import check_array
from scipy.sparse import csc_matrix

from torch.utils.data import Dataset

from typing import List

from dataclasses import dataclass

import zipfile
import json
import io
import os.path

from torch.utils.data import Sampler

from training import utility_score, get_predicted_actions, loss_fn

class u_metric(Metric):
    
    def __init__(self):
        self._name = "u"
        self._maximize = True

    def __call__(self, y_true, y_score, context):
        
        y_pred = torch.tensor(y_score, dtype=torch.float32, requires_grad=False, device=torch.device('cuda'))
        c = torch.tensor(context, dtype=torch.float32, requires_grad=False, device=torch.device('cuda'))

        return utility_score(c.to(device='cuda'), get_predicted_actions(y_pred), torch.device('cuda'), metric='u', clipbelow=False).cpu().item()

class t_metric(Metric):
    
    def __init__(self):
        self._name = "t"
        self._maximize = True

    def __call__(self, y_true, y_score, context):
        
        y_pred = torch.tensor(y_score, dtype=torch.float32, requires_grad=False, device=torch.device('cuda'))
        c = torch.tensor(context, dtype=torch.float32, requires_grad=False, device=torch.device('cuda'))

        return utility_score(c.to(device='cuda'), get_predicted_actions(y_pred), torch.device('cuda'), metric='t').cpu().item()

class p_metric(Metric):
    
    def __init__(self):
        self._name = "p"
        self._maximize = True

    def __call__(self, y_true, y_score, context):
        
        y_pred = torch.tensor(y_score, dtype=torch.float32, requires_grad=False, device=torch.device('cuda'))
        c = torch.tensor(context, dtype=torch.float32, requires_grad=False, device=torch.device('cuda'))

        return utility_score(c.to(device='cuda'), get_predicted_actions(y_pred), torch.device('cuda'), metric='p').cpu().item()

@dataclass
class MetricContainer:

    metric_names: List[str]
    prefix: str = ""

    def __post_init__(self):
        self.metrics = Metric.get_metrics_by_names(self.metric_names)
        self.names = [self.prefix + name for name in self.metric_names]

    def __call__(self, y_true, y_pred, context):

        logs = {}
        for metric in self.metrics:
            if isinstance(y_pred, list):
                res = np.mean(
                    [metric(y_true[:, i], y_pred[i], context) for i in range(len(y_pred))]
                )
            else:
                res = metric(y_true, y_pred, context)
            logs[self.prefix + metric._name] = res
        return logs

class JaneStreetTabNetDataset(Dataset):

    def __init__(self, x, y, c):
        self.x = x
        self.y = y
        self.c = c

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x, y, c = self.x[index], self.y[index], self.c[index]
        return x, y, c

    def update_rolling_features(self, scores):

        l = scores.shape[0]
        #self.y[:l] = scores[:,1]
        self.y[:l] = (scores[:,1] > scores[:,0])*1

        for day in np.unique(self.c[:l,0]):

            day_index = self.c[:,0] == day
            x_day = self.x[day_index]
            y_day = self.y[day_index]
            c_day = self.c[day_index]

            BUY = x_day[:,0] == 1
            SEL = x_day[:,0] == 0

            #TODO: correr hacia atras
            alloc = np.cumsum(c_day[:,1])
            alloc_taken_buy = np.cumsum(BUY * c_day[:,1] * y_day)
            alloc_taken_sel = np.cumsum(SEL * c_day[:,1] * y_day)

            self.x[day_index,-2] = (alloc_taken_buy + alloc_taken_sel) / alloc
            self.x[day_index,-1] = alloc_taken_buy / (alloc_taken_buy + alloc_taken_sel)

        # at the beginning of the day there is no variance
        self.x[:,-2:][np.isinf(self.x[:,-2:])] = 0
        self.x[:,-2:][np.isnan(self.x[:,-2:])] = 0

        assert(np.isnan(self.x[:,-2]).sum() == 0)
        assert(np.isinf(self.x[:,-2]).sum() == 0)
        assert(np.isnan(self.x[:,-1]).sum() == 0)
        assert(np.isinf(self.x[:,-1]).sum() == 0)




class JaneStreetTabNetSampler(Sampler):

    def __init__(self, X, y, c, batch_size, model, master=None):

        self.X = X
        self.y = y
        self.c = c
        self.batch_size = batch_size
        self.model = model
        self.master = master

        self.num_days = 0
        self.num_days_to_add = 50
        self.days = np.arange(int(c[-1,0]))
        self.filtered = None
        self.num_batches = 0
        self.t_target = 5

    def __iter__(self):

        if self.master is None:

            if not hasattr(JaneStreetTabNetSampler, 't') or JaneStreetTabNetSampler.t > self.t_target:

                self.num_days = min(self.num_days + self.num_days_to_add, self.days.shape[0])
                print('----------------------------')
                print('Total days:', self.num_days)

                # including first 'num_days' days in the batches
                self.filtered = np.where(np.isin(self.c[:,0], self.days[:int(self.num_days)]))[0]

                # resetting lr
                for param_group in self.model._optimizer.param_groups:
                    param_group['lr'] = np.mean((param_group['lr'], 1e-2*(0.994)**self.num_days))
            else:
                # scheduler
                for param_group in self.model._optimizer.param_groups:
                    param_group['lr'] *= 0.95
            
            for param_group in self.model._optimizer.param_groups:
                print('lr:', round(param_group['lr'],6))

            np.random.seed(0)
            np.random.shuffle(self.filtered)

        else:
            self.filtered = np.where(np.isin(self.c[:,0], self.master.days[:int(self.master.num_days)]))[0]

        self.num_batches = self.filtered.shape[0] // self.batch_size
        remaining = self.filtered.shape[0] % self.batch_size

        for batch_idx in np.arange(self.num_batches):
            yield self.filtered[batch_idx*self.batch_size : (batch_idx+1)*self.batch_size]

        if remaining > 0:
            yield self.filtered[-remaining:]

    def __len__(self):
        return self.num_batches


def my_collate(batch):
    return batch



def my_create_dataloaders(
    X_train, y_train, eval_set, weights, batch_size, num_workers, drop_last, pin_memory, context_fit, context_eval_set, model
):

    need_shuffle, sampler = create_sampler(weights, y_train)

    master = JaneStreetTabNetSampler(X_train, y_train, context_fit, batch_size, model)

    tds = JaneStreetTabNetDataset(X_train.astype(np.float32), y_train, context_fit)
    
    train_dataloader = DataLoader(
        tds,
        #batch_size=batch_size,
        #sampler=sampler,
        #shuffle=need_shuffle,
        batch_sampler=master,
        num_workers=num_workers,
        #collate_fn=my_collate,
        drop_last=False,
        pin_memory=pin_memory,
    )

    JaneStreetTabNetSampler.fittingset = tds

    valid_dataloaders = []
    i = 0

    for X, y in eval_set:
        valid_dataloaders.append(
            DataLoader(
                JaneStreetTabNetDataset(X.astype(np.float32), y, context_eval_set[i]),
                #batch_size=batch_size,
                #shuffle=False,
                batch_sampler=JaneStreetTabNetSampler(X, y, context_eval_set[i], batch_size, model, master),
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        )
        i = i + 1

    return train_dataloader, valid_dataloaders


class JaneStreetTabNetClassifier(TabNetClassifier):
    
    def __post_init__(self):
        super(JaneStreetTabNetClassifier, self).__post_init__()
        self._task = 'classification'
        self._default_loss = torch.nn.functional.cross_entropy
        self._default_metric = 'accuracy'
    
    # TabModel
    def fit(self, X_train, y_train, eval_set=None, eval_name=None, eval_metric=None, loss_fn=None, weights=0, max_epochs=100,
        patience=10, batch_size=1024, virtual_batch_size=128, num_workers=0, drop_last=False, callbacks=None,
        pin_memory=True, from_unsupervised=None, context_fit=None, context_eval_set=None
    ):

        # update model name

        self.context_fit = context_fit
        self.context_eval_set = context_eval_set
        
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.input_dim = X_train.shape[1]
        self._stop_training = False
        self.pin_memory = pin_memory and (self.device.type != "cpu")

        eval_set = eval_set if eval_set else []

        if loss_fn is None:
            self.loss_fn = self._default_loss
        else:
            self.loss_fn = loss_fn

        check_array(X_train)

        self.update_fit_params(
            X_train,
            y_train,
            eval_set,
            weights,
        )

        # Validate and reformat eval set depending on training data
        eval_names, eval_set = validate_eval_set(eval_set, eval_name, X_train, y_train)

        train_dataloader, valid_dataloaders = self._construct_loaders(
            X_train, y_train, eval_set
        )

        if from_unsupervised is not None:
            # Update parameters to match self pretraining
            self.__update__(**from_unsupervised.get_params())

        if not hasattr(self, "network"):
            self._set_network()
        self._update_network_params()
        self._set_metrics(eval_metric, eval_names)
        self._set_optimizer()
        self._set_callbacks(callbacks)

        if from_unsupervised is not None:
            print("Loading weights from unsupervised pretraining")
            self.load_weights_from_unsupervised(from_unsupervised)

        # Call method on_train_begin for all callbacks
        self._callback_container.on_train_begin()

        print("Number of parameters: %d" % sum([p.numel() for p in self.network.parameters()]))

        # Training loop over epochs
        for epoch_idx in range(self.max_epochs):

            # Call method on_epoch_begin for all callbacks
            self._callback_container.on_epoch_begin(epoch_idx)

            self._train_epoch(train_dataloader)

            # Apply predict epoch to all eval sets
            for eval_name, valid_dataloader in zip(eval_names, valid_dataloaders):
                self._predict_epoch(eval_name, valid_dataloader)

            # Call method on_epoch_end for all callbacks
            self._callback_container.on_epoch_end(
                epoch_idx, logs=self.history.epoch_metrics
            )

            if self._stop_training:
                break

        # Call method on_train_end for all callbacks
        self._callback_container.on_train_end()
        self.network.eval()

        # compute feature importance once the best model is defined
        self._compute_feature_importances(train_dataloader)
    
    # TabModel
    def _train_batch(self, X, y, con):

        batch_logs = {"batch_size": X.shape[0]}

        X = X.to(self.device).float()
        y = y.to(self.device).float()

        for param in self.network.parameters():
            param.grad = None

        output, M_loss = self.network(X)

        loss = self.compute_loss(output, y, con)
        # Add the overall sparsity loss
        loss -= self.lambda_sparse * M_loss

        # Perform backward pass and optimization
        loss.backward()
        if self.clip_value:
            clip_grad_norm_(self.network.parameters(), self.clip_value)
        self._optimizer.step()

        batch_logs["loss"] = loss.cpu().detach().numpy().item()

        return batch_logs
    
    def weight_updater(self, weights):

        if isinstance(weights, int):
            return weights
        elif isinstance(weights, dict):
            return {self.target_mapper[key]: value for key, value in weights.items()}
        else:
            return weights

    def prepare_target(self, y):
        return np.vectorize(self.target_mapper.get)(y)

    def compute_loss(self, y_pred, y_true, con):    
        return self.loss_fn(y_pred, y_true.long(), con)# <-------------------- CHANGE

    def update_fit_params(
        self,
        X_train,
        y_train,
        eval_set,
        weights,
    ):
        output_dim, train_labels = infer_output_dim(y_train)
        for X, y in eval_set:
            check_output_dim(train_labels, y)
        self.output_dim = output_dim
        self._default_metric = ('auc' if self.output_dim == 2 else 'accuracy')
        self.classes_ = train_labels
        self.target_mapper = {
            class_label: index for index, class_label in enumerate(self.classes_)
        }
        self.preds_mapper = {
            index: class_label for index, class_label in enumerate(self.classes_)
        }
        self.updated_weights = self.weight_updater(weights)

    def _construct_loaders(self, X_train, y_train, eval_set):

        # all weights are not allowed for this type of model
        y_train_mapped = self.prepare_target(y_train)
        for i, (X, y) in enumerate(eval_set):
            y_mapped = self.prepare_target(y)
            eval_set[i] = (X, y_mapped)

        train_dataloader, valid_dataloaders = my_create_dataloaders(
            X_train,
            y_train_mapped,
            eval_set,
            self.updated_weights,
            self.batch_size,
            self.num_workers,
            self.drop_last,
            self.pin_memory,
            self.context_fit,
            self.context_eval_set,
            self
        )
        return train_dataloader, valid_dataloaders
    
    def _train_epoch(self, train_loader):

        self.network.train()

        for batch_idx, (X, y, c) in enumerate(train_loader):
            self._callback_container.on_batch_begin(batch_idx)

            batch_logs = self._train_batch(X, y, c)# <-------------------- CHANGE

            self._callback_container.on_batch_end(batch_idx, batch_logs)

        epoch_logs = {"lr": self._optimizer.param_groups[-1]["lr"]}
        self.history.epoch_metrics.update(epoch_logs)

        return

    def _predict_epoch(self, name, loader):

        # Setting network on evaluation mode
        self.network.eval()

        list_y_true = []
        list_y_score = []
        list_context = []

        # Main loop
        for batch_idx, (X, y, c) in enumerate(loader):
            scores = self._predict_batch(X)
            list_y_true.append(y)
            list_y_score.append(scores)
            list_context.append(c)

        y_true, scores, contexts = self.stack_batches(list_y_true, list_y_score, list_context)

        #TODO: update_train(scores)
        if name == 'train':
            JaneStreetTabNetSampler.fittingset.update_rolling_features(scores)

        metrics_logs = self._metric_container_dict[name](y_true, scores, contexts)
        self.network.train()
        self.history.epoch_metrics.update(metrics_logs)
        return
    
    def stack_batches(self, list_y_true, list_y_score, list_context):
        y_true = np.hstack(list_y_true)
        y_score = np.vstack(list_y_score)
        y_score = softmax(y_score, axis=1)
        contexts = np.vstack(list_context)
        return y_true, y_score, contexts
    
    def _compute_feature_importances(self, loader):

        self.network.eval()
        feature_importances_ = np.zeros((self.network.post_embed_dim))
        for data, targets, c in loader:
            data = data.to(self.device).float()
            M_explain, masks = self.network.forward_masks(data)
            feature_importances_ += M_explain.sum(dim=0).cpu().detach().numpy()

        feature_importances_ = csc_matrix.dot(
            feature_importances_, self.reducing_matrix
        )
        self.feature_importances_ = feature_importances_ / np.sum(feature_importances_)

    def _set_metrics(self, metrics, eval_names):
        """Set attributes relative to the metrics.
        Parameters
        ----------
        metrics : list of str
            List of eval metric names.
        eval_names : list of str
            List of eval set names.
        """
        metrics = metrics or [self._default_metric]

        metrics = check_metrics(metrics)
        # Set metric container for each sets
        self._metric_container_dict = {}
        for name in eval_names:
            self._metric_container_dict.update(
                {name: MetricContainer(metrics, prefix=f"{name}_")}
            )

        self._metrics = []
        self._metrics_names = []
        for _, metric_container in self._metric_container_dict.items():
            self._metrics.extend(metric_container.metrics)
            self._metrics_names.extend(metric_container.names)

        # Early stopping metric is the last eval metric
        self.early_stopping_metric = (
            self._metrics_names[-1] if len(self._metrics_names) > 0 else None
        )

    def load_model(self, filepath):

        try:
            with zipfile.ZipFile(filepath) as z:
                with z.open("model_params.json") as f:
                    loaded_params = json.load(f)
                    loaded_params["device_name"] = self.device_name
                with z.open("network.pt") as f:
                    try:
                        saved_state_dict = torch.load(f, map_location=self.device)
                    except io.UnsupportedOperation:
                        # In Python <3.7, the returned file object is not seekable (which at least
                        # some versions of PyTorch require) - so we'll try buffering it in to a
                        # BytesIO instead:
                        saved_state_dict = torch.load(
                            io.BytesIO(f.read()),
                            map_location=self.device,
                        )
        except KeyError:
            raise KeyError("Your zip file is missing at least one component")

        self.__init__(**loaded_params)

        self._set_network()
        self.network.load_state_dict(saved_state_dict)
        self.network.eval()

        return

    def fit_tabnet(self, X_train, y_train, c_train, X_val, y_val, c_val, model=None, batch_size=50000, max_epochs=500, patience=500):

        class JaneStreetCallback(Callback):
        
            def on_epoch_end(self, epoch, logs=None):

                JaneStreetTabNetSampler.t = logs['train_t']

        torch.manual_seed(0)
        np.random.seed(0)
        clf = JaneStreetTabNetClassifier()

        if os.path.isfile("./jane_street_tabnet.zip"):
            clf.load_model("./jane_street_tabnet.zip")
            print("Parameters loaded from file.")
        elif model is None:
            print('clf is None, loading default.')
            torch.manual_seed(0)
            np.random.seed(0)
            clf = JaneStreetTabNetClassifier(n_d=32,
                                            n_a=32,
                                            n_steps=5,
                                            gamma=1.5,
                                            n_independent=1,
                                            n_shared=1,
                                            momentum=0.2,
                                            clip_value=2.,
                                            cat_idxs=[0],
                                            cat_dims=[2],
                                            cat_emb_dim=1,
                                            lambda_sparse=1e-05,
                                            optimizer_fn=torch.optim.Adam,
                                            optimizer_params=dict(lr=1e-3),
                                            mask_type='entmax')
        else:
            clf = model

        torch.manual_seed(0)
        np.random.seed(0)
        clf.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_name=['train', 'eval'],
            loss_fn = loss_fn,
            batch_size = batch_size,
            virtual_batch_size=batch_size//8,
            eval_metric=[t_metric, p_metric, u_metric],
            #num_workers=4, #*
            patience=patience,
            #weights=1,
            max_epochs=max_epochs,
            context_fit=c_train,
            context_eval_set=[c_train, c_val],
            callbacks=[JaneStreetCallback()]
        )
        # * num_workers bug with Windows:
        # https://github.com/pytorch/pytorch/issues/2341
        # https://github.com/pytorch/pytorch/issues/12831

        saved_filepath = clf.save_model("./models/jane_street_tabnet")
        print('Parameters saved in file', saved_filepath)

        return clf