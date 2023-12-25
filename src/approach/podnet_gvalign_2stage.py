import copy
import math
import torch
import warnings
from torch import nn
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.nn import Module, Parameter
from torch.utils.data import DataLoader
import numpy as np
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
from .lucir import BasicBlockNoRelu
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import nn
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.nn import Module, Parameter
from torch.utils.data import DataLoader
from .LAS_utils import mixup_data, mixup_criterion,LabelAwareSmoothing, LearnableWeightScaling
import  datasets.data_loader as stage2_utils
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import time
lambda_c_base = 5
lambda_f_base = 1

class Appr(Inc_Learning_Appr):
    """Class implementing the Learning a Unified Classifier Incrementally via Rebalancing (LUCI) approach
    described in http://dahua.me/publications/dhl19_increclass.pdf
    Original code available at https://github.com/hshustc/CVPR19_Incremental_Learning
    """

    # Sec. 4.1: "we used the method proposed in [29] based on herd selection" and "first one stores a constant number of
    # samples for each old class (e.g. R_per=20) (...) we adopt the first strategy"
    def __init__(self, model, device, nepochs=160, lr=0.1, lr_min=1e-4, lr_factor=10, lr_patience=8, clipgrad=10000,
                 momentum=0.9, wd=5e-4, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger=None, exemplars_dataset=None, lamb=5., pod_flat_factor=1., 
                 pod_spatial_factor=3., remove_adapt_lamda=False, remove_pod_flat=False, remove_pod_spatial=False, 
                 remove_cross_entropy=False, pod_pool_type="spatial"):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.lamb = lamb
        self.adapt_lamda = not remove_adapt_lamda

        self.lamda = self.lamb
        self.ref_model = None
        self.warmup_loss = self.warmup_luci_loss

        self.pod_flat = not remove_pod_flat
        self.pod_spatial = not remove_pod_spatial
        self.nca_loss = not remove_cross_entropy
        self._pod_flat_factor = pod_flat_factor
        self._pod_spatial_factor = pod_spatial_factor
        self._pod_pool_type = pod_pool_type
        self._n_classes = 0
        self._task_size = 0
        self.task_percent = 0
        self.lambda_c_base = 5
        self.lambda_f_base = 1
        # LUCIR is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        if not have_exemplars:
            warnings.warn("Warning: LUCIR is expected to use exemplars. Check documentation.")

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Sec. 4.1: "lambda base is set to 5 for CIFAR100 and 10 for ImageNet"
        parser.add_argument('--lamb', default=5., type=float, required=False,
                            help='Trade-off for distillation loss (default=%(default)s)')
        parser.add_argument('--remove-adapt-lamda', action='store_true', required=False,
                            help='Deactivate adapting lambda according to the number of classes (default=%(default)s)')
        parser.add_argument('--pod-spatial-factor', default=3., type=float, required=False, 
                            help='Scaling factor for pod spatial loss (default=%(default)s)')
        parser.add_argument('--pod-flat-factor', default=1., type=float, required=False, 
                            help='Scaling factor for pod flat loss (default=%(default)s)')
        parser.add_argument('--remove-pod-flat', action='store_true', required=False,
                    help='Deactivate POD flat loss constraint (default=%(default)s)')
        parser.add_argument('--remove-pod-spatial', action='store_true', required=False,
                            help='Deactivate POD spatial loss constraint (default=%(default)s)')
        parser.add_argument('--pod-pool-type', default='spatial', type=str, choices=["channels", "width", "height", "gap", "spatial"],
                        help='POD spatial pooling dimension used (default=%(default)s)', metavar="POOLTYPE")
        parser.add_argument('--remove-cross-entropy', action='store_true', required=False,
                            help='Deactivate cross entropy loss and use NCA loss instead (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        # Don't update heads when Less-Forgetting constraint is activated (from original code)
        params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)
    
    def _get_scheduler(self, optimizer):
        """Yet to figure out how to use this without modifying the super class file!!"""
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.nepochs)
        return scheduler
    def protosave(self, model, loader, current_task, num_of_classes):
        print("saving protos...")
        features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for images, targets in loader:
                images, target = images.to(self.device), targets.to(self.device)
            # Forward current model
                _, feature = model(images, return_features=True)
 
                labels.append(target.cpu())
                features.append(feature['features'].cpu())


        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)

        prototype = []
        radius = []
        class_label = []
        cov_list = []
        num_of_samples = []
        for class_index in range(num_of_classes):
            
            # breakpoint()
            data_index = (labels == class_index).nonzero()
            # print('class index', class_index, data_index.shape)
            embedding = features[data_index.squeeze(-1)]
            embedding = F.normalize(embedding, p=2, dim=-1)
            #print('dim of emd', embedding_this.shape)
            #print(c)
            feature_class_wise = embedding.numpy()
            cov = np.cov(feature_class_wise.T)
            cov_torch = torch.cov(embedding.T)
            # print('class index', class_index)
            # print('class index', class_index, torch.linalg.eig(cov_torch)[0])
            radius.append(np.trace(cov)/64)
            print('class index', class_index, 'number of samples',data_index.shape[0],'radius', radius[-1])
            num_of_samples.append(data_index.shape[0])
            embedding_mean = embedding.mean(0)
            prototype.append(embedding_mean )
            # cov_list.append(torch.tensor(cov))
            cov_list.append(cov_torch)
        
        self.radius = np.sqrt(np.mean(radius)) 
        self.radius_list = radius
        self.proto_list = torch.stack(prototype, dim=0)
        self.cov_list = torch.stack(cov_list, dim=0)
        self.num_of_samples = torch.tensor(num_of_samples)

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""
        self.t = t
        # Changes the new head to a CosineLinear
        print("have {} paramerters in total".format(sum(x.numel() for x in self.model.parameters())))
        self.model.heads[-1] = CosineLinear(self.model.heads[-1].in_features, self.model.heads[-1].out_features, nb_proxy=10, to_reduce=True)
       
        self.model.to(self.device)
        if t > 0:
            # Share sigma (Eta in paper) between all the heads
            self.model.heads[-1].sigma = self.model.heads[-2].sigma
            for h in self.model.heads[:-1]:
                for param in h.parameters():
                    param.requires_grad = False
            self.model.heads[-1].sigma.requires_grad = True
            # Eq. 7: Adaptive lambda
            if self.adapt_lamda:
                self.lamda = self.lamb * math.sqrt(sum([h.out_features for h in self.model.heads[:-1]])
                                                   / self.model.heads[-1].out_features)
        # The original code has an option called "imprint weights" that seems to initialize the new head.
        # However, this is not mentioned in the paper and doesn't seem to make a significant difference.
        super().pre_train_process(t, trn_loader)

    def train_loop(self, t, trn_loader, val_loader, args,tst_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)
        

        # FINETUNING TRAINING -- contains the epochs loop
        for parameter in self.model.model.parameters():
            parameter.requires_grad = True
        print("trainable_parameters_list....")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)
        super().train_loop(t, trn_loader, val_loader, args,tst_loader)
        # if t==0:
        #     torch.save(self.model.state_dict(),'modeltask0podnet50_10stepsltio.pt')

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)
        unique_classes = set()
        for images, labels in trn_loader:
            unique_classes.update(labels.unique().tolist())

        num_unique_classes = len(unique_classes)

        print("Number of unique classes:", num_unique_classes)
        self.protosave(self.model, trn_loader,t, num_unique_classes)
        # print('checking..', self.proto_list.shape, self.cov_list.shape,self.radius)
        class_id_most_samples = torch.argmax(self.num_of_samples)
        if t==0:
            self.cov_cls_ms_major = self.cov_list[class_id_most_samples]

            # breakpoint()
        if t >= 0:

            cov_cls_ms = self.cov_cls_ms_major.repeat(num_unique_classes, 1, 1)
            mean = self.proto_list
            distrib = MultivariateNormal(loc=mean, covariance_matrix=cov_cls_ms)
            for parameter in self.model.model.parameters():
                parameter.requires_grad = False

            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

            for parameter in self.model.heads.parameters():
                parameter.requires_grad = True
            
            print("trainable_parameters_list....")
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(name)
            params = self.model.heads.parameters()
            lr = 0.1
            best_loss = np.inf
            self.optimizer_classifier_tune = torch.optim.SGD(params, lr=lr, weight_decay=self.wd, momentum=self.momentum)
            for e in range(200):
                # Train
                clock0 = time.time()
                # self.train_epoch(t, trn_loader)
                self.optimizer_classifier_tune = self.classifier_tune(t, distrib, num_unique_classes)
                clock1 = time.time()
                self.eval_on_train = False
                if self.eval_on_train:
                    train_loss, train_acc, _ = self.eval(t, trn_loader)
                    clock2 = time.time()
                    print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                        e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                    self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                    self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
                else:
                    print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

                # Valid
                clock3 = time.time()
                valid_loss, valid_acc, _ = self.eval(t, val_loader)
                clock4 = time.time()
                print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    clock4 - clock3, valid_loss, 100 * valid_acc), end='')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

                # Adapt learning rate - patience scheme - early stopping regularization
                if valid_loss < best_loss:
                    # if the loss goes down, keep it as the best model and end line with a star ( * )
                    best_loss = valid_loss
                    patience = self.lr_patience
                    print(' *', end='')
                if e+1 in self.model.schedule_step:
                    lr/=self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    self.optimizer_classifier_tune.param_groups[0]['lr'] = lr
                    # self.model.set_state_dict(best_model)
                if e == 100:
                    self.optimizer_classifier_tune.param_groups[0]['lr'] = self.lr * 0.1
                    
                if  e == 150 :
                    self.optimizer_classifier_tune.param_groups[0]['lr'] = self.lr * 0.1
                self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
                print()

            valid_loss, valid_acc, _ = self.eval(t, val_loader)
            print('Valid loss:', valid_loss, 'Acc', valid_acc)

            

            max_task = args.max_task
            acc_taw = np.zeros((max_task, max_task))
            acc_tag = np.zeros((max_task, max_task))
            forg_taw = np.zeros((max_task, max_task))
            forg_tag = np.zeros((max_task, max_task))

            for u in range(t + 1):
                # test_loss, acc_taw[t, u], acc_tag[t, u] = self.eval(u, tst_loader[u])
                test_loss, acc_taw[t, u], acc_tag[t, u] = self.eval(u, tst_loader[u])
            
                if u < t:
                    forg_taw[t, u] = acc_taw[:t, u].max(0) - acc_taw[t, u]
                    forg_tag[t, u] = acc_tag[:t, u].max(0) - acc_tag[t, u]
                print('>>> Test on task {:2d} : loss={:.3f} | TAw acc={:5.1f}%, forg={:5.1f}%'
                    '| TAg acc={:5.1f}%, forg={:5.1f}% <<<'.format(u, test_loss,
                                                                    100 * acc_taw[t, u], 100 * forg_taw[t, u],
                                                                    100 * acc_tag[t, u], 100 * forg_tag[t, u]))
    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        # Make the old model return outputs without the sigma (eta in paper) factor
        for h in self.ref_model.heads:
            h.train()
        self.ref_model.freeze_all()
        
    def classifier_tune(self, t, distrib, num_of_classes):
        """Runs a single epoch"""
        # self.lws_models.eval()
        self.model.train()
        
        samples = distrib.rsample()
        # breakpoint()
        targets = torch.arange(num_of_classes).cuda()
        # print(sample.shape, targets)
        outputs = []
        for head in self.model.heads:
            outputs.append(head(samples.cuda().float()))

        outputs = torch.cat(outputs, dim=1)
        

        loss = nn.CrossEntropyLoss(None)(outputs, targets.long())
        # Backward
        self.optimizer_classifier_tune.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
        self.optimizer_classifier_tune.step()

        return self.optimizer_classifier_tune
    def classAug(self,x, y,alpha=20.0, mix_times=4):  # mixup based
        batch_size = x.size()[0]
        mix_data = []
        mix_target = []
        for _ in range(mix_times):
            index = torch.randperm(batch_size).cuda()
            for i in range(batch_size):
                if y[i] != y[index][i]:
                    new_label = self.generate_label(y[i].item(), y[index][i].item())
                    lam = np.random.beta(alpha, alpha)
                    if lam < 0.4 or lam > 0.6:
                        lam = 0.5
                    mix_data.append(lam * x[i] + (1 - lam) * x[index, :][i])
                    mix_target.append(new_label)

        new_target = torch.Tensor(mix_target)
        y = torch.cat((y, new_target.cuda().long()), 0)
        for item in mix_data:
            x = torch.cat((x, item.unsqueeze(0)), 0)
        return x, y

    def generate_label(self, y_a, y_b):
        
        y_a, y_b = y_a, y_b
        assert y_a != y_b
        if y_a > y_b:
            tmp = y_a
            y_a = y_b
            y_b = tmp
        label_index = ((2 * self.total_class - y_a - 1) * y_a) / 2 + (y_b - y_a) - 1
        
        return label_index + self.total_class

    def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
        # print(alpha)
        
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam, 0

    def mixup_criterion(self, pred, y_a, y_b, lam, org_bs):
        # pred = torch.cat([o['wsigma'] for o in pred], dim=1)[org_bs:]
        pred =  torch.cat(pred, dim=1)[org_bs:]
        return lam * nn.CrossEntropyLoss()(pred, y_a) + (1 - lam) * nn.CrossEntropyLoss()(pred, y_b)

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        # breakpoint()
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            images, targets = images.to(self.device), targets.to(self.device)
            org_bs = images.shape[0]
            
            mix_images,targets_a, targets_b, lam, index_perm = self.mixup_data(images, targets)
            images = torch.cat([images, mix_images])
            outputs, features_ = self.model(images, return_features=True)
            fmaps = features_['fmaps']
            features = features_['features']
            # Forward previous model
            ref_features = None
            ref_fmaps = None
            if t > 0:
                _, ref_features_ = self.ref_model(images, return_features=True)
                ref_features = ref_features_['features']
                ref_fmaps = ref_features_['fmaps']

            loss_pod = self.criterion(t, outputs, targets, features, fmaps, ref_features, ref_fmaps, org_bs=org_bs)
            loss_mix = self.mixup_criterion(outputs, targets_a, targets_b, lam, org_bs)
            # Backward
            loss = loss_pod + loss_mix
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward current model
                outputs, features_ = self.model(images.to(self.device), return_features=True)
                # outputs = [outputs[idx]['logits'] for idx in range(len(outputs))]
                fmaps = features_['fmaps']
                features = features_['features']      

                ref_features = None
                ref_fmaps = None
                if t > 0:
                    _, ref_features_ = self.ref_model(images.to(self.device), return_features=True)
                    ref_features = ref_features_['features']
                    ref_fmaps = ref_features_['fmaps']
          
                loss = self.criterion(t, outputs, targets.to(self.device), features, fmaps, ref_features, ref_fmaps)
                hits_taw, hits_tag,_,_ = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def criterion(self, t, outputs, targets, features, fmaps,  ref_features=None, ref_fmaps=None, ref_outputs=None, int_outputs = None, org_bs=None):
        loss = 0
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs[:org_bs]
       
        
        if self.nca_loss:
            lsc_loss = nca(outputs, targets)
            loss += lsc_loss
        else:
            ce_loss = nn.CrossEntropyLoss(None)(outputs, targets)
            loss += ce_loss
        
        if ref_features is not None:
            if self.pod_flat:
                factor = self._pod_flat_factor * math.sqrt(
                        self._n_classes / self._task_size
                    )
                # pod flat loss is equivalent to less forget constraint loss acting on the final embeddings
                pod_flat_loss = F.cosine_embedding_loss(features, ref_features.detach(),
                                    torch.ones(features.shape[0]).to(self.device)) * factor
                loss += pod_flat_loss

            if self.pod_spatial:
                factor = self._pod_spatial_factor * math.sqrt(
                    self._n_classes / self._task_size
                )
                spatial_loss = pod_spatial_loss(fmaps, ref_fmaps, collapse_channels=self._pod_pool_type) * factor        
                loss += spatial_loss
        return loss

    @staticmethod
    def warmup_luci_loss(outputs, targets):
        if type(outputs) == dict:
            # needed during train
            return torch.nn.functional.cross_entropy(outputs['wosigma'], targets)
        else:
            # needed during eval()
            return torch.nn.functional.cross_entropy(outputs, targets)



class CosineLinear(nn.Module):
    """
    Implementation inspired by https://github.com/zhchuu/continual-learning-reproduce/blob/master/utils/inc_net.py#L139
    """
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        if type(input) is dict:
            input = input['features']
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None and self.training:
            out = self.sigma * out

        return out


def reduce_proxies(out, nb_proxy):
    if nb_proxy == 1:
        return out
    bs = out.shape[0] # e.g. out.shape = [128, 500] for base task with 50 classes
    nb_classes = out.shape[1] / nb_proxy
    assert nb_classes.is_integer(), 'Shape error'
    nb_classes = int(nb_classes)

    simi_per_class = out.view(bs, nb_classes, nb_proxy) # shape becomes [128, 50, 10]
    attentions = F.softmax(simi_per_class, dim=-1)

    return (attentions * simi_per_class).sum(-1)


def pod_spatial_loss(list_attentions_old, list_attentions_new, normalize=True, collapse_channels="spatial"):
    """Pooled Output Distillation.
    Reference:
        * Douillard et al.
        Small Task Incremental Learning.
        arXiv 2020.
    Note: My comments assume an input attention vector of [128, 16, 32, 32] dimensions which is standard for CIFAR100 and Resnet-32 model
    :param list_attentions_old: A list of attention maps, each of shape (b, n, w, h).
    :param list_attentions_new: A list of attention maps, each of shape (b, n, w, h).
    :param collapse_channels: How to pool the channels.
    :return: A float scalar loss.
    """
    loss = torch.tensor(0.).to(list_attentions_new[0].device)
    for i, (a, b) in enumerate(zip(list_attentions_old, list_attentions_new)):
        assert a.shape == b.shape, 'Shape error'

        a = torch.pow(a, 2)
        b = torch.pow(b, 2)
        # collapse_channels = "spatial"
        # print("pod channel: ", a.shape)
        if collapse_channels == "channels":
            a = a.sum(dim=1).view(a.shape[0], -1)  # transforms a = [128, 16, 32, 32] into a = [128, 1024], i.e., sums up and removes the channel information and view() collapses the -1 labelled dimensions into one
            b = b.sum(dim=1).view(b.shape[0], -1)
        elif collapse_channels == "width":
            # pod-width and height trade plasticity for rigidity with less agressive pooling
            a = a.sum(dim=2).view(a.shape[0], -1)  # a = [128, 16, 32, 32] into [128, 512]: sums up along 2nd dim
            b = b.sum(dim=2).view(b.shape[0], -1)
        elif collapse_channels == "height":
            a = a.sum(dim=3).view(a.shape[0], -1)  # shape of (b, c * w), also into [128, 512]
            b = b.sum(dim=3).view(b.shape[0], -1)
        elif collapse_channels == "gap":
            # compute avg pool2d over each 32x32 image to reduce the dimension to 1x1
            a = F.adaptive_avg_pool2d(a, (1, 1))[..., 0, 0] # [..., 0, 0] preserves only the [0][0]th element of last two dimensions, i.e., [128, 16, 32, 32] into [128, 16], since 32x32 reduced to 1x1 and merged together
            b = F.adaptive_avg_pool2d(b, (1, 1))[..., 0, 0]
        elif collapse_channels == "spatial":
            a_h = a.sum(dim=3).view(a.shape[0], -1)  # [bs, c*w]
            b_h = b.sum(dim=3).view(b.shape[0], -1)  # [bs, c*w]
            a_w = a.sum(dim=2).view(a.shape[0], -1)  # [bs, c*h]
            b_w = b.sum(dim=2).view(b.shape[0], -1)  # [bs, c*h]
            a = torch.cat([a_h, a_w], dim=-1) # concatenates two [128, 512] to give [128, 1024], dim = -1 does concatenation along the last axis
            b = torch.cat([b_h, b_w], dim=-1)
        elif collapse_channels == "spatiochannel":
            a_h = a.sum(dim=3).view(a.shape[0], -1)  # [bs, c*w]
            b_h = b.sum(dim=3).view(b.shape[0], -1)  # [bs, c*w]
            a_w = a.sum(dim=2).view(a.shape[0], -1)  # [bs, c*h]
            b_w = b.sum(dim=2).view(b.shape[0], -1)  # [bs, c*h]
            a1 = torch.cat([a_h, a_w], dim=-1) # concatenates two [128, 512] to give [128, 1024], dim = -1 does concatenation along the last axis
            b1 = torch.cat([b_h, b_w], dim=-1)
            
            a2 = a.sum(dim=1).view(a.shape[0], -1)  # transforms a = [128, 16, 32, 32] into a = [128, 1024], i.e., sums up and removes the channel information and view() collapse the -1 labelled dimensions into one
            b2 = b.sum(dim=1).view(b.shape[0], -1)
            a = torch.cat([a1, a2], dim=-1)
            b = torch.cat([b1, b2], dim=-1)
        elif collapse_channels == "spatiogap":
            a_h = a.sum(dim=3).view(a.shape[0], -1)  # [bs, c*w]
            b_h = b.sum(dim=3).view(b.shape[0], -1)  # [bs, c*w]
            a_w = a.sum(dim=2).view(a.shape[0], -1)  # [bs, c*h]
            b_w = b.sum(dim=2).view(b.shape[0], -1)  # [bs, c*h]
            a1 = torch.cat([a_h, a_w], dim=-1) # concatenates two [128, 512] to give [128, 1024], dim = -1 does concatenation along the last axis
            b1 = torch.cat([b_h, b_w], dim=-1)

            a2 = F.adaptive_avg_pool2d(a, (1, 1))[..., 0, 0] # [..., 0, 0] preserves only the [0][0]th element of last two dimensions, i.e., [128, 16, 32, 32] into [128, 16], since 32x32 reduced to 1x1 and merged together
            b2 = F.adaptive_avg_pool2d(b, (1, 1))[..., 0, 0]

            a = torch.cat([a1, a2], dim=-1)
            b = torch.cat([b1, b2], dim=-1)
        else:
            raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
        loss += layer_loss

    return loss / len(list_attentions_old)


def nca(
    similarities,
    targets,
    class_weights=None,
    scale=1.0,
    margin=0.6,
    exclude_pos_denominator=False,
    hinge_proxynca=False,
):
    """Compute AMS cross-entropy loss.
    Copied from: https://github.com/arthurdouillard/incremental_learning.pytorch/blob/master/inclearn/lib/losses/base.py
    Reference:
        * Goldberger et al.
          Neighbourhood components analysis.
          NeuriPS 2005.
        * Feng Wang et al.
          Additive Margin Softmax for Face Verification.
          Signal Processing Letters 2018.
    :param similarities: Result of cosine similarities between weights and features.
    :param targets: Sparse targets.
    :param scale: Multiplicative factor, can be learned.
    :param margin: Margin applied on the "right" (numerator) similarities.
    :param memory_flags: Flags indicating memory samples, although it could indicate
                         anything else.
    :return: A float scalar loss.
    """
    margins = torch.zeros_like(similarities)
    margins[torch.arange(margins.shape[0]), targets] = margin
    similarities = scale * (similarities - margin)

    if exclude_pos_denominator:  # NCA-specific
        similarities = similarities - similarities.max(1)[0].view(-1, 1)  # Stability

        disable_pos = torch.zeros_like(similarities)
        disable_pos[torch.arange(len(similarities)),
                    targets] = similarities[torch.arange(len(similarities)), targets]

        numerator = similarities[torch.arange(similarities.shape[0]), targets]
        denominator = similarities - disable_pos

        losses = numerator - torch.log(torch.exp(denominator).sum(-1))
        if class_weights is not None:
            losses = class_weights[targets] * losses

        losses = -losses
        if hinge_proxynca:
            losses = torch.clamp(losses, min=0.)

        loss = torch.mean(losses)
        return loss

    return F.cross_entropy(similarities, targets, weight=class_weights, reduction="mean")


class BasicBlockNoRelu(nn.Module):
    expansion = 1

    def __init__(self, conv1, bn1, relu, conv2, bn2, downsample):
        super(BasicBlockNoRelu, self).__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.relu = relu
        self.conv2 = conv2
        self.bn2 = bn2
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        # Removed final ReLU
        return out
