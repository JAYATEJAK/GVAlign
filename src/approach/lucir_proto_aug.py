import copy
import math
import torch
import warnings
import time
import numpy as np
from torch import nn
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.nn import Module, Parameter
from torch.utils.data import DataLoader
from .LAS_utils import mixup_data, mixup_criterion,LabelAwareSmoothing, LearnableWeightScaling
import  datasets.data_loader as stage2_utils
from torch.distributions.multivariate_normal import MultivariateNormal
import scipy
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
# import statsmodels.tools as st
# import statsmodels.stats.correlation_tools.cov_nearest as create_mat
#
class Appr(Inc_Learning_Appr):
    """Class implementing the Learning a Unified Classifier Incrementally via Rebalancing (LUCI) approach
    described in http://dahua.me/publications/dhl19_increclass.pdf
    Original code available at https://github.com/hshustc/CVPR19_Incremental_Learning
    """

    # Sec. 4.1: "we used the method proposed in [29] based on herd selection" and "first one stores a constant number of
    # samples for each old class (e.g. R_per=20) (...) we adopt the first strategy"
    def __init__(self, model, device, nepochs=160, lr=0.1, lr_min=1e-4, lr_factor=10, lr_patience=8, clipgrad=10000,
                 momentum=0.9, wd=5e-4, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger=None, exemplars_dataset=None, lamb=10., lamb_mr=1., dist=0.5, K=2,
                 remove_less_forget=False, remove_margin_ranking=False, remove_adapt_lamda=False,schedule_step = [80,120]):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)                
        self.lamb = lamb
        self.lamb_mr = lamb_mr
        self.dist = dist
        self.K = K
        self.less_forget = not remove_less_forget
        self.margin_ranking = not remove_margin_ranking
        self.adapt_lamda = not remove_adapt_lamda
        self.lws_models = torch.nn.ModuleList()

        self.lamda = self.lamb
        self.ref_model = None

        self.warmup_loss = self.warmup_luci_loss
        # LUCIR is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        if not have_exemplars:
            warnings.warn("Warning: LUCIR is expected to use exemplars. Check documentation.")

        self.radius = []
        self.radius_list = []
        self.prototype = []
        self.class_label = []
        self.conv_list = []

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
        # Loss weight for the Inter-Class separation loss constraint, set to 1 in the original code
        parser.add_argument('--lamb-mr', default=1., type=float, required=False,
                            help='Trade-off for the MR loss (default=%(default)s)')
        # Sec 4.1: "m is set to 0.5 for all experiments"
        parser.add_argument('--dist', default=.5, type=float, required=False,
                            help='Margin threshold for the MR loss (default=%(default)s)')
        # Sec 4.1: "K is set to 2"
        parser.add_argument('--K', default=2, type=int, required=False,
                            help='Number of "new class embeddings chosen as hard negatives '
                                 'for MR loss (default=%(default)s)')
        # Flags for ablating the approach
        parser.add_argument('--remove-less-forget', action='store_true', required=False,
                            help='Deactivate Less-Forget loss constraint(default=%(default)s)')
        parser.add_argument('--remove-margin-ranking', action='store_true', required=False,
                            help='Deactivate Inter-Class separation loss constraint (default=%(default)s)')
        parser.add_argument('--remove-adapt-lamda', action='store_true', required=False,
                            help='Deactivate adapting lambda according to the number of classes (default=%(default)s)')
        return parser.parse_known_args(args)
    def _get_optimizer(self):
        """Returns the optimizer"""
        if self.less_forget:
            # Don't update heads when Less-Forgetting constraint is activated (from original code)
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""
        lws_model = LearnableWeightScaling(num_classes=self.model.task_cls[t]).to(self.device)
        self.lws_models.append(lws_model)
        if t == 0:
            # Sec. 4.1: "the ReLU in the penultimate layer is removed to allow the features to take both positive and
            # negative values"
            if self.model.model.__class__.__name__ == 'ResNet':
                old_block = self.model.model.layer3[-1]
                self.model.model.layer3[-1] = BasicBlockNoRelu(old_block.conv1, old_block.bn1, old_block.relu,
                                                               old_block.conv2, old_block.bn2, old_block.downsample)
            else:
                warnings.warn("Warning: ReLU not removed from last block.")
        # Changes the new head to a CosineLinear
        self.model.heads[-1] = CosineLinear(self.model.heads[-1].in_features, self.model.heads[-1].out_features)
        self.model.to(self.device)
        if t > 0:
            # Share sigma (Eta in paper) between all the heads
            self.model.heads[-1].sigma = self.model.heads[-2].sigma
            # Fix previous heads when Less-Forgetting constraint is activated (from original code)
            if self.less_forget:
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
    
    def protosave(self, model, loader, current_task):
        print("saving protos...")
        features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for images, targets in loader:
                images, target = images.to(self.device), targets.to(self.device)
            # Forward current model
                _, feature = model(images, return_features=True)
            # for i, (indexs, images, target) in enumerate(loader):
                # feature = model.feature(images.to(self.device))
                # if feature.shape[0] == self.args.batch_size:
                # labels.append(target.cpu().numpy())   
                # features.append(feature.cpu().numpy())
                
                # labels.append(target.view(-1,1))   
                # features.append(feature)

                labels.append(target.cpu())
                features.append(feature.cpu())


        # breakpoint()
        # labels = torch.vstack(labels)
        # labels = labels.squeeze(1)
        # features = torch.vstack(features)

        # print(labels.shape, features.shape)

        # print(c)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        # print(labels.shape, features.shape)
        # print(c)
        
        # labels = np.array(labels.cpu())
        # labels_set = np.unique(labels)
        # labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        # features = np.array(features.cpu())
        # features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
        # feature_dim = features.shape[1]

        # breakpoint()

        prototype = []
        radius = []
        class_label = []
        cov_list = []
        for class_index in range(50):
            
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
            embedding_mean = embedding.mean(0)
            prototype.append(embedding_mean )
            # cov_list.append(torch.tensor(cov))
            cov_list.append(cov_torch)
        
        self.radius = np.sqrt(np.mean(radius)) 
        self.radius_list = radius
        self.proto_list = torch.stack(prototype, dim=0)
        self.cov_list = torch.stack(cov_list, dim=0)

    def train_loop(self, t, trn_loader, val_loader, args, tst_loader):
        """Contains the epochs loop"""

        checkpoint = torch.load('/home/teja/long_tail_cil/Long-Tailed-CIL/cifar100/50base_6tasks/results/Basic_expts_with_saved_models_300_epochs/cifar100_lt_lucir_no_gs_fixd_0/models/task0.ckpt')
        self.model.load_state_dict(checkpoint)
        print(self.model)
        self.protosave(self.model, trn_loader,t)
        print('checking..', self.proto_list.shape, self.cov_list.shape,self.radius)
        # breakpoint()
        # print()
        # print(c)

        # print(c)

    # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                    batch_size=trn_loader.batch_size,
                                                    shuffle=True,
                                                    num_workers=trn_loader.num_workers,
                                                    pin_memory=trn_loader.pin_memory)

        
        print("checking with count")
        count = torch.zeros(100)
        max_label = torch.zeros(1)
        for _,target in trn_loader:
            # max_label = torch.max(max_label, target)
            for j in target:
                max_label = torch.max(max_label, j)
                count[j] +=1

        class_num_list = count.tolist()
        print('class_count', class_num_list[:int(max_label+1)] )

        
        count_norm = count/torch.max(count)
        count_norm = (2 - count_norm) * 1.5
        print(count_norm[:int(max_label+1)])
        z = 0
        for j in class_num_list[:int(max_label+1)]:
            print('class id', z, 'no of samples', j, 'scaling factor', count_norm[z])
            z += 1
        
       
        lr = self.lr
        best_loss = np.inf
        patience = self.lr_patience
        # self.optimizer = self._get_optimizer()
        # print(self.model.model)
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
        self.optimizer = torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

        # breakpoint()
        # print(c)

        # Loop epochs
        mean = self.proto_list
        cov = self.cov_list
        radius_list =self.radius_list
        # cov_pos = []
        # for C_mat in cov:
            # breakpoint()
            # cov_pos.append(create_mat(C_mat, method='clipped', threshold=1e-15, n_fact=100, return_all=False))
            # print(C_mat.shape)
        # print(c)
        # distrib = MultivariateNormal(loc=mean, covariance_matrix=cov)
        print(mean.shape, cov.shape, len(radius_list))
        new_radius = torch.tensor(radius_list) * count_norm[:50]
        k = 0
        for j in new_radius:
            print('old->', radius_list[k], 'new->', j, 'scale->', count_norm[k])
            k += 1
        # breakpoint()
        # print(c)
        # distrib = scipy.stats.multivariate_normal(mean=mean.cpu().numpy(), cov=cov.cpu().numpy())



        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            # self.train_epoch(t, trn_loader)
            self.classifier_tune(t, new_radius)
            clock1 = time.time()
            if True:
            # if self.eval_on_train:
                train_loss, train_acc, _ = self.eval(t, tst_loader)
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
                self.optimizer.param_groups[0]['lr'] = lr
                # self.model.set_state_dict(best_model)
            self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
            print()
        # print(c)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

  
    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        # Make the old model return outputs without the sigma (eta in paper) factor
        for h in self.ref_model.heads:
            h.train()
        self.ref_model.freeze_all()

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.lws_models.eval()
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            images, targets = images.to(self.device), targets.to(self.device)
            # Forward current model
            outputs, features = self.model(images, return_features=True)
            lwsoutputs=[]
            ref_outputs = None
            ref_features = None
            if t > 0:
                ref_outputs, ref_features = self.ref_model(images, return_features=True)
            loss = self.criterion(t, outputs, targets, ref_outputs, features, ref_features,lwsoutputs)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def classifier_tune(self, t, new_radius):
        """Runs a single epoch"""
        # self.lws_models.eval()
        self.model.train()

        # sample = distrib.rsample()
        # print(sample.shape)
        samples = self.proto_list
        N_0I = torch.randn(samples.shape)
        # breakpoint()
        # print((new_radius * N_0I )[0])
        noise = torch.einsum('i,ij->ij',new_radius,N_0I)
        # print(noise[0])
        # print(N_0I[0] * new_radius[0])

        samples = torch.randn(samples.shape)
        samples_ = samples + noise
        # print(samples.shape)


        # breakpoint()
        # print(c)
        targets = torch.arange(50).cuda()
        # print(sample.shape, targets)

        outputs = self.model.heads[-1](samples_.cuda().float())


       

        # print(samples_dist.sample)
        # breakpoint()
        
        
        loss = nn.CrossEntropyLoss(None)(outputs['wsigma'], targets.long())
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def criterion(self, t, outputs, targets, ref_outputs=None, features=None, ref_features=None,stage2 = False,lwsoutputs = []):
        """Returns the loss value"""
        if ref_outputs is None or ref_features is None or stage2:
            # print('stage2')
            if type(outputs[0]) == dict:
                outputs = torch.cat([o['wsigma'] for o in outputs], dim=1)
            else:
                outputs = torch.cat(outputs, dim=1)
            # Eq. 1: regular cross entropy
            loss = nn.CrossEntropyLoss(None)(outputs, targets)
        else:
            if self.less_forget:
                # Eq. 6: Less-Forgetting constraint
                loss_dist = nn.CosineEmbeddingLoss()(features, ref_features.detach(),
                                                     torch.ones(targets.shape[0]).to(self.device)) * self.lamda
            else:
                # Scores before scale, [-1, 1]
                ref_outputs = torch.cat([ro['wosigma'] for ro in ref_outputs], dim=1).detach()
                old_scores = torch.cat([o['wosigma'] for o in outputs[:-1]], dim=1)
                num_old_classes = ref_outputs.shape[1]

                # Eq. 5: Modified distillation loss for cosine normalization
                loss_dist = nn.MSELoss()(old_scores, ref_outputs) * self.lamda * num_old_classes

            loss_mr = torch.zeros(1).to(self.device)
            if self.margin_ranking:
                # Scores before scale, [-1, 1]
                outputs_wos = torch.cat([o['wosigma'] for o in outputs], dim=1)
                num_old_classes = outputs_wos.shape[1] - outputs[-1]['wosigma'].shape[1]

                # Sec 3.4: "We select those new classes that yield highest responses to x (...)"
                # The index of hard samples, i.e., samples from old classes
                hard_index = targets < num_old_classes
                hard_num = hard_index.sum()

                if hard_num > 0:
                    # Get "ground truth" scores
                    gt_scores = outputs_wos.gather(1, targets.unsqueeze(1))[hard_index]
                    gt_scores = gt_scores.repeat(1, self.K)

                    # Get top-K scores on novel classes
                    max_novel_scores = outputs_wos[hard_index, num_old_classes:].topk(self.K, dim=1)[0]

                    assert (gt_scores.size() == max_novel_scores.size())
                    assert (gt_scores.size(0) == hard_num)
                    # Eq. 8: margin ranking loss
                    loss_mr = nn.MarginRankingLoss(margin=self.dist)(gt_scores.view(-1, 1),
                                                                     max_novel_scores.view(-1, 1),
                                                                     torch.ones(hard_num * self.K).to(self.device).view(-1,1))
                    loss_mr *= self.lamb_mr

            # Eq. 1: regular cross entropy
            loss_ce = nn.CrossEntropyLoss()(torch.cat([o['wsigma'] for o in outputs], dim=1), targets)
            # Eq. 9: integrated objective
            loss = loss_dist + loss_ce + loss_mr
        return loss

    @staticmethod
    def warmup_luci_loss(outputs, targets):
        if type(outputs) == dict:
            # needed during train
            return torch.nn.functional.cross_entropy(outputs['wosigma'], targets)
        else:
            # needed during eval()
            return torch.nn.functional.cross_entropy(outputs, targets)

# Sec 3.2: This class implements the cosine normalizing linear layer module using Eq. 4
class CosineLinear(Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)  # for initializaiton of sigma

    def forward(self, input, stochastic=None, cs_stoc=None):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out_s = self.sigma * out
        else:
            out_s = out
        if self.training:
            return {'wsigma': out_s, 'wosigma': out}
        else:
            return out_s


# This class implements a ResNet Basic Block without the final ReLu in the forward
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
