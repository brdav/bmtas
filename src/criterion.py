import itertools
from collections import defaultdict
import torch
import torch.nn as nn

from src import utils


class WeightedSumLoss(nn.Module):
    """
    Overall multi-task loss, consisting of a weighted sum of individual task losses. With
    optional resource loss.
    """

    def __init__(self, tasks, resource_loss_weight=0, model=None):
        super().__init__()
        self.tasks = tasks
        self.loss_dict = nn.ModuleDict()
        for task in self.tasks:
            if task == 'semseg':
                self.loss_dict[task] = CrossEntropyLoss()
            elif task == 'human_parts':
                self.loss_dict[task] = CrossEntropyLoss()
            elif task == 'sal':
                self.loss_dict[task] = BalancedBinaryCrossEntropyLoss()
            elif task == 'normals':
                self.loss_dict[task] = L1Loss()
            elif task == 'edge':
                self.loss_dict[task] = BalancedBinaryCrossEntropyLoss(
                    pos_weight=0.95)
            else:
                raise ValueError
        self.loss_weights = {
            'semseg': 1.0,
            'human_parts': 2.0,
            'sal': 5.0,
            'normals': 10.0,
            'edge': 50.0,
        }
        if resource_loss_weight > 0:
            self.loss_dict['resource'] = BranchingLoss(model)
            self.loss_weights['resource'] = resource_loss_weight
            self.res_flag = True
        else:
            self.res_flag = False

    def forward(self, out, lab, task=None, omit_resource=False):
        losses = []
        if task:
            losses.append(
                self.loss_weights[task] * self.loss_dict[task](out[task], lab[task]))
        else:
            for t in self.tasks:
                losses.append(
                    self.loss_weights[t] * self.loss_dict[t](out[t], lab[t]))

        if self.res_flag and not omit_resource:
            losses.append(
                self.loss_weights['resource'] * self.loss_dict['resource']())

        tot_loss = sum(losses)
        return tot_loss


class CrossEntropyLoss(nn.Module):
    """
    Cross entropy loss with ignore regions.
    """

    def __init__(self, ignore_index=255):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index,
                                             reduction='mean')

    def forward(self, out, label):
        label = torch.squeeze(label, dim=1).long()
        loss = self.criterion(out, label)
        return loss


class BalancedBinaryCrossEntropyLoss(nn.Module):
    """
    Balanced binary cross entropy loss with ignore regions.
    """

    def __init__(self, pos_weight=None, ignore_index=255):
        super().__init__()
        self.pos_weight = pos_weight
        self.ignore_index = ignore_index

    def forward(self, output, label):
        mask = (label != self.ignore_index)
        masked_label = torch.masked_select(label, mask)
        masked_output = torch.masked_select(output, mask)

        # weighting of the loss, default is HED-style
        if self.pos_weight is None:
            num_labels_neg = torch.sum(1.0 - masked_label)
            num_total = torch.numel(masked_label)
            w = num_labels_neg / num_total
        else:
            w = torch.tensor(self.pos_weight, device=output.device)
        factor = 1. / (1 - w)

        loss = nn.functional.binary_cross_entropy_with_logits(
            masked_output,
            masked_label,
            pos_weight=w*factor,
            reduction='mean'
        )
        loss /= factor
        return loss


class L1Loss(nn.Module):
    """
    L1 loss with ignore regions.
    normalize: normalization for surface normals
    """

    def __init__(self, normalize=True, ignore_index=255):
        super().__init__()
        self.normalize = normalize
        self.criterion = nn.L1Loss(reduction='sum')
        self.ignore_index = ignore_index

    def forward(self, out, label):
        if self.normalize:
            out = utils.normalize_tensor(out, dim=1)

        mask = torch.prod((label != self.ignore_index),
                          dim=1, keepdim=True).bool()
        n_valid = torch.sum(mask).item()
        masked_out = torch.masked_select(out, mask)
        masked_label = torch.masked_select(label, mask)
        loss = self.criterion(masked_out, masked_label)
        loss /= max(n_valid, 1e-6)
        return loss


class BranchingLoss(nn.Module):
    """
    Proxyless resource loss for generating efficient branched networks. The loss is equal to
    the expected value of FLOPS of a branched architecture sampled from the supergraph. The
    FLOPS are calculated with the help of a look-up table.
    """

    def __init__(self, model):
        super().__init__()

        self.alphas = model.alphas
        self.n_tasks = len(model.tasks)

        my_partitions = [k for k in self.partition(
            list(range(self.n_tasks))) if not len(k) == self.n_tasks]
        self.total_l = len(model.encoder)  # number of layers in the encoder
        self.total_k = len(my_partitions) + 1  # the number of task groupings

        # we register some buffers that are needed for loss calculation
        self._initialize_weights(my_partitions, model)
        self._initialize_ancestors(my_partitions)
        self._initialize_indices(my_partitions)

    def forward(self):
        total_p = self.alphas[0].new_zeros((self.total_l, self.total_k))
        for l in range(self.total_l):
            # last dim indexes filter group
            sampling_prob = nn.functional.softmax(self.alphas[l], dim=-1)
            for k, (resind, parent) in enumerate(zip(self.resind_buffers(),
                                                     self.ancestor_buffers())):
                # conditional probability of task grouping
                p_cond = torch.sum(torch.prod(
                    torch.take(sampling_prob, resind), dim=1))
                if l > 0:
                    p = p_cond * torch.sum(total_p[l - 1, parent])
                else:
                    p = p_cond
                total_p[l, k] = p
            total_p[l, self.total_k - 1] = 1. - torch.sum(total_p[l])
        loss = torch.sum(total_p * self.weights)
        return loss

    def resind_buffers(self):
        for i in range(self.total_k - 1):
            yield getattr(self, 'resind_{}'.format(i))

    def ancestor_buffers(self):
        for i in range(self.total_k - 1):
            yield getattr(self, 'ancestors_{}'.format(i))

    def partition(self, input_set):
        # credit: https://stackoverflow.com/questions/19368375/set-partitions-in-python
        if len(input_set) == 1:
            yield [input_set]
            return
        first = input_set[0]
        for smaller in self.partition(input_set[1:]):
            # insert `first` in each of the subpartition's subsets
            for n, subset in enumerate(smaller):
                yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
            # put `first` in its own subset
            yield [[first]] + smaller

    def _initialize_weights(self, partitions, model):
        """ Initialize the resource weights for every possible task grouping / partition.
        The weight is equal to the number of branches in that layer (times the FLOPS per branch)
        considering that the tasks would be grouped accordingly. The weights are stored in a buffer.
        """
        flops = model.get_flops()
        gflops = torch.tensor(flops).view(-1, 1).repeat(1, self.total_k) / 1e9
        nr_parts = []
        for k in partitions:
            nr_parts.append(len(k))
        nr_parts.append(self.n_tasks)  # to account for task-specific case
        nr_parts = torch.tensor(nr_parts).repeat(self.total_l, 1)
        weights = gflops * nr_parts
        self.register_buffer('weights', weights)

    def _initialize_indices(self, partitions):
        """ For every possible partition, store the indices for sampling a
        flattened 'probs' matrix leading to that partition in a buffer.
        Example buffer: torch.tensor([[0, 4, 8, 12], [0, 5, 8, 12], [1, 6, 10, 14]])
        """
        for i, k in enumerate(partitions):
            ind_list = []
            for candidate in itertools.product(range(self.n_tasks), repeat=self.n_tasks):
                candidate_partition = defaultdict(list)
                for idx, element in enumerate(candidate):
                    candidate_partition[element].append(idx)
                if sorted(candidate_partition.values()) == sorted(k):
                    ind_list.append(
                        [v + u * self.n_tasks for u, v in enumerate(candidate)])
            self.register_buffer('resind_{}'.format(
                i), torch.tensor(ind_list, dtype=torch.long))

    def _initialize_ancestors(self, partitions):
        """ For every task partition k, find the corresponding ancestor task partitions.
        The ancestor task partitions are the set of partitions of which k is a refinement.
        Registers a buffer for every partition:
        Every buffer contains the indices to the ancestors of that particular partition.
        Example buffer: torch.tensor([0, 3, 4, 5, 11])
        """
        # there might be more efficient ways to do this, but we only need to do it upon init
        def is_refinement(partition_1, partition_2):
            # checks whether partition_1 is a refinement of partition_2
            for part_1 in partition_1:
                if any(set(part_1).issubset(set(part_2)) for part_2 in partition_2):
                    continue
                return False
            return True

        for i, k in enumerate(partitions):
            ancestors = []
            for candidate_idx, candidate in enumerate(partitions):
                if is_refinement(k, candidate):
                    ancestors.append(candidate_idx)
            self.register_buffer('ancestors_{}'.format(
                i), torch.tensor(ancestors, dtype=torch.long))
