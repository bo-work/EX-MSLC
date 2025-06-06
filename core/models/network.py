from .repr import Encoder
from .subnet import SubNet

import torch
import torch.nn as nn
from torch.autograd import Variable

class AdaptiveAutoEncoderClustering(nn.Module):
    def __init__(self, encoder_dims=(60, 40, 20), kernel_size=10, n_kernels=None, subnet_dims=[50, 30], input_shape=[83]):
        super(AdaptiveAutoEncoderClustering, self).__init__()

        assert n_kernels is not None, f"The number of kernels must be defined"
        assert n_kernels > 0, f"The number of kernels ({n_kernels}) must be greater than 0"
        assert kernel_size > 0, f"The dimensionality of kernels ({n_kernels}) must be greater than 0"

        self.input_shape = input_shape if isinstance(input_shape, list) else [input_shape]
        self.kernel_size = kernel_size
        self.n_kernels = n_kernels
        self.subnet_dims = subnet_dims
        self.encoder_dims = self.input_shape + list(encoder_dims) + [kernel_size]
        encoder_dims.reverse()
        self.decoder_dims = [kernel_size] + encoder_dims + self.input_shape
        self.labels_ = None


        self.sub_nets = nn.ParameterList([])
        self.sub_nets_list = []

        self.add_sub_net()

        for i in range(len(self.sub_nets_list), self.n_kernels):
            self.add_sub_net()

    def add_sub_net(self):
        sub_net = SubNet(self.encoder_dims[-1], self.subnet_dims)
        sub_net.encoder = Encoder(layers_dims=self.encoder_dims)
        sub_net.decoder = Encoder(layers_dims=self.decoder_dims)
        self.sub_nets_list.append(sub_net)
        self.sub_nets = nn.ModuleList(self.sub_nets_list)

    def remove_sub_net(self, idx):
        self.sub_nets_list.pop(idx)
        del self.sub_nets[idx]
        self.sub_nets = nn.ModuleList(self.sub_nets_list)

    def reset(self) -> None:
        pass

    @property
    def n_kernels_(self):
        return len(self.sub_nets)

    def loss(self, labels = None):
        if self.training:
            assert labels is not None, "True labels must be provided during training."
        self.labels_ = labels

        classification_loss = 0
        rebuilding_loss = 0
        clustering_loss_close = 0
        clustering_loss_dist = 0
        clustering_loss = 0
        labels = self.labels_    #(bs)

        for i in range(self.n_kernels_):
            idx = (labels == i).nonzero()
            idx_diff = (labels != i).nonzero()

            embeddings = self.sub_nets[i].encoder.outputs
            rebuildings = self.sub_nets[i].decoder.outputs
            outputs = self.sub_nets[i].outputs

            targets = torch.zeros_like(outputs)
            targets[idx] = 1.
            classification_loss += ((outputs - targets) ** 2).mean()
            rebuilding_loss += (torch.mean((rebuildings - self.input_) ** 2, dim=1) * targets.squeeze()).mean()

            if len(idx) > 0:
                clustering_loss_close += ((embeddings[idx].view(-1, self.kernel_size) -
                                           self.sub_nets[i].kernel_weights) ** 2).mean()
                if len(idx_diff) > 0:
                    for j in list(set(labels[idx_diff])):
                        tmp_loss = ((self.sub_nets[i].kernel_weights.clone().detach() -
                                     self.sub_nets[j].kernel_weights) ** 2).mean()
                        clustering_loss_dist += torch.clamp(1 - tmp_loss, min=0)

        n_k_dist = (self.n_kernels_ - 1 if self.n_kernels_ > 1 else 1) * self.n_kernels_
        clustering_loss = (clustering_loss_close + clustering_loss_dist/n_k_dist) / self.n_kernels_
        return classification_loss + clustering_loss + 10 * rebuilding_loss, classification_loss*len(outputs), clustering_loss*len(outputs), rebuilding_loss*len(outputs)

    def WCEloss(self, wce=[]):
        classification_loss = 0
        rebuilding_loss = 0
        clustering_loss_close = 0
        clustering_loss_dist = 0
        clustering_loss = 0
        labels = self.labels_  # (bs)

        for i in range(self.n_kernels_):
            idx = (labels == i).nonzero()
            idx_diff = (labels != i).nonzero()

            embeddings = self.sub_nets[i].encoder.outputs
            rebuildings = self.sub_nets[i].decoder.outputs
            outputs = self.sub_nets[i].outputs

            targets = torch.zeros_like(outputs)
            targets[idx] = 1.
            classification_loss += wce[i] * ((outputs - targets) ** 2).mean()
            rebuilding_loss += wce[i] * (torch.mean((rebuildings - self.input_) ** 2, dim=1) * targets.squeeze()).mean()

            if len(idx) > 0:
                clustering_loss_close += ((embeddings[idx].view(-1, self.kernel_size) -
                                           self.sub_nets[i].kernel_weights) ** 2).mean()
                if len(idx_diff) > 0:
                    for j in list(set(labels[idx_diff])):
                        tmp_loss = ((self.sub_nets[i].kernel_weights.clone().detach() -
                                     self.sub_nets[j].kernel_weights) ** 2).mean()
                        clustering_loss_dist += torch.clamp(1 - tmp_loss, min=0)

        n_k_dist = (self.n_kernels_ - 1 if self.n_kernels_ > 1 else 1) * self.n_kernels_
        clustering_loss = wce[i] * (clustering_loss_close + clustering_loss_dist / n_k_dist) / self.n_kernels_
        return classification_loss + clustering_loss + 10 * rebuilding_loss, classification_loss * len(
            outputs), clustering_loss * len(outputs), rebuilding_loss * len(outputs)

    def backward(self, losses: list):
        for loss in losses:
            loss.backward(retain_graph=True)

    # outputs size: [bs, n_kernels]
    def forward(self, x, labels=None):
        if self.training:
            assert labels is not None, "True labels must be provided during training."
        self.labels_ = labels
        self.input_ = x

        outputs = []
        for i in range(self.n_kernels_):
            sub_net = self.sub_nets[i]
            embedding = sub_net.encoder(x)
            rebuilding = sub_net.decoder(embedding)
            output_probs = sub_net(embedding)
            outputs.append(output_probs.view(embedding.shape[0], 1, -1))

        outputs = torch.cat(tuple(outputs), 1)   # (bs, 6, 1)
        outputs = torch.squeeze(outputs)  # (bs, 6)
        # outputs = torch.softmax(outputs, dim=1)

        return outputs
