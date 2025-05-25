import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.nn.init as init


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConvTranspose2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.ConvTranspose2d(*args, **kwargs)

        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size)
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class BasicBlock(MetaModule):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = MetaConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                MetaConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                MetaBatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(MetaModule):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = MetaConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)
        self.conv3 = MetaConv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = MetaBatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                MetaConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                MetaBatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(MetaModule):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = MetaConv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = MetaBatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = MetaLinear(512*block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNet_mlcmslc(MetaModule):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet_mlcmslc, self).__init__()
        self.in_planes = 64
        self.conv1 = MetaConv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = MetaBatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = MetaLinear(512*block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_h=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        hidden = out.view(out.size(0), -1)
        out = self.linear(hidden)
        if return_h:
            return out, hidden
        else:
            return out


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class ResNet_mlc(MetaModule):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_mlc, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, return_h=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        hidden = out.view(out.size(0), -1)
        out = self.linear(hidden)
        if return_h:
            return out, hidden
        else:
            return out

def ResNet18(num_classes):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

def ResNet34(num_classes):
    return ResNet(BasicBlock, [3,4,6,3], num_classes)
def ResNet34mlc(num_classes):
    return ResNet_mlcmslc(BasicBlock, [3,4,6,3], num_classes)
def ResNet32(num_classes):
    return ResNet_mlc(BasicBlock, [5, 5, 5], num_classes)

def ResNet50(num_classes):
    return ResNet(Bottleneck, [3,4,6,3], num_classes)

def ResNet101(num_classes):
    return ResNet(Bottleneck, [3,4,23,3], num_classes)

def ResNet152(num_classes):
    return ResNet(Bottleneck, [3,8,36,3], num_classes)

class VNet(MetaModule):
    def __init__(self, input, hidden, output):
        super(VNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden, output)



    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return F.sigmoid(out)


class SLCorNet(MetaModule):
    def __init__(self, input, input2, num_classes, output=1, output2=1, hidden1=100, hidden2=100):
        super(SLCorNet, self).__init__()
        self.vnet1 = VNet(input, hidden1, output)
        self.vnet2 = VNet(input2, hidden2, output2)

        self.num_classes = num_classes

    def loss(self):
        l1 = torch.sum(self.cost2_v * self.l_lambda) / len(self.cost2_v)
        l2 = torch.sum(self.cost3_v * (self.l_beta) * (1 - self.l_lambda)) / len(self.cost3_v) + torch.sum(
            self.cost4_v * (1 - self.l_beta) * (1 - self.l_lambda)) / (len(self.cost4_v))
        l_f_meta = l1 + l2

        return l_f_meta

    def forward(self, targets, y_f_hat, soft_labels_1):
        target_var = to_var(targets, requires_grad=False).long()

        cost2 = F.cross_entropy(y_f_hat, target_var, reduce=False)
        cost2_v = torch.reshape(cost2, (len(cost2), 1))
        self.cost2_v = cost2_v
        l_lambda = self.vnet1(cost2_v.data)
        self.l_lambda = l_lambda

        z = torch.max(soft_labels_1, dim=1)[1].long().cuda()
        cost3 = F.cross_entropy(y_f_hat, z, reduce=False)
        cost3_v = torch.reshape(cost3, (len(cost3), 1))
        self.cost3_v = cost3_v
        l_beta = self.vnet2(cost3_v.data)
        self.l_beta = l_beta

        y_f_hat_n = torch.max(y_f_hat, dim=1)[1].long().cuda()
        cost4 = F.cross_entropy(y_f_hat, y_f_hat_n, reduce=False)
        cost4_v = torch.reshape(cost4, (len(cost4), 1))
        self.cost4_v = cost4_v

        y_g_wide2 = ((1 - l_beta)  * F.softmax(y_f_hat,dim=1)) + (l_beta * soft_labels_1.float().cuda())
        target_var_nc = torch.zeros(targets.size()[0], self.num_classes).scatter_(1, targets.view(-1, 1), 1)
        y_g_wide = l_lambda.cuda() * target_var_nc.cuda() + y_g_wide2.cuda() * (1 - l_lambda.cuda())

        return y_g_wide


class enSLCorNet1(MetaModule):
    def __init__(self, input, input2, num_classes, output=1, output2=1, hidden1=100, hidden2=100):
        super(SLCorNet, self).__init__()
        self.vnet1 = VNet(input, hidden1, output)
        self.vnet2 = VNet(input2, hidden2, output2)

        self.num_classes = num_classes

    def loss(self):
        l1 = torch.sum(self.cost2_v * self.l_lambda) / len(self.cost2_v)
        l2 = torch.sum(self.cost3_v * (self.l_beta) * (1 - self.l_lambda)) / len(self.cost3_v) + torch.sum(
            self.cost4_v * (1 - self.l_beta) * (1 - self.l_lambda)) / (len(self.cost4_v))
        l_f_meta = l1 + l2

        return l_f_meta

    def forward(self, targets, y_f_hat, soft_labels_1):
        target_var = to_var(targets, requires_grad=False).long()

        cost2 = F.cross_entropy(y_f_hat, target_var, reduce=False)
        cost2_v = torch.reshape(cost2, (len(cost2), 1))
        self.cost2_v = cost2_v
        l_lambda = self.vnet1(cost2_v.data)
        self.l_lambda = l_lambda

        z = torch.max(soft_labels_1, dim=1)[1].long().cuda()
        cost3 = F.cross_entropy(y_f_hat, z, reduce=False)
        cost3_v = torch.reshape(cost3, (len(cost3), 1))
        self.cost3_v = cost3_v
        l_beta = self.vnet2(cost3_v.data)
        self.l_beta = l_beta

        y_f_hat_n = torch.max(y_f_hat, dim=1)[1].long().cuda()
        cost4 = F.cross_entropy(y_f_hat, y_f_hat_n, reduce=False)
        cost4_v = torch.reshape(cost4, (len(cost4), 1))
        self.cost4_v = cost4_v

        y_g_wide2 = ((1 - l_beta)  * F.softmax(y_f_hat,dim=1)) + (l_beta * soft_labels_1.float().cuda())
        target_var_nc = torch.zeros(targets.size()[0], self.num_classes).scatter_(1, targets.view(-1, 1), 1)
        y_g_wide = l_lambda.cuda() * target_var_nc.cuda() + y_g_wide2.cuda() * (1 - l_lambda.cuda())

        return y_g_wide


class LCorNet(nn.Module):
    def __init__(self, hx_dim, cls_dim, h_dim, num_classes):
        super().__init__()

        self.num_classes = num_classes
        self.in_class = self.num_classes
        self.hdim = h_dim
        self.cls_emb = nn.Embedding(self.in_class, cls_dim)

        in_dim = hx_dim + cls_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hdim),
            nn.Tanh(),
            nn.Linear(self.hdim, self.hdim),
            nn.Tanh(),
            nn.Linear(self.hdim, num_classes + int(False), bias=(not False))
        )

        # if self.args.sparsemax:
        #     from sparsemax import Sparsemax
        #     self.sparsemax = Sparsemax(-1)

        self.init_weights()

        # if self.args.tie:
        #     print('Tying cls emb to output cls weight')
        #     self.net[-1].weight = self.cls_emb.weight

    def init_weights(self):
        nn.init.xavier_uniform_(self.cls_emb.weight)
        nn.init.xavier_normal_(self.net[0].weight)
        nn.init.xavier_normal_(self.net[2].weight)
        nn.init.xavier_normal_(self.net[4].weight)

        self.net[0].bias.data.zero_()
        self.net[2].bias.data.zero_()

        if not False:
            assert self.in_class == self.num_classes, 'In and out classes conflict!'
            self.net[4].bias.data.zero_()

    def get_alpha(self):
        return self.alpha if False else torch.zeros(1)

    def soft_cross_entropy(self, logit, pseudo_target, reduction='mean'):
        loss = -(pseudo_target * F.log_softmax(logit, -1)).sum(-1)
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'none':
            return loss
        elif reduction == 'sum':
            return loss.sum()
        else:
            raise NotImplementedError('Invalid reduction: %s' % reduction)

    def forward(self, hx, y):
        bs = hx.size(0)

        y_emb = self.cls_emb(y)
        hin = torch.cat([hx, y_emb], dim=-1)

        logit = self.net(hin)

        out = F.softmax(logit, -1)


        return out







