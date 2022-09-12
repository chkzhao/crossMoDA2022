import torch

from torch import nn
from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class ccbn(nn.Module):
  def __init__(self, num_features, num_classes=2):
    super().__init__()
    self.num_features = num_features
    self.embed = nn.Embedding(num_classes, num_features * 2)
    self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
    self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
  def forward(self, x, y, alpha=None):
    if alpha is not None:
        gamma0,beta0 = self.embed(torch.zeros(y.shape).cuda().long()).chunk(2, 1)
        gamma1, beta1 = self.embed(torch.ones(y.shape).cuda().long()).chunk(2, 1)
        gamma = gamma0*alpha.squeeze(dim=2).squeeze(dim=2) + gamma1*(1-alpha.squeeze(dim=2).squeeze(dim=2))
        beta = beta0*alpha.squeeze(dim=2).squeeze(dim=2) + beta1*(1-alpha.squeeze(dim=2).squeeze(dim=2))
    else:
        gamma, beta = self.embed(y).chunk(2, 1)
    out = x
    out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
    return out