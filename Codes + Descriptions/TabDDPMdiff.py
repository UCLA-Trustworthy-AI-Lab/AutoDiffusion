import functools
import torch
import torch.nn as nn
import numpy as np
import tqdm.notebook
import random
import math
import torch.nn as nn
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
from torch import Tensor

from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from scipy import integrate

device = 'cuda'  #@param ['cuda', 'cpu'] {'type':'string'}
torch.cuda.empty_cache()

###########################################################################################################################################
ModuleType = Union[str, Callable[..., nn.Module]]

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def _is_glu_activation(activation: ModuleType):
    return (
        isinstance(activation, str)
        and activation.endswith('GLU')
        or activation in [ReGLU, GEGLU]
    )


def _all_or_none(values):
    assert all(x is None for x in values) or all(x is not None for x in values)

def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)

class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)

def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    return (
        (
            ReGLU()
            if module_type == 'ReGLU'
            else GEGLU()
            if module_type == 'GEGLU'
            else getattr(nn, module_type)(*args)
        )
        if isinstance(module_type, str)
        else module_type(*args)
    )


class MLP(nn.Module):
    """The MLP model used in [gorishniy2021revisiting].

    The following scheme describes the architecture:

    .. code-block:: text

          MLP: (in) -> Block -> ... -> Block -> Linear -> (out)
        Block: (in) -> Linear -> Activation -> Dropout -> (out)

    Examples:
        .. testcode::

            x = torch.randn(4, 2)
            module = MLP.make_baseline(x.shape[1], [3, 5], 0.1, 1)
            assert module(x).shape == (len(x), 1)

    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    class Block(nn.Module):
        """The main building block of `MLP`."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            activation: ModuleType,
            dropout: float,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.activation(self.linear(x)))

    def __init__(
        self,
        *,
        d_in: int,
        d_layers: List[int],
        dropouts: Union[float, List[float]],
        activation: Union[str, Callable[[], nn.Module]],
        d_out: int,
    ) -> None:
        """
        Note:
            `make_baseline` is the recommended constructor.
        """
        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        assert len(d_layers) == len(dropouts)
        assert activation not in ['ReGLU', 'GEGLU']

        self.blocks = nn.ModuleList(
            [
                MLP.Block(
                    d_in=d_layers[i - 1] if i else d_in,
                    d_out=d,
                    bias=True,
                    activation=activation,
                    dropout=dropout,
                )
                for i, (d, dropout) in enumerate(zip(d_layers, dropouts))
            ]
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    @classmethod
    def make_baseline(
        cls: Type['MLP'],
        d_in: int,
        d_layers: List[int],
        dropout: float,
        d_out: int,
    ) -> 'MLP':
        """Create a "baseline" `MLP`.

        This variation of MLP was used in [gorishniy2021revisiting]. Features:

        * :code:`Activation` = :code:`ReLU`
        * all linear layers except for the first one and the last one are of the same dimension
        * the dropout rate is the same for all dropout layers

        Args:
            d_in: the input size
            d_layers: the dimensions of the linear layers. If there are more than two
                layers, then all of them except for the first and the last ones must
                have the same dimension. Valid examples: :code:`[]`, :code:`[8]`,
                :code:`[8, 16]`, :code:`[2, 2, 2, 2]`, :code:`[1, 2, 2, 4]`. Invalid
                example: :code:`[1, 2, 3, 4]`.
            dropout: the dropout rate for all hidden layers
            d_out: the output size
        Returns:
            MLP

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        assert isinstance(dropout, float)
        if len(d_layers) > 2:
            assert len(set(d_layers[1:-1])) == 1, (
                'if d_layers contains more than two elements, then'
                ' all elements except for the first and the last ones must be equal.'
            )
        return MLP(
            d_in=d_in,
            d_layers=d_layers,  # type: ignore
            dropouts=dropout,
            activation='ReLU',
            d_out=d_out,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x

###########################################################################################################################################

# f(x,t)
def drift_coeff(x, t, beta_1, beta_0):
   t = torch.tensor(t)
   beta_t = beta_0 + t * (beta_1 - beta_0)
   drift = -0.5 * beta_t * x
   return drift

# g(t)
def diffusion_coeff(t, beta_1, beta_0):
    t = torch.tensor(t)
    beta_t = beta_0 + t * (beta_1 - beta_0)
    diffusion = torch.sqrt(beta_t)
    return diffusion

def marginal_prob_mean(x, t, beta_1, beta_0):
  #x = x.to(device)
  t = torch.tensor(t)
  log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
  mean = torch.exp(log_mean_coeff)[:, None] * x
  return mean

def marginal_prob_std(t, beta_1, beta_0):
  t = torch.tensor(t)
  log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
  std = 1 - torch.exp(2. * log_mean_coeff)
  return torch.sqrt(std)


drift_coeff_fn = functools.partial(drift_coeff, beta_1=20, beta_0=0.1)
diffusion_coeff_fn = functools.partial(diffusion_coeff, beta_1=20, beta_0=0.1)
marginal_prob_mean_fn = functools.partial(marginal_prob_mean, beta_1=20, beta_0=0.1)
marginal_prob_std_fn = functools.partial(marginal_prob_std, beta_1=20, beta_0=0.1)

def min_max_scaling(factor, scale=(0, 1)):

  std = (factor - factor.min()) / (factor.max() - factor.min())
  new_min = torch.tensor(scale[0])
  new_max = torch.tensor(scale[1])
  return std * (new_max - new_min) + new_min


def compute_v(ll, alpha, beta):

    v = -torch.ones(ll.shape).to(ll.device)
    v[torch.gt(ll, beta)] = torch.tensor(0., device=v.device)
    v[torch.le(ll, alpha)] = torch.tensor(1., device=v.device)

    if ll[torch.eq(v, -1)].shape[0] !=0 and ll[torch.eq(v, -1)].shape[0] !=1 :
        v[torch.eq(v, -1)] = min_max_scaling(ll[torch.eq(v, -1)], scale=(1, 0)).to(v.device)
    else:
        v[torch.eq(v, -1)] = torch.tensor(0.5, device=v.device)
    return v


def loss_fn(model, Input_Data, T, eps=1e-5):
    N, input_dim = Input_Data.shape  
    loss_values = torch.empty(N)
    
    for row in range(N):
        random_t = torch.rand(T) * (1. - eps) + eps
        
        # Compute Perturbed data from SDE
        mean = marginal_prob_mean_fn(Input_Data[row,:], random_t).to(device)
        std = marginal_prob_std_fn(random_t).to(device)
        z = torch.randn(T, input_dim).to(device)
        perturbed_data = mean + z * std[:, None]
        
        score = model(perturbed_data, random_t).to(device)
        loss_row = torch.mean(torch.sum((score * std[:,None] + z)**2, dim=1))
        
        loss_values[row] = loss_row
    return loss_values.to(device)


class MLPDiffusion(nn.Module):
    def __init__(self, d_in, rtdl_params, dim_t = 128):
        super().__init__()
        self.dim_t = dim_t

        rtdl_params['d_in'] = dim_t
        rtdl_params['d_out'] = d_in

        self.mlp = MLP.make_baseline(**rtdl_params)
        
        self.proj = nn.Linear(d_in, dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
    
    def forward(self, x, timesteps):
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        x = self.proj(x) + emb
        return self.mlp(x)


def train_diffusion(latent_features, T, eps, sigma, lr, \
                    num_batches_per_epoch, maximum_learning_rate, weight_decay, n_epochs, batch_size):
    
    rtdl_params={
        'd_in': latent_features.shape[1],
        'd_layers': [256,256],
        'dropout': 0.0,
        'd_out': latent_features.shape[1],
    }
        
    ScoreNet = MLPDiffusion(latent_features.shape[1], rtdl_params)
    ScoreNet_Parallel = torch.nn.DataParallel(ScoreNet)
    ScoreNet_Parallel = ScoreNet_Parallel.to(device)

    optimizer = Adam(ScoreNet_Parallel.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=maximum_learning_rate,
        steps_per_epoch=num_batches_per_epoch,
        epochs=n_epochs,
    )

    tqdm_epoch = tqdm.notebook.trange(n_epochs)
    losses = []
    
    for epoch in tqdm_epoch:
      batch_idx = random.choices(range(latent_features.shape[0]), k=batch_size)  ## Choose random indices 
      batch_X = latent_features[batch_idx,:]  
      
      loss_values = loss_fn(ScoreNet_Parallel, batch_X, T, eps)
      loss = torch.mean(loss_values)
    
      optimizer.zero_grad()
      loss.backward() 
      optimizer.step()
      lr_scheduler.step()

      # Print the training loss over the epoch.
      losses.append(loss.item())
      tqdm_epoch.set_description('Average Loss: {:5f}'.format(loss.item()))
        
    return ScoreNet

def Euler_Maruyama_sampling(model, T, N, P, device):
    time_steps = torch.linspace(1., 1e-5, T) 
    step_size = time_steps[0] - time_steps[1] 

    Gen_data = torch.empty(N, P)

    init_x = torch.randn(N, P)
    X = init_x.to(device)
    
    tqdm_epoch = tqdm.notebook.trange(T)
    
    with torch.no_grad():
        for epoch in tqdm_epoch:
            time_step = time_steps[epoch].unsqueeze(0).to(device)

            # Predictor step (Euler-Maruyama)
            f = drift_coeff_fn(X, time_step).to(device)
            g = diffusion_coeff_fn(time_step).to(device)
            X = X - ( f - (g**2) * ( model(X, time_step) )  ) * step_size.to(device) + torch.sqrt(step_size).to(device)*g*torch.randn_like(X).to(device)
            tqdm_epoch.set_description('Diffusion Level: {:5f}'.format(epoch))

    Gen_data = X.cpu()
    
    return Gen_data