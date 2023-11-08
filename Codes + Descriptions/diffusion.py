import functools
import torch
import torch.nn as nn
import numpy as np
import tqdm.notebook
import random

from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from scipy import integrate

device = 'cuda'  #@param ['cuda', 'cpu'] {'type':'string'}
torch.cuda.empty_cache()

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


class ConcatSquash(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear_h = nn.Linear(input_dim, output_dim)
        self.hyper_bias = nn.Linear(1, output_dim, bias=False)
        self.hyper_gate = nn.Linear(1, output_dim, bias=False)

    def forward(self, x, t):
        return self.linear_h(x) * torch.sigmoid(self.hyper_bias(t.view(-1,1)) + self.hyper_gate(t.view(-1,1)))

class TabNetwork(nn.Module):
    def __init__(self, hidden_dims, converted_table_dim):
        super().__init__()
        
        self.hidden_dims = hidden_dims
        self.converted_table_dim = converted_table_dim
        
        modules = []                                                            
        dim = self.converted_table_dim                                              
        
        for item in list(self.hidden_dims):                                          
          modules.append(ConcatSquash(dim, dim + item))                           
          dim = dim + item                                                           
          modules.append(nn.ELU())                                             

        modules.append(nn.Linear(dim, self.converted_table_dim))                     
        self.all_modules = nn.ModuleList(modules)                              
        self.marginal_prob_std_fn = marginal_prob_std_fn

    def forward(self, x, t):
        modules = self.all_modules
        temb = x; time = t;
        m_idx = 0
        
        for _ in range(len(self.hidden_dims)):
            temb1 = modules[m_idx](x=temb, t=time)
            m_idx += 1
            temb = modules[m_idx](temb1)
            m_idx += 1
            
        h = modules[m_idx](temb)
        score_net = h/self.marginal_prob_std_fn(t)[:,None]
            
        return score_net
    
def train_diffusion(latent_features, T, hidden_dims, converted_table_dim, eps, sigma, lr, \
                    num_batches_per_epoch, maximum_learning_rate, weight_decay, n_epochs, batch_size):
    
    ScoreNet = TabNetwork(hidden_dims, converted_table_dim) # Stasy Architecture
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
    
    #alpha0 = 0.25
    #beta0 = 0.95
    
    for epoch in tqdm_epoch:
      batch_idx = random.choices(range(latent_features.shape[0]), k=batch_size)  ## Choose random indices 
      batch_X = latent_features[batch_idx,:]  
      
      loss_values = loss_fn(ScoreNet_Parallel, batch_X, T, eps)
      
      #q_alpha = torch.tensor(alpha0 + torch.log( torch.tensor(1+0.0001718*epoch* (1-alpha0), dtype=torch.float32))).clamp_(max=1).to(device)
      #q_beta = torch.tensor(beta0 + torch.log( torch.tensor(1+0.0001718*epoch* (1-beta0), dtype=torch.float32) )).clamp_(max=1).to(device)

      #alpha = torch.quantile(loss_values, q_alpha)
      #beta = torch.quantile(loss_values, q_beta)
      #assert alpha <= beta
      #v = compute_v(loss_values, alpha, beta)      
      
      #loss = torch.mean(v*loss_values)
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

###############################################################################################################################################

def prob_flow(score, T, N, P, device):
    init_x = torch.randn(N, P, device=device)
    
    time = np.linspace(1., 1e-5, T) 
    shape = init_x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        shape = sample.shape
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
        with torch.no_grad():
          score_wrap = score(sample, time_steps)
        return score_wrap.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        time_steps = np.ones((shape[0],)) * t

        t = torch.tensor(t).to(device)
        x = torch.tensor(x).to(device)
        f = drift_coeff_fn(x, t).cpu().numpy().reshape((-1,))
        g = diffusion_coeff_fn(t).cpu().numpy()
        
        x = x.reshape(shape)
        return  f - 0.5 * (g**2) * score_eval_wrapper(x, time_steps)

    res = integrate.solve_ivp(ode_func, (1., 1e-5), init_x.reshape(-1).cpu().numpy(), t_eval = time, rtol=1e-5, atol=1e-5, method='RK45')  
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)
    
    return x.float().cpu()

###############################################################################################################################################

def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))

def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))

def drift_fn(model, x, vec_t):
    N, P = x.shape
    
    mean = marginal_prob_mean_fn(x, vec_t).to(device)
    std = marginal_prob_std_fn(vec_t).to(device)
    z = torch.randn(N, P).to(device)
    perturbed_data = mean + z * std[:, None]
    score_eval = model(perturbed_data, vec_t).to(device)

    f = drift_coeff_fn(perturbed_data, vec_t)
    g = diffusion_coeff_fn(vec_t)
    drift_fn = f - 0.5 * (g**2) * score_eval
    
    return drift_fn

def get_div_fn(fn):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

  def div_fn(x, t, eps):

    grad_fn_eps_list = []
    for epsilon in eps:
      with torch.enable_grad():
        x.requires_grad_(True)
        fn_eps = torch.sum(fn(x, t) * epsilon)
        grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]

      x.requires_grad_(False)
      grad_fn_eps_list.append(torch.sum(grad_fn_eps * epsilon, dim=tuple(range(1, len(x.shape)))))

    return torch.mean(torch.stack(grad_fn_eps_list), 0)

  return div_fn

def div_fn(model, x, t, noise):
    return get_div_fn(lambda xx, tt: drift_fn(model, xx, tt))(x, t, noise)

def ode_func(t, x):
    sample = from_flattened_numpy(x[:-shape[0]], shape).to(device).type(torch.float32)
    vec_t = torch.ones(sample.shape[0], device=device) * t
    drift = to_flattened_numpy(drift_fn(score, sample, vec_t))
    logp_grad = to_flattened_numpy(div_fn(score, sample, vec_t, epsilon))
    return np.concatenate([drift, logp_grad], axis=0)

#init = np.concatenate([to_flattened_numpy(data), np.zeros((shape[0],))], axis = 0)
#res = integrate.solve_ivp(ode_func, (1e-5, 1.), init, rtol=1e-5, atol=1e-5, method='RK45')