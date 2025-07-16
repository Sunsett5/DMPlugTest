import torch
from algos.base_algo import Base_Algo
from torch.nn import Parameter
from optim.sf_adamw import AdamWScheduleFree
import tqdm

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

class DAPS(Base_Algo):
    def __init__(self, model, H_funcs, sigma_0, cls_fn=None, eta0=1e-4, delta=1e-2, order=5, nonlinear=False, betas = []):
        super().__init__(model, H_funcs, sigma_0, cls_fn)
        self.nonlinear = nonlinear
        self.eta0 = eta0
        self.delta = delta
        self.betas = torch.tensor(betas, dtype=torch.float32).cuda()
        self.order = order

    @ torch.no_grad()
    def ode(self, xt, t, classes=None):
        n = xt.shape[0]
        skip = t // (self.order - 1)
        if skip > 0:
            seq = range(0, t, skip)
        else:
            seq = [0]
        # print(list(seq))
        # print(seq)
        seq = list(seq)[1:] + [t]
        seq_next = [-1] + list(seq[:-1])
        b = self.betas
        for i, j in zip(reversed(seq), reversed(seq_next)):
            # steps.append(i)
            t = (torch.ones(n) * i).to(xt.device)
            next_t = (torch.ones(n) * j).to(xt.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            if self.cls_fn == None:
                et = self.model(xt, t)
            else:
                et = self.model(xt, t, self.classes)
                et = et[:, :3]
                et = et - (1 - at).sqrt()[0,0,0,0] * self.cls_fn(x,t,self.classes)
            if et.size(1) == 6:
                et = et[:, :3]
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_t = x0_t.clip(-1, 1)
            xt_next = at_next.sqrt() * x0_t + (1-at_next).sqrt() * et
            xt = xt_next
            # print(xt.norm())
        return xt
    @ torch.enable_grad()
    def langevin(self, x0, y_0, eta, at, N=100, nonlinear=False):
        rt = max((1-at[0,0,0,0]).sqrt(), 1e-4)
        # sigma_0 = self.sigma_0
        sigma_0 = 0.02
        x0_variable = x0.detach().clone().requires_grad_()
        for _ in range(N):
            # loss = torch.sum((x0_variable - x0)**2) / (2*rt**2) + torch.sum((self.H_funcs.H(x0_variable)-y_0)**2) / (2*sigma_0**2)
            # error = y_0 - self.H_funcs.H(x0_t)
            if self.sigma_0 == 0 and not nonlinear:
                loss = torch.sum((self.H_funcs.H(x0_variable)-y_0)**2)/eta/2
                # loss = torch.sum((x0_variable - x0)**2) / (2*rt**2) + torch.sum((self.H_funcs.H(x0_variable)-y_0)**2) / (2*sigma_0**2)
            # elif at[0,0,0,0] == 1:
            #     loss = torch.sum((x0_variable - x0)**2)/eta/2
            else:
                loss = torch.sum((x0_variable - x0)**2) / (2*rt**2) + torch.sum((self.H_funcs.H(x0_variable)-y_0)**2) / (2*sigma_0**2)
                # loss = torch.sum((x0_variable - x0)**2) * (2*sigma_0**2) / (2*rt**2) + torch.sum((self.H_funcs.H(x0_variable)-y_0)**2)
            grad = torch.autograd.grad(outputs=loss, inputs=x0_variable)[0]
            x0_variable = x0_variable - eta * grad + (2*eta)**0.5 * torch.randn_like(x0)
            # print(x0_variable.norm())
        return x0_variable.detach()

    @ torch.no_grad()
    def cal_x0(self, xt, t, at, at_next, y_0, T=1000, classes=None):
        # print(t)
        x0_t = self.ode(xt, int(t[0]), classes=classes)
        eta = self.eta0 * (self.delta + t[0]/T * (1-self.delta))
        # eta = at[0,0,0,0]
        x0_t_hat = self.langevin(x0_t, y_0, eta, at, nonlinear=self.nonlinear)
        x0_t = x0_t_hat
        add_up = (1-at_next).sqrt() * torch.randn_like(x0_t)
        return x0_t, add_up
    
    def map_back(self, x0_t, y_0, add_up, at_next, at):
        # x0_hat = x0_t + self.H_funcs.H_pinv(y_0 - self.H_funcs.forward(x0_t)).view(y_0.shape[0], 3, x0_t.shape[2], x0_t.shape[3])
        xt_next = at_next.sqrt() * x0_t + add_up
        return xt_next