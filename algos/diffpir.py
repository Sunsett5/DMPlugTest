import torch
from algos.base_algo import Base_Algo
from torch.nn import Parameter
from optim.sf_adamw import AdamWScheduleFree

class DiffPIR(Base_Algo):
    def __init__(self, model, H_funcs, sigma_0, cls_fn=None, lam=1.0, eta=0.85, lr=0.1):
        super().__init__(model, H_funcs, sigma_0, cls_fn)
        self.lam = lam
        self.eta = eta
        self.lr = lr

    @ torch.no_grad()
    def cal_x0(self, xt, t, at, at_next, y_0, classes=None):
        # xt.requires_grad_(True)
        if self.cls_fn == None:
            et = self.model(xt, t)
        else:
            et = self.model(xt, t, self.classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0,0,0,0] * self.cls_fn(x,t,self.classes)
        if et.size(1) == 6:
            et = et[:, :3]
        # et = et.clip(-1, 1)
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        x0_t = x0_t.clip(-1, 1)
        sigma_t_bar_square = (1-at[0,0,0,0])/at[0,0,0,0]
        sigma_t_bar_square = max(sigma_t_bar_square, 1e-8)
        # print(sigma_t_bar_square)
        rho_t = self.lam * self.sigma_0 ** 2 / sigma_t_bar_square
        # print(rho_t)
        x0_t_hat = x0_t.detach().clone().cpu()
        with torch.enable_grad():
            x0_t_hat = Parameter(x0_t_hat.cuda().requires_grad_())
            optimizer = AdamWScheduleFree([x0_t_hat], lr=self.lr, foreach=False)
            # optimizer = torch.optim.SGD([x0_t_hat], lr=0.01, momentum=0.9)
            for k in range(50):
                # print(k)
                optimizer.zero_grad()
                loss = torch.sum((self.H_funcs.H(x0_t_hat)-y_0)**2) + rho_t * torch.sum((x0_t_hat - x0_t)**2)
                loss.backward()
                optimizer.step()
        x0_t = x0_t_hat
        et_new = (xt) / (1-at).sqrt()
        add_up = (1-at_next).sqrt() * ((1 - self.eta**2)**0.5 * et_new + self.eta * torch.randn_like(x0_t))
        return x0_t, add_up
    
    def map_back(self, x0_t, y_0, add_up, at_next, at):
        # x0_hat = x0_t + self.H_funcs.H_pinv(y_0 - self.H_funcs.forward(x0_t)).view(y_0.shape[0], 3, x0_t.shape[2], x0_t.shape[3])
        xt_next = at_next.sqrt() * x0_t + add_up - at.sqrt() * x0_t / (1-at).sqrt() * (1-at_next).sqrt() * (1 - self.eta**2)**0.5
        return xt_next