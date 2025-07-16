import torch
from algos.base_algo import Base_Algo
from torch.nn import Parameter
from optim.sf_adamw import AdamWScheduleFree
# from torch.optim import 

class ReSample(Base_Algo):
    def __init__(self, model, H_funcs, sigma_0, cls_fn=None, gamma=100.0, eta=0.85):
        super().__init__(model, H_funcs, sigma_0, cls_fn)
        self.gamma = gamma
        self.eta = eta

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
        x0_t_hat = x0_t.detach().clone().cpu()
        with torch.enable_grad():
            x0_t_hat = Parameter(x0_t_hat.cuda().requires_grad_())
            # optimizer = AdamWScheduleFree([x0_t_hat], lr=0.01, foreach=False)
            optimizer = torch.optim.SGD([x0_t_hat], lr=0.01, momentum=0.9)
            for k in range(50):
                #print(k)
                optimizer.zero_grad()
                loss = torch.sum((self.H_funcs.H(x0_t_hat)-y_0)**2)
                loss.backward()
                optimizer.step()
        # x0_t = x0_t_hat
        c1 = (1 - at_next).sqrt() * self.eta
        c2 = (1 - at_next).sqrt() * ((1 - self.eta ** 2) ** 0.5)
        xt_next_prime = at_next.sqrt() * x0_t + c1 * torch.randn_like(xt) + c2 * et
        sigma_t_square = self.gamma * (1-at_next[0,0,0,0])/at[0,0,0,0] * (1-at[0,0,0,0]/at_next[0,0,0,0])
        if sigma_t_square == 0:
            var = torch.tensor(0)
            mean = xt_next_prime
        else:
            var = sigma_t_square * (1-at_next[0,0,0,0]) / (sigma_t_square + 1 - at_next[0,0,0,0])
            mean = ((1-at_next) * xt_next_prime) / (sigma_t_square + 1 - at_next[0,0,0,0])
        # print(var)
        # xt_next = mean + var.sqrt() * torch.randn_like(xt)
        add_up = mean + var.sqrt() * torch.randn_like(xt)
        return x0_t_hat, add_up
    
    def map_back(self, x0_t, y_0, add_up, at_next, at):
        # x0_hat = x0_t + self.H_funcs.H_pinv(y_0 - self.H_funcs.forward(x0_t)).view(y_0.shape[0], 3, x0_t.shape[2], x0_t.shape[3])
        sigma_t_square = self.gamma * (1-at_next[0,0,0,0])/at[0,0,0,0] * (1-at[0,0,0,0]/at_next[0,0,0,0])
        if sigma_t_square == 0:
            xt_next = x0_t
        else:
            xt_next = sigma_t_square * at_next.sqrt() * x0_t / (sigma_t_square + 1 - at_next[0,0,0,0]) + add_up
        return xt_next