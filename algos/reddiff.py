import torch
from algos.base_algo import Base_Algo


class RED_diff(Base_Algo):
    def __init__(self, model, H_funcs, sigma_0, cls_fn=None, eta=2.0):
        super().__init__(model, H_funcs, sigma_0, cls_fn)
        self.eta = eta
        # self.lam = lam

    @ torch.no_grad()
    def cal_x0(self, xt, x0_t_last, t, at, at_next, y_0, classes=None):
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
        if x0_t_last is None:
            x0_t_last = x0_t
        with torch.enable_grad():
            x0_t_last = x0_t_last.detach().clone().requires_grad_(True)
            # x0_t.requires_grad_(True)
            # print(x0_t.norm())
            # mat = self.H_funcs.H_pinv(y_0) - self.H_funcs.H_pinv(self.H_funcs.H(x0_t))
            # mat = mat.view(xt.shape[0], xt.shape[1], xt.shape[2], xt.shape[3]).detach()
            loss = torch.sum((y_0 - self.H_funcs.H(x0_t_last))**2)
            grad = torch.autograd.grad(outputs=loss, inputs=x0_t_last)[0]
        norm = torch.linalg.norm(grad)
        # add_up = c1 * torch.randn_like(x0_t) + c2 * et
        add_up = (1-at_next).sqrt() * torch.randn_like(x0_t)
        x0_t = x0_t_last + 1.0 * ((x0_t - x0_t_last) * 1.0 - grad * self.eta)
        # x0_t = x0_t.clip(-1, 1)
        return x0_t, add_up
    
    def map_back(self, x0_t, y_0, add_up, at_next, at):
        # x0_hat = x0_t + self.H_funcs.H_pinv(y_0 - self.H_funcs.forward(x0_t)).view(y_0.shape[0], 3, x0_t.shape[2], x0_t.shape[3])
        xt_next = at_next.sqrt() * x0_t + add_up
        return xt_next