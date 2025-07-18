import torch
from algos.base_algo import Base_Algo

class DPS(Base_Algo):
    def __init__(self, model, H_funcs, sigma_0, cls_fn=None, eta=1.0, lam=1.0):
        super().__init__(model, H_funcs, sigma_0, cls_fn)
        self.eta = eta
        self.lam = lam

    @ torch.enable_grad()
    def cal_x0(self, xt, t, at, at_next, y_0, noise='ddpm', classes=None):
        xt.requires_grad_(True)
        if self.cls_fn == None:
            et = self.model(xt, t)
        else:
            et = self.model(xt, t, self.classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0,0,0,0] * self.cls_fn(xt,t,self.classes)
        if et.size(1) == 6:
            et = et[:, :3]
        # et = et.clip(-1, 1)
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        x0_t = x0_t.clip(-1, 1)
        # print(x0_t.norm())
        # mat = self.H_funcs.H_pinv(y_0) - self.H_funcs.H_pinv(self.H_funcs.H(x0_t))
        # mat = mat.view(xt.shape[0], xt.shape[1], xt.shape[2], xt.shape[3]).detach()
        error = y_0 - self.H_funcs.H(x0_t)
        loss = torch.sum(error**2)
        grad = torch.autograd.grad(outputs=loss, inputs=xt)[0]
        norm = torch.linalg.norm(grad)
        if noise == 'ddpm':
            c1 = self.eta * ((1-at[0,0,0,0]/at_next[0,0,0,0]) * (1-at_next[0,0,0,0])/(1-at[0,0,0,0])).sqrt()
        elif noise == 'ddim':
            c1 = 0
        else:
            raise ValueError("Unsupported noise type: {}".format(noise))
        c2 = (1-at_next[0,0,0,0] - c1**2).sqrt()
        vt = ((1-at_next[0,0,0,0]) / (1-at[0,0,0,0])) * (1 - at[0,0,0,0] / at_next[0,0,0,0])
        rt = (vt / (1+vt)).sqrt()
        # add_up = c1 * torch.randn_like(x0_t) + c2 * et + at.sqrt() * grad * 1.0
        add_up = c1 * torch.randn_like(x0_t) + c2 * et
        x0_t -= 1 / at_next.sqrt() * grad * self.lam / loss.sqrt()
        return x0_t, add_up
    
    def map_back(self, x0_t, y_0, add_up, at_next, at):
        # x0_hat = x0_t + self.H_funcs.H_pinv(y_0 - self.H_funcs.forward(x0_t)).view(y_0.shape[0], 3, x0_t.shape[2], x0_t.shape[3])
        xt_next = at_next.sqrt() * x0_t + add_up
        return xt_next