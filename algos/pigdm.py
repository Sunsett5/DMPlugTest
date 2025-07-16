import torch
from algos.base_algo import Base_Algo

class PiGDM(Base_Algo):
    def __init__(self, model, H_funcs, sigma_0, cls_fn=None, eta=1.0, lam=1.0):
        super().__init__(model, H_funcs, sigma_0, cls_fn)  
        self.eta = eta
        self.lam = lam
    @ torch.enable_grad()
    def cal_x0(self, xt, t, at, at_next, y_0, classes=None):
        if self.sigma_0 == 0:
            xt.requires_grad_(True)
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
            # print(x0_t.norm())
            mat = self.H_funcs.H_pinv(y_0) - self.H_funcs.H_pinv(self.H_funcs.H(x0_t))
            mat = mat.view(xt.shape[0], xt.shape[1], xt.shape[2], xt.shape[3]).detach()
            loss = torch.sum(mat * x0_t)
            grad = torch.autograd.grad(outputs=loss, inputs=xt)[0]
            norm = torch.linalg.norm(grad)
            c1 = self.eta * ((1-at[0,0,0,0]/at_next[0,0,0,0]) * (1-at_next[0,0,0,0])/(1-at[0,0,0,0])).sqrt()
            c2 = (1-at_next[0,0,0,0] - c1**2).sqrt()
            vt = ((1-at_next[0,0,0,0]) / (1-at[0,0,0,0])) * (1 - at[0,0,0,0] / at_next[0,0,0,0])
            rt = (vt / (1+vt)).sqrt()
            # add_up = c1 * torch.randn_like(x0_t) + c2 * et + at.sqrt() * grad * 1.0
            add_up = c1 * torch.randn_like(x0_t) + c2 * et
            x0_t += at.sqrt() / at_next.sqrt() * grad * self.lam
        else:
            xt.requires_grad_(True)
            if self.cls_fn == None:
                et = self.model(xt, t)
            else:
                et = self.model(xt, t, classes)
                et = et[:, :3]
                et = et - (1 - at).sqrt()[0,0,0,0] * self.cls_fn(x,t,classes)
            if et.size(1) == 6:
                et = et[:, :3]
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_t = x0_t.clip(-1, 1)
            mat1 = self.H_funcs.Ut(y_0 - self.H_funcs.H(x0_t))
            # mat1 = H_funcs.H_pinv(y) - H_funcs.H_pinv(H_funcs.H(x0_t))
            mat1 = mat1.view(xt.shape[0], -1).detach()
            rt = (1-at[0,0,0,0]).sqrt()
            scale = self.sigma_0 / rt
            mat2 = self.H_funcs.H_scaled_inv(self.H_funcs.H(x0_t), scale).view(xt.shape[0], -1)
            loss = torch.sum(mat1 * mat2)
            grad = torch.autograd.grad(outputs=loss, inputs=xt)[0]
            norm = torch.linalg.norm(grad)
            c1 = self.eta * ((1-at[0,0,0,0]/at_next[0,0,0,0]) * (1-at_next[0,0,0,0])/(1-at[0,0,0,0])).sqrt()
            c2 = (1-at_next[0,0,0,0] - c1**2).sqrt()
            add_up = c1 * torch.randn_like(x0_t) + c2 * et
            x0_t += at.sqrt() / at_next.sqrt() * grad * self.lam
        return x0_t, add_up
    
    def map_back(self, x0_t, y_0, add_up, at_next, at):
        # x0_hat = x0_t + self.H_funcs.H_pinv(y_0 - self.H_funcs.forward(x0_t)).view(y_0.shape[0], 3, x0_t.shape[2], x0_t.shape[3])
        xt_next = at_next.sqrt() * x0_t + add_up
        return xt_next