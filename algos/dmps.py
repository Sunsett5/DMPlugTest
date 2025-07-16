import torch
from algos.base_algo import Base_Algo

class DMPS(Base_Algo):
    def __init__(self, model, H_funcs, sigma_0, cls_fn=None, eta=0.85):
        super().__init__(model, H_funcs, sigma_0, cls_fn)
        self.eta = eta

    @ torch.no_grad()
    def cal_x0(self, xt, t, at, at_next, y_0, classes=None):
        # print(guidance)
        guidance = self.H_funcs.H_dmps_guidance(xt, y_0, at[0,0,0,0], self.sigma_0).view(xt.shape[0], xt.shape[1], xt.shape[2], xt.shape[3])
        random_noise = torch.randn_like(xt)
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
        # c1 = self.eta * ((1-at[0,0,0,0]/at_next[0,0,0,0]) * (1-at_next[0,0,0,0])/(1-at[0,0,0,0])).sqrt()
        # c2 = (1-at_next[0,0,0,0] - c1**2).sqrt()
        c1 = self.eta * (1-at_next[0,0,0,0]).sqrt()
        c2 = (1-self.eta**2)**0.5 * (1-at_next[0,0,0,0]).sqrt()
        at_no_bar = at[0,0,0,0]/at_next[0,0,0,0]
        # x_t_temp = at_next.sqrt() * x0_t + c1 * random_noise + c2 * et
        # guidance = self.H_funcs.H_dmps_guidance(x_t_temp, y_0, at[0,0,0,0], self.sigma_0).view(xt.shape[0], xt.shape[1], xt.shape[2], xt.shape[3])
        x0_t += (1-at_no_bar)/(at_no_bar.sqrt() * at_next.sqrt()) * guidance * 1.0
        # x0_t += (1-at).sqrt()/at_next.sqrt() * guidance * 1.0
        add_up = c1 * random_noise + c2 * et
        return x0_t, add_up
    
    @torch.no_grad()
    def map_back(self, x0_t, y_0, add_up, at_next, at):
        # x0_hat = x0_t + self.H_funcs.H_pinv(y_0 - self.H_funcs.H(x0_t)).view(y_0.shape[0], 3, x0_t.shape[2], x0_t.shape[3])
        xt_next = at_next.sqrt() * x0_t + add_up
        return xt_next