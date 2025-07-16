import torch
from algos.base_algo import Base_Algo

class DDNM(Base_Algo):
    def __init__(self, model, H_funcs, sigma_0, cls_fn=None, eta=0.85):
        super().__init__(model, H_funcs, sigma_0, cls_fn)
        self.eta = eta

    @ torch.no_grad()
    def get_pred_x(self, gt, y_0, at_next):
        # return gt
        if self.sigma_0 == 0:
            return gt
        else:
            x0_t = gt
            x = torch.randn_like(x0_t)
            singulars = self.H_funcs.singulars()
            # print(singulars.shape)
            Sigma = torch.zeros(x.shape[1]*x.shape[2]*x.shape[3], device=x.device)
            Sigma[:singulars.shape[0]] = singulars
            Inv_Sigma = 1 / Sigma
            Inv_Sigma[Sigma==0] = 0
            U_t_y = self.H_funcs.Ut(y_0)
            Sigma = Sigma.view([1, x.shape[1], x.shape[2], x.shape[3]]).repeat(x.shape[0], 1, 1, 1)
            Inv_Sigma = Inv_Sigma.view([1, x.shape[1], x.shape[2], x.shape[3]]).repeat(x.shape[0], 1, 1, 1)
            V_t_x0_t = self.H_funcs.Vt(x0_t).view([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])
            
            lambda_t = torch.ones_like(V_t_x0_t)
            sigma_t = (1 - at_next[0,0,0,0]) ** 0.5
            change_idx = 1.0 * (sigma_t < at_next[0,0,0,0].sqrt()*self.sigma_0*Inv_Sigma)
            lambda_t = (1-change_idx) * lambda_t + change_idx * Sigma * sigma_t * (1-self.eta**2)**0.5/at_next[0,0,0,0].sqrt()/self.sigma_0
            x0_t = x0_t - self.H_funcs.V((lambda_t * self.H_funcs.Vt(self.H_funcs.H_pinv(self.H_funcs.H(x0_t) - y_0).view([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])).view([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])).view(x.shape[0], -1)).view([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])
            return x0_t


    @ torch.no_grad()
    def cal_x0(self, xt, t, at, at_next, y_0, classes=None):
        
        if self.sigma_0 == 0:
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
            x0_t = x0_t + self.H_funcs.H_pinv(y_0 - self.H_funcs.H(x0_t)).view(y_0.shape[0], 3, x0_t.shape[2], x0_t.shape[3])
            add_up = self.eta * (1-at_next).sqrt() * torch.randn_like(x0_t) + (1-self.eta**2)**0.5 * (1-at_next).sqrt() * et
        else:
            x = torch.randn_like(xt)
            singulars = self.H_funcs.singulars()
            # print(singulars.shape)
            Sigma = torch.zeros(x.shape[1]*x.shape[2]*x.shape[3], device=x.device)
            Sigma[:singulars.shape[0]] = singulars
            Inv_Sigma = 1 / Sigma
            Inv_Sigma[Sigma==0] = 0
            U_t_y = self.H_funcs.Ut(y_0)
            Sigma = Sigma.view([1, x.shape[1], x.shape[2], x.shape[3]]).repeat(x.shape[0], 1, 1, 1)
            Inv_Sigma = Inv_Sigma.view([1, x.shape[1], x.shape[2], x.shape[3]]).repeat(x.shape[0], 1, 1, 1)
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
            V_t_et = self.H_funcs.Vt(et).view([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])
            V_t_x0_t = self.H_funcs.Vt(x0_t).view([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])
            
            lambda_t = torch.ones_like(V_t_x0_t)
            sigma_t = (1 - at_next[0,0,0,0]) ** 0.5
            change_idx = 1.0 * (sigma_t < at_next[0,0,0,0].sqrt()*self.sigma_0*Inv_Sigma)
            lambda_t = (1-change_idx) * lambda_t + change_idx * Sigma * sigma_t * (1-self.eta**2)**0.5/at_next[0,0,0,0].sqrt()/self.sigma_0
            random_noise = torch.randn_like(V_t_x0_t)
            epsilon_tmp = torch.zeros_like(V_t_x0_t)
            change_idx = 1.0 * (sigma_t >= at_next[0,0,0,0].sqrt()*self.sigma_0*Inv_Sigma)
            epsilon_tmp = (1-change_idx) * epsilon_tmp + change_idx * (sigma_t**2-at_next[0,0,0,0]*self.sigma_0**2*Inv_Sigma**2) * random_noise
            change_idx = 1.0 * (sigma_t < at_next[0,0,0,0].sqrt()*self.sigma_0*Inv_Sigma)
            epsilon_tmp = (1-change_idx) * epsilon_tmp + change_idx * self.eta * sigma_t * random_noise
            change_idx = 1.0 * (Sigma==0)
            epsilon_tmp = (1-change_idx) * epsilon_tmp + change_idx * (sigma_t * (1-self.eta**2)**0.5 * V_t_et + sigma_t * self.eta * random_noise)
            # print(H_funcs.H(x0_t) - y_0)
            # print(H_funcs.H_pinv(H_funcs.H(x0_t) - y_0))
            x0_t = x0_t - self.H_funcs.V((lambda_t * self.H_funcs.Vt(self.H_funcs.H_pinv(self.H_funcs.H(x0_t) - y_0).view([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])).view([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])).view(x.shape[0], -1)).view([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])

            add_up = self.H_funcs.V(epsilon_tmp.view([epsilon_tmp.shape[0], -1])).view(x.shape)
        return x0_t, add_up
    
    @torch.no_grad()
    def map_back(self, x0_t, y_0, add_up, at_next, at):
        xt_next = at_next.sqrt() * x0_t + add_up
        return xt_next