import torch
from algos.base_algo import Base_Algo

class DDRM(Base_Algo):
    def __init__(self, model, H_funcs, sigma_0, cls_fn=None, etaB=1, etaA=0.85, etaC=0.85):
        super().__init__(model, H_funcs, sigma_0, cls_fn)
        self.etaA = etaA
        self.etaB = etaB
        self.etaC = etaC

    @torch.no_grad()
    def get_pred_x(self, gt, y_0, at_next):
        if self.sigma_0 == 0:
            return gt
        else:
            H_funcs = self.H_funcs
            sigma_0 = self.sigma_0
            etaA = self.etaA
            etaB = self.etaB
            etaC = self.etaC
            x0_t = gt
            x = torch.randn_like(x0_t)
            singulars = H_funcs.singulars()
            Sigma = torch.zeros(x.shape[1]*x.shape[2]*x.shape[3], device=x.device)
            Sigma[:singulars.shape[0]] = singulars
            U_t_y = H_funcs.Ut(y_0)
            Sig_inv_U_t_y = U_t_y / singulars[:U_t_y.shape[-1]]

            #variational inference conditioned on y
            sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
            V_t_x0 = H_funcs.Vt(x0_t)
            SVt_x0 = (V_t_x0 * Sigma)[:, :U_t_y.shape[1]]

            falses = torch.zeros(V_t_x0.shape[1] - singulars.shape[0], dtype=torch.bool, device=x0_t.device)
            cond_before_lite = singulars * sigma_next > sigma_0
            cond_after_lite = singulars * sigma_next < sigma_0
            cond_before = torch.hstack((cond_before_lite, falses))
            cond_after = torch.hstack((cond_after_lite, falses))

            std_nextC = sigma_next * etaC
            sigma_tilde_nextC = torch.sqrt(sigma_next ** 2 - std_nextC ** 2)

            std_nextA = sigma_next * etaA
            sigma_tilde_nextA = torch.sqrt(sigma_next**2 - std_nextA**2)
            
            diff_sigma_t_nextB = torch.sqrt(sigma_next ** 2 - sigma_0 ** 2 / singulars[cond_before_lite] ** 2 * (etaB ** 2))

            #missing pixels
            Vt_x0_t_mod_next = V_t_x0
            #less noisy than y (after)
            Vt_x0_t_mod_next[:, cond_after] = V_t_x0[:, cond_after] + sigma_tilde_nextA * ((U_t_y - SVt_x0) / sigma_0)[:, cond_after_lite]            
            #noisier than y (before)
            Vt_x0_t_mod_next[:, cond_before] = Sig_inv_U_t_y[:, cond_before_lite] * etaB + (1 - etaB) * V_t_x0[:, cond_before]
            #aggregate all 3 cases and give next prediction
            x0_t = H_funcs.V(Vt_x0_t_mod_next).view(gt.shape)
            return x0_t

    @ torch.no_grad()
    def cal_x0(self, xt, t, at, at_next, y_0, classes=None):
        H_funcs = self.H_funcs
        sigma_0 = self.sigma_0
        etaA = self.etaA
        etaB = self.etaB
        etaC = self.etaC
        x = torch.randn_like(xt)
        singulars = H_funcs.singulars()
        Sigma = torch.zeros(x.shape[1]*x.shape[2]*x.shape[3], device=x.device)
        Sigma[:singulars.shape[0]] = singulars
        U_t_y = H_funcs.Ut(y_0)
        Sig_inv_U_t_y = U_t_y / singulars[:U_t_y.shape[-1]]

        
        #setup iteration variables
        # x = H_funcs.V(init_y.view(x.size(0), -1)).view(*x.size())
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

        #variational inference conditioned on y
        sigma = (1 - at).sqrt()[0, 0, 0, 0] / at.sqrt()[0, 0, 0, 0]
        sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
        xt_mod = xt / at.sqrt()[0, 0, 0, 0]
        V_t_x = H_funcs.Vt(xt_mod)
        SVt_x = (V_t_x * Sigma)[:, :U_t_y.shape[1]]
        V_t_x0 = H_funcs.Vt(x0_t)
        SVt_x0 = (V_t_x0 * Sigma)[:, :U_t_y.shape[1]]

        falses = torch.zeros(V_t_x0.shape[1] - singulars.shape[0], dtype=torch.bool, device=xt.device)
        cond_before_lite = singulars * sigma_next > sigma_0
        cond_after_lite = singulars * sigma_next < sigma_0
        cond_before = torch.hstack((cond_before_lite, falses))
        cond_after = torch.hstack((cond_after_lite, falses))

        std_nextC = sigma_next * etaC
        sigma_tilde_nextC = torch.sqrt(sigma_next ** 2 - std_nextC ** 2)

        std_nextA = sigma_next * etaA
        sigma_tilde_nextA = torch.sqrt(sigma_next**2 - std_nextA**2)
        
        diff_sigma_t_nextB = torch.sqrt(sigma_next ** 2 - sigma_0 ** 2 / singulars[cond_before_lite] ** 2 * (etaB ** 2))

        #missing pixels
        Vt_x0_t_mod_next = V_t_x0
        Vt_add_up_next = sigma_tilde_nextC * H_funcs.Vt(et) + std_nextC * torch.randn_like(V_t_x0)

        #less noisy than y (after)
        Vt_x0_t_mod_next[:, cond_after] = V_t_x0[:, cond_after] + sigma_tilde_nextA * ((U_t_y - SVt_x0) / sigma_0)[:, cond_after_lite]
        Vt_add_up_next[:, cond_after]= std_nextA * torch.randn_like(V_t_x0[:, cond_after])
        
        #noisier than y (before)
        Vt_x0_t_mod_next[:, cond_before] = Sig_inv_U_t_y[:, cond_before_lite] * etaB + (1 - etaB) * V_t_x0[:, cond_before]
        Vt_add_up_next[:, cond_before] = diff_sigma_t_nextB * torch.randn_like(U_t_y)[:, cond_before_lite]
        #aggregate all 3 cases and give next prediction
        x0_t = H_funcs.V(Vt_x0_t_mod_next).view(xt.shape[0], xt.shape[1], xt.shape[2], xt.shape[3])
        add_up = H_funcs.V(Vt_add_up_next).view(xt.shape[0], xt.shape[1], xt.shape[2], xt.shape[3]) * at_next.sqrt()
        return x0_t, add_up
    
    @torch.no_grad()
    def map_back(self, x0_t, y_0, add_up, at_next, at):
        # x0_hat = x0_t + self.H_funcs.H_pinv(y_0 - self.H_funcs.H(x0_t)).view(y_0.shape[0], 3, x0_t.shape[2], x0_t.shape[3])
        xt_next = at_next.sqrt() * x0_t + add_up
        return xt_next