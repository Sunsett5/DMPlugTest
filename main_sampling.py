import random
import argparse, os, yaml
import torch
import torchvision.utils as tvu
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from util.img_utils import Blurkernel, clear_color, generate_tilt_map, mask_generator
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from ddim_sampler import *
import shutil
import lpips
from tqdm import tqdm

from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download
from models.diffusion import Model
from guided_diffusion.unet_ffhq import create_model as create_model_ffhq
from algos.ddnm import DDNM
from algos.pigdm import PiGDM
from algos.ddrm import DDRM
from algos.dps import DPS
from algos.reddiff import RED_diff
from algos.diffpir import DiffPIR
from algos.dmps import DMPS
from algos.resample import ReSample
from algos.daps import DAPS
from algos.dmplug import DMPlug

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def init_model(opt, config, model_config, device):
    if config.model_type == 'simple':    
        model = Model(config)
        # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
        if config.data.dataset == "CIFAR10":
            name = "cifar10"
        elif config.data.dataset == "LSUN":
            name = f"lsun_{config.data.category}"
        elif config.data.dataset == 'CelebA_HQ':
            name = 'celeba_hq'
        else:
            raise ValueError
        if name != 'celeba_hq':
            ckpt = get_ckpt_path(f"ema_{name}", prefix=opt.exp)
            print("Loading checkpoint {}".format(ckpt))
        elif name == 'celeba_hq':
            #ckpt = '~/.cache/diffusion_models_converted/celeba_hq.ckpt'
            ckpt = 'models/celeba_hq.ckpt'
            if not os.path.exists(ckpt):
                download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt', ckpt)
        else:
            raise ValueError
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device)
        model = torch.nn.DataParallel(model)

    elif config.model_type == 'openai':
        raise NotImplementedError("OpenAI model is not implemented yet (need to change path to the model)")
        config_dict = vars(config.model)
        model = create_model(**config_dict)
        if config.model.use_fp16:
            model.convert_to_fp16()
        if config.model.class_cond:
            ckpt = os.path.join(opt.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (config.data.image_size, config.data.image_size))
            if not os.path.exists(ckpt):
                download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (config.data.image_size, config.data.image_size), ckpt)
        else:
            ckpt = os.path.join(opt.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
            if not os.path.exists(ckpt):
                download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt', ckpt)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device)
        model.eval()
        model = torch.nn.DataParallel(model)

    elif config.model_type == 'ffhq':
        model = create_model_ffhq(**model_config)
        model = model.to(device)
        model.eval()

    return model


def init_algo(opt, model, H_funcs=None, sigma_0=0.01, deg=None, betas=None):
    if opt.algo == 'ddnm':
        algo = DDNM(model, H_funcs, sigma_0)
    elif opt.algo == 'pigdm':
        if 'celeba' in opt.config:
            lam = 1.0
        else:
            lam = 1.0
        algo = PiGDM(model, H_funcs, sigma_0, lam=lam)
    elif opt.algo == 'ddrm':
        algo = DDRM(model, H_funcs, sigma_0)
    elif opt.algo == 'dps':
        if 'celeba' in opt.config:
            if 'cs' in deg:
                lam = 1.0
            elif deg == 'deblur_nonlinear':
                lam = 1.0
            elif 'hdr' in deg:
                lam = 1.0
            elif 'phase' in deg:
                lam = 0.4
            elif 'deblur_aniso' in deg:
                lam = 1.0
            elif 'box' in deg:
                lam = 1.0
            elif 'sr4' in deg:
                lam = 1.0
            else:
                lam = 1.0
        elif 'ffhq' in opt.config:
            if 'cs' in deg:
                lam = 1.0
            elif deg == 'deblur_nonlinear':
                lam = 1.0
            elif 'deblur_aniso' in deg:
                lam = 1.0
            elif 'inpainting' in deg:
                lam = 1.0
            elif 'sr4' in deg:
                lam = 1.0
            else:
                lam = 1.0
        else:
            lam = 1.0
        algo = DPS(model, H_funcs, sigma_0, lam=lam)
    elif opt.algo == 'reddiff':
        if 'celeba' in opt.config:
            if 'inp' in deg:
                if 'box' in deg:
                    eta = 0.4
                else:
                    eta = 0.5
            elif 'cs' in deg:
                eta = 0.5
            elif deg == 'deblur_nonlinear':
                eta = 0.2
            elif 'hdr' in deg:
                eta = 0.1
            elif 'sr_bicubic' in deg:
                eta = 3.0
            elif 'sr4' in deg:
                eta = 7.0
            elif 'deblur_aniso' in deg:
                eta = 0.5
            else:
                eta = 1.0
        elif 'ffhq' in opt.config:
            if 'cs' in deg:
                eta = 0.5
            elif deg == 'deblur_nonlinear':
                eta = 0.2
            elif 'deblur_aniso' in deg:
                eta = 0.7
            elif 'inpainting' in deg:
                eta = 0.4
            elif 'sr4' in deg:
                eta = 7.0
            else:
                eta = 1.0
        else:
            eta = 1.0
        algo = RED_diff(model, H_funcs, sigma_0, eta=eta)
    elif opt.algo == 'diffpir':
        lam = 7.0
        algo = DiffPIR(model, H_funcs, sigma_0, lam=lam)
    elif opt.algo == 'dmps':
        algo = DMPS(model, H_funcs, sigma_0)
    elif opt.algo == 'resample':
        gamma = 100
        if 'celeba' in opt.config:
            if 'cs' in deg:
                lam = 1.0
            elif deg == 'deblur_nonlinear':
                lam = 1.0
            elif 'hdr' in deg:
                lam = 1.0
            elif 'phase' in deg:
                lam = 0.4
            elif 'deblur_aniso' in deg:
                lam = 1.0
            elif 'box' in deg:
                lam = 1.0
            elif 'sr4' in deg:
                lam = 1.0
            else:
                lam = 1.0
        elif 'ffhq' in opt.config:
            if 'cs' in deg:
                lam = 1.0
            elif deg == 'deblur_nonlinear':
                lam = 1.0
            elif 'deblur_aniso' in deg:
                lam = 1.0
            elif 'inpainting' in deg:
                lam = 1.0
            elif 'sr4' in deg:
                lam = 1.0
            else:
                lam = 1.0
        else:
            lam = 1.0
        algo = ReSample(model, H_funcs, sigma_0, gamma=gamma, lam=lam)
    elif opt.algo == 'daps':
        algo = DAPS(model, H_funcs, sigma_0, betas=betas, nonlinear=not H_funcs.is_linear())
    elif opt.algo == 'dmplug_lbfgs' or opt.algo == 'dmplug_adam' or opt.algo == 'hmc':
        algo = DMPlug(model, H_funcs, sigma_0)
    else:
        raise NotImplementedError
    
    return algo

def prepare_measurement(opt, task_config, device):
    ## get degradation matrix ##
    deg = opt.deg
    H_funcs = None
    if 'sr' in deg:
        if deg[:10] == 'sr_bicubic':
            factor = int(deg[10:])
            from obs_functions.Hfuncs import SRConv
            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
                else:
                    return 0
            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1/factor)*(i- np.floor(factor*4/2) +0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(device)
            H_funcs = SRConv(kernel / kernel.sum(), \
                            config.data.channels, config.data.image_size, device, stride = factor)
        else:
            # Super-Resolution
            blur_by = int(deg[2:])
            from obs_functions.Hfuncs import SuperResolution
            H_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, device)
    elif 'inp' in deg:
        if 'box' in deg:
            missing = torch.zeros([config.data.image_size, config.data.image_size, config.data.channels])
            # left = random.randint(16, 112)
            # up = random.randint(16, 112)
            left = 64
            up = 64
            missing[left:left+128, left:left+128, :] = 1.0
            missing = missing.view(-1).to(device).long()
            missing = torch.nonzero(missing).squeeze() 
            print(missing.shape)
        else:
            # Random inpainting
            missing_r = 3 * torch.randperm(config.data.image_size**2)[:int(config.data.image_size**2 * 0.5)].to(device).long()
            missing_g = missing_r + 1
            missing_b = missing_g + 1
            missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
        from obs_functions.Hfuncs import Inpainting
        H_funcs = Inpainting(config.data.channels, config.data.image_size, missing, device)
    elif 'deblur_gauss' in deg:
        # Gaussian Deblurring
        from obs_functions.Hfuncs import Deblurring
        sigma = 10
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
        kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(device)
        H_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, config.data.image_size, device)
    elif 'phase' in deg:
        # Phase Retrieval
        from obs_functions.Hfuncs import PhaseRetrievalOperator
        H_funcs = PhaseRetrievalOperator(oversample=2.0, device=device)
    elif 'hdr' in deg:
        # HDR
        from obs_functions.Hfuncs import HDR
        H_funcs = HDR()   
    elif 'cs' in deg:
        compress_by = int(deg[2:])
        from obs_functions.Hfuncs import WalshHadamardCS
        H_funcs = WalshHadamardCS(config.data.channels, config.data.image_size, compress_by, torch.randperm(config.data.image_size**2, device=device), device)
    elif deg == 'deblur_aniso':
        from obs_functions.Hfuncs import Deblurring2D
        sigma = 20
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
        kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(device)
        sigma = 1
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
        kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(device)
        H_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), config.data.channels, config.data.image_size, device)
    elif deg == 'deblur_nonlinear':
        from obs_functions.Hfuncs import NonlinearBlurOperator
        H_funcs = NonlinearBlurOperator(device, opt_yml_path='./bkse/options/generate_blur/default.yml')
    elif deg == 'color':
        from obs_functions.Hfuncs import Colorization
        H_funcs = Colorization(config.data.image_size, device)
    else:
        print("ERROR: degradation type not supported")
        quit()

    # for linear observations
    # if 'sr' in deg or 'inp' in deg or 'deblur_gauss' in deg:
    opt.sigma_0 = 2 * opt.sigma_0 #to account for scaling to [-1,1]
    sigma_0 = opt.sigma_0

    return H_funcs, sigma_0, deg


def sample_image(opt, config=None, model_config=None, device='cuda'):
    H_funcs, sigma_0, deg = prepare_measurement(opt, config, device)
    model = init_model(opt, config, model_config, device)
    betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
    algo = init_algo(opt, model, H_funcs, sigma_0, deg, betas)

    #get original images and corrupted y_0
    dataset, test_dataset = get_dataset(opt, config)
    
    device_count = torch.cuda.device_count()
    
    if opt.subset_start >= 0 and opt.subset_end > 0:
        assert opt.subset_end > opt.subset_start
        test_dataset = torch.utils.data.Subset(test_dataset, range(opt.subset_start, opt.subset_end))
    else:
        opt.subset_start = 0
        opt.subset_end = len(test_dataset)

    print(f'Dataset has size {len(test_dataset)}')    
    
    def seed_worker(worker_id):
        worker_seed = opt.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(opt.seed)
    if 'phase' in opt.deg:
        if config.sampling.batch_size > 1:
            key = input('Recommend using batch size 1. Current batch size is {}, switch to 1? [y/n]'.format(config.sampling.batch_size))
            if key == 'y':
                config.sampling.batch_size = 1
                print('switch to 1')
            else:
                print('keep using {}'.format(config.sampling.batch_size))

    val_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )
        
    # step size
    if opt.default_lr: # using default step size to reproduce the metrics
        raise NotImplementedError("Default step size is not implemented yet")
        N = 1
        steps=opt.timesteps
        if 'imagenet' in opt.config:
            dataset_name = 'imagenet'
        elif 'celeba' in opt.config:
            dataset_name = 'celeba'
        elif 'ffhq' in opt.config:
            dataset_name = 'ffhq'
        else:
            dataset_name = 'unknown'
        # print(deg)
        # print(steps)
        # print(sigma_0)
        # print(dataset_name)
        #lr = get_default_lr(deg, steps, sigma_0, dataset_name)
    else:
        lr = opt.lr
        N = opt.N

    print(f'Start from {opt.subset_start}')
    idx_init = opt.subset_start
    idx_so_far = opt.subset_start
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_lpips = 0.0
    pbar = tqdm(val_loader)
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
    
    for x_orig, classes in pbar:
        x_orig = x_orig.to(device)
        x_orig = data_transform(config, x_orig)

        y_0 = H_funcs.H(x_orig).detach()
        y_0 = y_0 + sigma_0 * torch.randn_like(y_0)
        # print(sigma_0)
        y_pinv = H_funcs.H_pinv(y_0).view(y_0.shape[0], config.data.channels, config.data.image_size, config.data.image_size)
        # print(y_0.shape)
        os.makedirs(opt.image_folder, exist_ok=True)

        for i in range(len(y_0)):
            tvu.save_image(
                inverse_data_transform(config, y_pinv[i]), os.path.join(opt.image_folder, f"y0_{idx_so_far + i}.png")
            )
            tvu.save_image(
                inverse_data_transform(config, x_orig[i]), os.path.join(opt.image_folder, f"orig_{idx_so_far + i}.png")
            )

        x = torch.randn(
                        y_0.shape[0],
                        config.data.channels,
                        config.data.image_size,
                        config.data.image_size,
                        device=device,
                    )
        

        skip = (opt.num_timesteps) // opt.timesteps
        seq = list(range(0, opt.num_timesteps-1, skip))
        seq_next = [-1] + list(seq[:-1])
        xt = x
        n = x.shape[0]
        
        b = torch.from_numpy(betas).float().to(device)
        
        if opt.algo == 'dmplug_lbfgs':
            xt = dmplug_lbfgs(x, n, b, seq, seq_next, algo, opt, y_0, H_funcs)
        elif opt.algo == 'dmplug_adam':
            xt = dmplug_adam(x, n, b, seq, seq_next, algo, opt, y_0, H_funcs, x_orig)
        elif opt.algo == 'hmc':
            xt = hmc(x, n, b, seq, seq_next, algo, opt, y_0, H_funcs, x_orig)
        else:
            with torch.no_grad():
                xt = iterative_sampling(x, n, b, seq, seq_next, algo, opt, y_0)

        x = [inverse_data_transform(config, y) for y in xt]

        for j in range(len(x)):
            tvu.save_image(
                x[j], os.path.join(opt.image_folder, f"{idx_so_far + j}_{i}.png")
            )
            if i == len(x)-1 or i == -1:
                orig = inverse_data_transform(config, x_orig[j])
                # print(torch.norm(orig[0]))
                mse = torch.mean((x[j].to(device) - orig) ** 2)
                psnr = 10 * torch.log10(1 / mse)
                avg_psnr += psnr
                # print(x[j].shape)
                avg_ssim += ssim(x[j].detach().cpu().numpy(), orig.detach().cpu().numpy(), data_range=x[j].detach().cpu().numpy().max() - x[j].detach().cpu().numpy().min(), channel_axis=0)
                LPIPS = loss_fn_vgg(2*orig-1.0, 2*torch.tensor(x[j]).to(torch.float32).cuda()-1.0)
                avg_lpips += LPIPS[0,0,0,0]
        idx_so_far += y_0.shape[0]

        pbar.set_description("PSNR:{}, SSIM:{}, LPIPS:{}".format(avg_psnr / (idx_so_far - idx_init), avg_ssim / (idx_so_far - idx_init), avg_lpips / (idx_so_far - idx_init)))

    avg_psnr = avg_psnr / (idx_so_far - idx_init)
    avg_ssim = avg_ssim / (idx_so_far - idx_init)
    avg_lpips = avg_lpips / (idx_so_far - idx_init)
    print("Total Average PSNR: %.2f" % avg_psnr)
    print("Total Average SSIM: %.4f" % avg_ssim)
    print("Total Average LPIPS: %.4f" % avg_lpips)
    print("Number of samples: %d" % (idx_so_far - idx_init))

def dmplug_lbfgs(x, n, b, seq, seq_next, algo, opt, y_0, H_funcs):
    x = x.requires_grad_()
    params_group1 = {'params': x, 'lr': 1, 'max_iter': 20}
    optimizer = torch.optim.LBFGS([params_group1])

    def closure():
        optimizer.zero_grad()
        xt = iterative_sampling(x, n, b, seq, seq_next, algo, opt, y_0, tqdm_disable=True).clip(-1, 1)
        error = y_0 - H_funcs.H(xt)
        loss = torch.sum(error**2)
        loss.backward()
        return loss

    epochs = 300  # SR, inpainting, nonlinear deblurring: 300
    for iterator in tqdm(range(epochs)):
        optimizer.step(closure)

    xt = iterative_sampling(x, n, b, seq, seq_next, algo, opt, y_0, tqdm_disable=True).clip(-1, 1)

    return xt.detach()

def dmplug_adam(x, n, b, seq, seq_next, algo, opt, y_0, H_funcs, x_orig):
    x = x.requires_grad_()
    params_group1 = {'params': x, 'lr': 1e-2}
    optimizer = torch.optim.Adam([params_group1])

    orig_pic = []
    for j in range(len(x_orig)):
        orig_pic.append(inverse_data_transform(config, x_orig[j]))
    psnr_list = []

    epochs = 500
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        xt = iterative_sampling(x, n, b, seq, seq_next, algo, opt, y_0, tqdm_disable=True).clip(-1, 1)
        x_save = [inverse_data_transform(config, y) for y in xt]
        for j in range(len(x_save)):
            tvu.save_image(
                x_save[j], os.path.join(opt.image_folder, f"dmplug_{epoch}.png")
            )
            mse = torch.mean((x_save[j].to(device) - orig_pic[j]) ** 2)
            psnr = 10 * torch.log10(1 / mse)
            psnr_list.append(psnr.item())
            print('PSNR:', psnr.item())
        error = y_0 - H_funcs.H(xt)
        loss = torch.sum(error**2)
        loss.backward()
        optimizer.step()

    # plot the PSNR
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(psnr_list, label='PSNR')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PSNR')
    plt.savefig(os.path.join(opt.image_folder, 'dmplug_psnr.png'))

    return xt.detach()

def hmc(x, n, b, seq, seq_next, algo, opt, y_0, H_funcs, x_orig):
    x = x.requires_grad_()
    L = max(1,math.floor(opt.tau/opt.epsilon))
    epochs = 80

    orig_pic = []
    for j in range(len(x_orig)):
        orig_pic.append(inverse_data_transform(config, x_orig[j]))
    psnr_list = []

    temp = opt.temp
    accepted = 0
    rejected = 0

    for epoch in tqdm(range(epochs)):

        # initialize momentum
        p = torch.randn_like(x, device=device) * math.sqrt(opt.m)
        xt = iterative_sampling(x, n, b, seq, seq_next, algo, opt, y_0, tqdm_disable=True).clip(-1, 1)
        loss = torch.sum((y_0 - H_funcs.H(xt))**2)
        loss_grad = torch.autograd.grad(loss, x, retain_graph=False)[0]

        H = (1/temp) * ((1/2) * torch.sum(x**2, dim=(1, 2, 3)) + (1/(2 * opt.sigma_y**2)) * loss.detach() + (1/2)* torch.sum(p * p, dim=(1, 2, 3)) * opt.m**(-1))

        x_proposal = x.detach().clone().requires_grad_(True)

        # update momentum
        p = p - (1/temp) * (opt.epsilon / 2) * (x_proposal.detach() + 1/(2 * opt.sigma_y**2) * loss_grad)

        for l in range(L):

            x_proposal = x_proposal + opt.epsilon * opt.m**(-1) * p 
            x_proposal = x_proposal.detach().requires_grad_(True)

            xt = iterative_sampling(x_proposal, n, b, seq, seq_next, algo, opt, y_0, tqdm_disable=True).clip(-1, 1)
            loss = torch.sum((y_0 - H_funcs.H(xt))**2)
            loss_grad = torch.autograd.grad(loss, x_proposal, retain_graph=False)[0]

            p = p - (1/temp) * opt.epsilon * (x_proposal.detach() + 1/(2 * opt.sigma_y**2) * loss_grad)

        p = p + (1/temp) * (opt.epsilon / 2) * (x_proposal.detach() + 1/(2 * opt.sigma_y**2) * loss_grad)

        H_proposal = (1/temp) * ((1/2) * torch.sum(x_proposal**2, dim=(1, 2, 3)) + (1/(2 * opt.sigma_y**2)) * loss.detach() + (1/2)* torch.sum(p * p, dim=(1, 2, 3)) * opt.m**(-1))

        delta_H = H_proposal - H
        acceptance_ratio = min(torch.tensor([1], device=device), torch.exp(-delta_H))
        accept = torch.rand(1).item() < acceptance_ratio.item()
        print('Acceptance ratio:', acceptance_ratio.item(), 'Delta:', delta_H.item(), 'Accept:', accept)
        if accept:
            accepted += 1
            rejected = 0
            if opt.annealed_temp:
                opt.sigma_y = opt.sigma_y * 0.94
                #opt.tau = opt.tau * 0.95
                #opt.epsilon = opt.epsilon * 0.95
                #L = max(1,math.floor(opt.tau/opt.epsilon))

            x = x_proposal.detach().clone().requires_grad_(True)
            x_save = [inverse_data_transform(config, y) for y in xt]

            for j in range(len(x_save)):
                tvu.save_image(
                    x_save[j], os.path.join(opt.image_folder, f"hmc_{epoch}.png")
                )
                mse = torch.mean((x_save[j].to(device) - orig_pic[j]) ** 2)
                psnr = 10 * torch.log10(1 / mse)
                psnr_list.append(psnr.item())
                print('PSNR:', psnr.item())
        else:
            rejected += 1
            if rejected > 2:
                opt.tau = opt.tau * 0.9
                opt.epsilon = opt.epsilon * 0.9
                L = max(1,math.floor(opt.tau/opt.epsilon))
                rejected = 0
                print('Reducing tau and epsilon, new tau:', opt.tau, 'new epsilon:', opt.epsilon, 'new L:', L)
            continue

    # plot the PSNR
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(psnr_list, label='PSNR')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PSNR')
    plt.savefig(os.path.join(opt.image_folder, 'hmc_psnr.png'))

    if opt.refine:
        xt = dmplug_adam(x, n, b, seq, seq_next, algo, opt, y_0, H_funcs, x_orig)

    return xt.detach()

def iterative_sampling(xt, n, b, seq, seq_next, algo, opt, y_0, tqdm_disable=False):

    x0_t_last = None

    for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), disable=tqdm_disable):
        t = (torch.ones(n) * i).to(xt.device)
        next_t = (torch.ones(n) * j).to(xt.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        if opt.algo == 'reddiff':
            x0_t, add_up = algo.cal_x0(xt, x0_t_last, t, at, at_next, y_0, opt.noise)
        else:
            x0_t, add_up = algo.cal_x0(xt, t, at, at_next, y_0, opt.noise)

        x0_t_last = x0_t
        xt_next = algo.map_back(x0_t, y_0, add_up, at_next, at)
        xt = xt_next

    return xt

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="?",
        help="dataset",
        default="celeba"
    )

    parser.add_argument(
        "--default_lr", action="store_true", help="Using the best step sizes to reproduce the results in the paper"
    )
    parser.add_argument(
        "--lr",
        type=float,
        nargs="?",
        help="Step-Size",
        default=1.0
    )
    parser.add_argument(
        "--N", type=int, default=1, help="N repeats"
    )
    parser.add_argument(
        "--deg", type=str, required=True, help="Degradation"
    )
    parser.add_argument(
        "--sigma_0", type=float, required=True, help="Sigma_0"
    )
    parser.add_argument(
        "--tau", type=float, default=1.0, help="Tau for HMC"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.1, help="Epsilon for HMC"
    )
    parser.add_argument(
        "--sigma_y", type=float, default=0.5, help="sigma_y for HMC (measurement noise)"
    )
    parser.add_argument(
        "--m", type=float, help="Mass Matrix Variance", default=1.0
    )
    parser.add_argument(
        "--temp", type=float, default=1.0, help="Starting Temperature for HMC sampling"
    )
    parser.add_argument(
        "--annealed_temp", action="store_true", default=False, help="Anneal the temperature during HMC sampling"
    )
    parser.add_argument(
        "--noise", type=str, default="ddpm", help="Type of Noise"
    )

    parser.add_argument(
        "--num_timesteps",
        type=int,
        nargs="?",
        help="Maximum timestep for beta schedule",
        default=1000
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="?",
        help="Number of timesteps for actual sampling",
        default=10
    )
    parser.add_argument(
        '--subset_start', type=int, default=-1
    )
    parser.add_argument(
        '--subset_end', type=int, default=-1
    )
    parser.add_argument(
        "--algo",
        type=str,
        nargs="?",
        help="Algorithm to use for sampling",
        default='resample'
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Refine the HMC results with DMPlug_LBFGS"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="exp/samples/ffhq/00000",
        help="The folder name of samples",
    )
    return parser

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

if __name__ == "__main__":
    # Load configurations
    parser = get_parser()
    device = torch.device("cuda")
    opt, unknown = parser.parse_known_args()
    #config = load_yaml('configs/config_{}.yaml')
    opt.config = 'configs/config_{}.yml'.format(opt.dataset)
    with open(opt.config, "r") as f:
        config = yaml.safe_load(f)
    model_config = config['model']
    config = dict2namespace(config)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    sample_image(opt=opt, config=config, model_config=model_config, device=device)