
import argparse, os, yaml
import torch
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
from util.img_utils import Blurkernel, clear_color, generate_tilt_map, mask_generator
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from ddim_sampler import *
import shutil
import lpips
from tqdm import tqdm

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def dmplug(model, scheduler, logdir, img='00000', eta=0, tau=0.1, epsilon=0.01, zeta=10, cutoff=100, dataset='celeba',img_model_config=None,task_config=None,device='cuda'):
    dtype = torch.float32
    gt_img_path = './data/{}/{}.png'.format(dataset,img)
    gt_img = Image.open(gt_img_path).convert("RGB")
    shutil.copy(gt_img_path, os.path.join(logdir, 'gt.png'))
    ref_numpy = np.array(gt_img) / 255.0
    x = ref_numpy * 2 - 1
    x = x.transpose(2, 0, 1)
    ref_img = torch.Tensor(x).to(dtype).to(device).unsqueeze(0)
    ref_img.requires_grad = False
    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])

    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
            **measure_config['mask_opt']
        )
        mask = mask_gen(ref_img)
        mask = mask[:, 0, :, :].unsqueeze(dim=0)
        # Forward measurement model (Ax + n)
        y = operator.forward(ref_img, mask=mask)
        y_n = noiser(y)
    else:
        # Forward measurement model (Ax + n)
        mask=None
        y = operator.forward(ref_img)
        y_n = noiser(y)

    y_n.requires_grad = False
    plt.imsave(os.path.join(logdir, 'measurement.png'), clear_color(y_n))

    # DMPlug
    Z = torch.randn((1, 3, img_model_config['image_size'], img_model_config['image_size']), device=device, dtype=dtype, requires_grad=True)
    sse = torch.nn.MSELoss(reduction='sum').to(device)

    epochs = 300 # SR, inpainting: 5,000, nonlinear deblurring: 10,000
    L = max(1,math.floor(tau/epsilon))
    psnrs = []
    ssims = []
    losses = []
    lpipss = []
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    accepted = 0
    sig_p = 1

    
    for iterator in tqdm(range(epochs)):
        zeta *= 1.02
        # initialize momentum
        p = torch.randn_like(Z, device=device, dtype=dtype) * sig_p
        if iterator == 0:
            output, measurement_err, _ = ddim_denoise(model, scheduler, Z, eta, measure_config, operator, y_n, sse, mask)
            H = zeta * measurement_err.detach() + (1/2)* torch.sum(p * p, dim=(1, 2, 3))

        Z_proposal = Z.detach().clone().requires_grad_(True)

        p_original = p.detach().clone()
        p_norm = torch.linalg.norm(p_original).item()

        for l in range(L):
        
            output, measurement_err, measurement_grad = ddim_denoise(model, scheduler, Z_proposal, eta, measure_config, operator, y_n, sse, mask)

            # update momentum
            p = p - (epsilon / 2) * (Z_proposal.detach() + zeta * measurement_grad)

            Z_proposal = Z_proposal + epsilon * p 
            Z_proposal = Z_proposal.detach().requires_grad_(True)

            output, measurement_err, measurement_grad = ddim_denoise(model, scheduler, Z_proposal, eta, measure_config, operator, y_n, sse, mask)

            p = p - (epsilon / 2) * (Z_proposal.detach() + zeta * measurement_grad)

        norm_change = torch.linalg.norm(p - p_original).item()
        print('leap', l,  'norm', norm_change, 'ratio', norm_change / p_norm)
        sig_p *= norm_change / p_norm * 2

        H_proposal = zeta * measurement_err.detach() + (1/2)* torch.sum(p * p, dim=(1, 2, 3))

        delta_H = H_proposal - H
        acceptance_ratio = min(torch.tensor([1], device=device), torch.exp(-delta_H))
        accept = torch.rand(1).item() < acceptance_ratio.item()
        print('Iteration:', iterator, 'H:', H.item(), 'H_proposal:', H_proposal.item())
        print('Acceptance ratio:', acceptance_ratio.item(), 'result', accept)
        if accept:
            accepted += 1
            Z = Z_proposal.detach().clone().requires_grad_(True)
            H = H_proposal.clone()
        else:
            continue

        with torch.no_grad():
            output_numpy = output.detach().cpu().squeeze().numpy()
            output_numpy = (output_numpy + 1) / 2
            output_numpy = np.transpose(output_numpy, (1, 2, 0))
            # calculate psnr
            tmp_psnr = peak_signal_noise_ratio(ref_numpy, output_numpy)
            psnrs.append(tmp_psnr)
            print('PSNR: ', tmp_psnr)
            # calculate ssim
            tmp_ssim = structural_similarity(ref_numpy, output_numpy, channel_axis=2, data_range=1)
            ssims.append(tmp_ssim)
            # calculate lpips
            rec_img_torch = torch.from_numpy(output_numpy).permute(2, 0, 1).unsqueeze(0).float().to(device)
            gt_img_torch = torch.from_numpy(ref_numpy).permute(2, 0, 1).unsqueeze(0).float().to(device)
            rec_img_torch = rec_img_torch * 2 - 1
            gt_img_torch = gt_img_torch * 2 - 1
            lpips_alex = loss_fn_alex(gt_img_torch, rec_img_torch).item()
            lpipss.append(lpips_alex)

            if iterator % 1 == 0:
                plt.imsave(os.path.join(logdir, f"img_{iterator}.png"), output_numpy)

            if len(psnrs) == 1 or (len(psnrs) > 1 and tmp_psnr > np.max(psnrs[:-1])):
                best_img = output_numpy

        losses.append(measurement_err.item())

    plt.imsave(os.path.join(logdir, "rec_img.png"), best_img)

    plt.plot(np.array(losses), label='all')
    plt.legend()
    plt.savefig(os.path.join(logdir, 'loss.png'))
    plt.close()

    plt.plot(np.array(psnrs))
    plt.title('Max PSNR: {}'.format(np.max(np.array(psnrs))))
    plt.savefig(os.path.join(logdir, 'psnr.png'))
    plt.close()

    psnr_res = np.max(psnrs)
    ssim_res = np.max(ssims)
    lpips_res = np.min(lpipss)
    
    print('PSNR: {}, SSIM: {}, LPIPS: {}'.format(psnr_res, ssim_res, lpips_res))
    print('Last {} samples: PSNR: {}, SSIM: {}, LPIPS: {}'.format(cutoff, np.mean(psnrs[-cutoff:]), np.mean(ssims[-cutoff:]), np.mean(lpipss[-cutoff:])))
    print('Accepted samples:', accepted, 'out of', epochs, 'acceptance ratio:', accepted / epochs)

def ddim_denoise(model, scheduler, Z, eta, measure_config, operator, y_n, sse, mask):
    model.eval()

    for i, tt in enumerate(scheduler.timesteps):
        t = (torch.ones(1) * tt).cuda()
        
        if i == 0:
            input = Z
        else:
            input = x_t

        noise_pred = model(input, t)
        noise_pred = noise_pred[:, :3]
        ddpm_output = scheduler.step(noise_pred, tt, input, return_dict=True, use_clipped_model_output=True, eta=eta)

        x_t = ddpm_output.prev_sample

    output = torch.clamp(x_t, -1, 1)

    if measure_config['operator']['name'] == 'inpainting':
        measurement_err = sse(operator.forward(output, mask=mask), y_n)
    else:
        measurement_err = sse(operator.forward(output), y_n)

    measurement_grad = torch.autograd.grad(measurement_err, Z, retain_graph=False)[0]

    return output, measurement_err, measurement_grad

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=0.0
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="logdir",
        default="./results"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="?",
        help="dataset",
        default="celeba"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fast sampling",
        default=3
    )
    parser.add_argument(
        "--lr",
        type=float,
        nargs="?",
        help="lr of z",
        default=0.01
    )
    parser.add_argument(
        "--task",
        type=str,
        nargs="?",
        help="super_resolution,inpainting,nonlinear_deblur",
        default='super_resolution'
    )
    parser.add_argument(
        "--img",
        type=int,
        nargs="?",
        help="image id",
        default=0
    )
    return parser
def torch_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == "__main__":
    torch_seed(123)
    # Load configurations
    parser = get_parser()
    device = torch.device("cuda")
    opt, unknown = parser.parse_known_args()
    img_model_config = 'configs/model_config_{}.yaml'.format(opt.dataset)
    task_config = 'configs/tasks/{}_config.yaml'.format(opt.task)
    img_model_config = load_yaml(img_model_config)
    model = create_model(**img_model_config)
    model = model.to(device)
    model.eval()
    task_config = load_yaml(task_config)
    # Define the DDIM scheduler
    scheduler = DDIMScheduler()
    scheduler.set_timesteps(opt.custom_steps)
    img = str(opt.img).zfill(5)
    logdir = os.path.join(opt.logdir, opt.task, opt.dataset, img)
    os.makedirs(logdir,exist_ok=True)
    # DMPlug
    dmplug(model, scheduler, logdir, img=img, eta=opt.eta, tau=0.02, epsilon=0.002, zeta=60, cutoff=100, dataset=opt.dataset, img_model_config=img_model_config, task_config = task_config, device=device)