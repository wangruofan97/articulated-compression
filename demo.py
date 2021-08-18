"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import Generator
from modules.region_predictor import RegionPredictor, svd
from modules.bg_motion_predictor import BGMotionPredictor
from modules.avd_network import AVDNetwork
from modules import compress
from modules.util import make_coordinate_grid
from animate import get_animation_region_params
import matplotlib
import gzip
import os

matplotlib.use('Agg')

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")


def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.load(f)

    generator = Generator(num_regions=config['model_params']['num_regions'],
                          num_channels=config['model_params']['num_channels'],
                          **config['model_params']['generator_params'])
    if not cpu:
        generator.cuda()

    region_predictor = RegionPredictor(num_regions=config['model_params']['num_regions'],
                                       num_channels=config['model_params']['num_channels'],
                                       estimate_affine=config['model_params']['estimate_affine'],
                                       **config['model_params']['region_predictor_params'])
    if not cpu:
        region_predictor.cuda()

    bg_predictor = BGMotionPredictor(num_channels=config['model_params']['num_channels'],
                                       **config['model_params']['bg_predictor_params'])
    if not cpu:
        bg_predictor.cuda()

    avd_network = AVDNetwork(num_regions=config['model_params']['num_regions'],
                             **config['model_params']['avd_network_params'])
    if not cpu:
        avd_network.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    region_predictor.load_state_dict(checkpoint['region_predictor'])
    bg_predictor.load_state_dict(checkpoint['bg_predictor'])
    if 'avd_network' in checkpoint:
        avd_network.load_state_dict(checkpoint['avd_network'])

    if not cpu:
        generator = DataParallelWithCallback(generator)
        region_predictor = DataParallelWithCallback(region_predictor)
        bg_predictor = DataParallelWithCallback(bg_predictor)
        avd_network = DataParallelWithCallback(avd_network)

    generator.eval()
    region_predictor.eval()
    bg_predictor.eval()
    avd_network.eval()

    return generator, region_predictor, bg_predictor, avd_network


def make_animation(source_image, driving_video, generator, region_predictor, bg_predictor, avd_network,
                   animation_mode='standard', cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        source_region_params = region_predictor(source)
        driving_region_params_initial = region_predictor(driving[:, :, 0])

        kp_list = {'affine': [], 'covar': [], 'shift': []} 
        
        driving_region_params_initial['shift'] *= 256
        driving_region_params_initial['shift'] = driving_region_params_initial['shift'].round()
        driving_region_params_initial['shift'] /= 256.
        region = driving_region_params_initial['heatmap']
        shape = region.shape
        region = region.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], region.type()).unsqueeze_(0).unsqueeze_(0)
        mean = (region * grid).sum(dim=(2, 3))

        mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2)
        covar = torch.matmul(mean_sub.unsqueeze(-1), mean_sub.unsqueeze(-2))
        covar = covar * region.unsqueeze(-1)
        covar = covar.sum(dim=(2, 3))
        driving_region_params_initial['covar'] = covar
        # torch.matmul(driving_region_params_initial['affine'], driving_region_params_initial['affine'].permute(0, 1, 3, 2))

        shape = covar.shape
        covar = covar.view(-1, 2, 2)
        u, s, v = svd(covar)
        d = torch.diag_embed(s ** 0.5)
        sqrt = torch.matmul(u, d)
        sqrt = sqrt.view(*shape)
        driving_region_params_initial['affine'] = sqrt

        print(driving_region_params_initial['heatmap'].shape)
        print(driving_region_params_initial['shift'].shape)
        print(driving_region_params_initial['affine'].shape)
        print(driving_region_params_initial['covar'].shape)
        kp_list['affine'].append(driving_region_params_initial['affine'].cpu().detach().numpy())
        # kp_list['covar'].append(driving_region_params_initial['covar'].cpu().detach().numpy())
        kp_list['shift'].append(driving_region_params_initial['shift'].cpu().detach().numpy())
        # source_region_params = driving_region_params_initial
        # print(source_region_params)

        for frame_idx in tqdm(range(driving.shape[2])):
            if frame_idx >= 120:
                break
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            driving_region_params = region_predictor(driving_frame)
            driving_region_params['u'] = 0
            driving_region_params['d'] = 0
            # driving_region_params['heatmap'] *= 0
            # driving_region_params['affine'] *= 0
            
            driving_region_params['shift'] *= 256
            driving_region_params['shift'] = driving_region_params['shift'].round()
            driving_region_params['shift'] /= 256.
            region = driving_region_params['heatmap']
            shape = region.shape
            region = region.unsqueeze(-1)
            grid = make_coordinate_grid(shape[2:], region.type()).unsqueeze_(0).unsqueeze_(0)
            mean = (region * grid).sum(dim=(2, 3))

            mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2)
            covar = torch.matmul(mean_sub.unsqueeze(-1), mean_sub.unsqueeze(-2))
            covar = covar * region.unsqueeze(-1)
            covar = covar.sum(dim=(2, 3))
            driving_region_params['covar'] = covar
            # torch.matmul(driving_region_params_initial['affine'], driving_region_params_initial['affine'].permute(0, 1, 3, 2))

            shape = covar.shape
            covar = covar.view(-1, 2, 2)
            u, s, v = svd(covar)
            d = torch.diag_embed(s ** 0.5)
            sqrt = torch.matmul(u, d)
            sqrt = sqrt.view(*shape)
            driving_region_params['affine'] = sqrt
            
            kp_list['affine'].append(driving_region_params['affine'].cpu().detach().numpy())
            # kp_list['covar'].append(driving_region_params['covar'].cpu().detach().numpy())
            kp_list['shift'].append(driving_region_params['shift'].cpu().detach().numpy())

            new_region_params = get_animation_region_params(source_region_params, driving_region_params,
                                                            driving_region_params_initial, avd_network=avd_network,
                                                            mode=animation_mode)
            # print(new_region_params['affine'])
            # bg_params=bg_predictor(source, driving_frame)
            out = generator(source, source_region_params=source_region_params, driving_region_params=new_region_params)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

        bytes = 458
        for i in range(len(kp_list["affine"])-2,-1,-1):
            kp_list["affine"][i+1] -= kp_list["affine"][i]
        # for i in range(len(kp_list["covar"])-2,-1,-1):
        #     kp_list["covar"][i+1] -= kp_list["covar"][i]
        for i in range(len(kp_list["shift"])-2,-1,-1):
            kp_list["shift"][i+1] -= kp_list["shift"][i]
        with open("1_af.npy", "wb") as keypoints:
            np.save(file= keypoints, arr = np.array(kp_list['affine']))
        compress.compress("1_af.npy", "1_af.cp")
        # bytes += os.path.getsize("1_af.cp")
        # with open("1_co.npy", "wb") as keypoints:
        #     np.save(file= keypoints, arr = np.array(kp_list['covar']))
        # compress.compress("1_co.npy", "1_co.cp")
        # bytes += os.path.getsize("1_co.cp")
        with open("1_sh.npy", "wb") as keypoints:
            np.save(file= keypoints, arr = np.array(kp_list['shift']))
        compress.compress("1_sh.npy", "1_sh.cp")
        bytes += os.path.getsize("1_sh.cp")

        print(bytes / 128. / len(kp_list["affine"]) * 30)
    return predictions

def add_kp_list(kp_list,kp_driving):
	#accumulates the driving kps for buffer-wise entropy coding
	kp_list['affine'].append(kp_driving['affine'])
	kp_list['covar'].append(kp_driving['covar'])
	kp_list['shift'].append(kp_driving['shift'])

def main(opt):
    source_image = imageio.imread(opt.source_image)
    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']
    reader.close()
    driving_video = imageio.mimread(opt.driving_video, memtest=False)

    source_image = resize(source_image, opt.img_shape)[..., :3]
    driving_video = [resize(frame, opt.img_shape)[..., :3] for frame in driving_video]
    generator, region_predictor, bg_predictor, avd_network = load_checkpoints(config_path=opt.config,
                                                                checkpoint_path=opt.checkpoint, cpu=opt.cpu)
    predictions = make_animation(source_image, driving_video, generator, region_predictor, bg_predictor, avd_network,
                                 animation_mode=opt.mode, cpu=opt.cpu)
    imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='ted384.pth', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='sup-mat/source.png', help="path to source image")
    parser.add_argument("--driving_video", default='sup-mat/driving.mp4', help="path to driving video")
    parser.add_argument("--result_video", default='result.mp4', help="path to output")

    parser.add_argument("--mode", default='avd', choices=['standard', 'relative', 'avd'],
                        help="Animation mode")
    parser.add_argument("--img_shape", default="384,384", type=lambda x: list(map(int, x.split(','))),
                        help='Shape of image, that the model was trained on.')
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    main(parser.parse_args())
