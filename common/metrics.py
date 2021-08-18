from skimage.transform import resize
from skimage.metrics import structural_similarity
from modules.model import Vgg19, ImagePyramide
from modules.util import AntiAliasInterpolation2d

from scipy.ndimage.filters import gaussian_filter as __gaussian_filter
from scipy.ndimage.filters import convolve as __convolve
from scipy.ndimage.filters import correlate as __correlate
from scipy.fftpack import fftshift as __fftshift

from skimage import img_as_ubyte, img_as_float
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import imageio
from autocrop import Cropper
from skvideo.measure import msssim


import torch 
import json
import os


def wpsnr(original, decoded, seq = False):
    '''
    Computes a weighted PSNR using the constrast sensitivity function
    -Lower weights are assigned to less important frequency components.
    '''
    def __genetate_meshgrid(x, y):
        f = lambda u: u / 2 + 0.5 - 1
        (H, W) = map(f, (x, y))
        return (H, W)

    def __create_complex_planes(x, y):
        (H, W) = __genetate_meshgrid(x, y)
        (xplane, yplane) = np.mgrid[-H:H + 1, -W:W + 1]
        return (xplane, yplane)

    def __get_evaluated_contrast_sensivity(plane):
        w = 0.7
        angle = np.angle(plane)
        return ((1.0 - w) / 2.0) * np.cos(4.0 * angle) + (1.0 + w) / 2.0
    
    def __get_radial_frequency(x, y):
        (xplane, yplane) = __create_complex_planes(x, y)
        nfreq = 60
        plane = (xplane + 1.0j * yplane) / x * 2.0 * nfreq
        s = __get_evaluated_contrast_sensivity(plane)
        radfreq = abs(plane) / s
        return radfreq

    def __generate_CSF(radfreq):
        a = -((0.114 * radfreq) ** 1.1)
        csf = 2.6 * (0.0192 + 0.114 * radfreq) * np.exp(a)
        f = radfreq < 7.8909
        csf[f] = 0.9809
        return csf

    def __weighted_fft_domain(ref, quer, csf):
        err = ref.astype('double') - quer.astype('double')
        err_wt = __fftshift(np.fft.fft2(err)) * csf
        im = np.fft.fft2(ref)
        return (err, err_wt, im)
        
    def __get_weighted_error_power(err_wt):
        return (err_wt * np.conj(err_wt)).sum()

    def __get_signal_power(im):
        return (im * np.conj(im)).sum()

    def __get_ratio(mss, mse):
        if mse != 0:
            ratio = 10.0 * np.log10(mss / mse)
        else:
            ratio = float("inf")
        return np.real(ratio)
        
    if seq:
        wpsnr_list = []
        for idx in range(len(original)):
            org = img_as_ubyte(original[idx][:,:,0])
            dec = img_as_ubyte(decoded[idx][:,:,0])
    		
            size = org.shape
            (x, y) = (size[0], size[1])
            radfreq = __get_radial_frequency(x, y)
            csf = __generate_CSF(radfreq)
            (err, err_wt, im) = __weighted_fft_domain(org, dec, csf)
            mse = __get_weighted_error_power(err_wt)
            mss = __get_signal_power(im)
            wsr = __get_ratio(mss, mse)
            wpsnr_list.append(wsr)
        return wpsnr_list
    else:
        org = img_as_ubyte(original[:,:,0])
        dec = img_as_ubyte(decoded[:,:,0])
        size = org.shape
        (x, y) = (size[0], size[1])
        radfreq = __get_radial_frequency(x, y)
        csf = __generate_CSF(radfreq)
        (err, err_wt, im) = __weighted_fft_domain(org, dec, csf)
        mse = __get_weighted_error_power(err_wt)
        mss = __get_signal_power(im)
        wsr = __get_ratio(mss, mse)
        return wsr	



	

def save_video(original_video, predictions, error_video, hevc_video,error_video_hevc,result_video_path , filename, fps):
	output_video = []
	for idx in range(len(original_video)):
		composite_frame = np.concatenate((original_video[idx],predictions[idx],error_video[idx],hevc_video[idx]), axis=1)
		output_video.append(composite_frame)
	imageio.mimsave(result_video_path +filename+ "/videos/"+filename+"_av.mp4", [img_as_ubyte(frame) for frame in output_video], fps=fps)
		


def mse(original, decoded):
	'''Mean squared error'''
	mse = np.mean((img_as_ubyte(original)-img_as_ubyte(decoded))**2)
	return mse
	
def pixel_loss(kp_source, kp_driving, kp=True):
	'''returns eucledian distance between two keypoints(1) or jacobians(0)'''
	if kp:
		diff = np.subtract(kp_source['value'].cpu().detach().numpy().flatten(), kp_driving['value'].cpu().detach().numpy().flatten() )
		sq_diff = [i**2 for i in diff]
		loss = np.sqrt(np.sum(sq_diff))
	else:
		diff = np.subtract(kp_source['jacobian'].cpu().detach().numpy().flatten(), kp_driving['jacobian'].cpu().detach().numpy().flatten() )
		sq_diff = [i**2 for i in diff]
		loss = np.sqrt(np.sum(sq_diff))
	return loss
	
def crop_psnr(original, decoded, seq=False):
	'''Peak signal to noise ratio'''
	
	if seq:
		error_video = []
		seq_psnr = []
		cropper = Cropper(height=180, width=180)
		for idx in range(len(decoded)):
			error_img = original[idx]-decoded[idx]
			org = cropper.crop(img_as_ubyte(original[idx][:,:,0]))
			dec = cropper.crop(img_as_ubyte(decoded[idx][:,:,0]))
			mse_val = mse(org, dec)
			psnr_val = 10*np.log10(255**2/mse_val)
			
			error_video.append(error_img)
			seq_psnr.append(psnr_val)
		return seq_psnr, error_video
					
	else:
		error_img = img_as_float(original)-decoded
		mse_val = mse(original[:,:,0], decoded[:,:,0])
		psnr_val = 10*np.log10(255**2/mse_val)
		return psnr_val
		
def crop_ssim(original, decoded, seq=False):
	'''structual similarity index'''
	if seq:
		seq_ssim = []
		cropper = Cropper(height=180, width=180)
		for idx in range(len(decoded)):
			ref = cropper.crop(img_as_ubyte(original[idx][:,:,0]))
			dec = cropper.crop(img_as_ubyte(decoded[idx][:,:,0]))
			ssim_val = structural_similarity(ref, dec, data_range = np.amax(dec)-np.amin(dec))
			seq_ssim.append(ssim_val)
		return seq_ssim
	else:
		ref = img_as_ubyte(original[:,:,0])
		dec = img_as_ubyte(decoded[:,:,0])
		ssim_val = structural_similarity(ref, dec, data_range = np.amax(dec)-np.amin(dec))
		return ssim_val
		
def crop_perceptual_loss(original, decoded,seq = False):
	loss_weights = [1, 1, 1, 1, 1]
	scales  = [1, 0.5, 0.25,0.125]
	vgg = Vgg19()
	pyramid = ImagePyramide(scales, 3)
	
	if torch.cuda.is_available():
		vgg = vgg.cuda()
		pyramid = pyramid.cuda()	
	if seq:
		seq_perceptual_loss = []
		for idx in tqdm(range(len(decoded))):
			cropper = Cropper(height=180, width=180)	
			org_frame = cropper.crop(img_as_ubyte(original[idx]))
			dec_frame = cropper.crop(img_as_ubyte(decoded[idx]))
			
			org = torch.tensor(img_as_float(org_frame)[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
			dec = torch.tensor(img_as_float(dec_frame)[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
			
			if torch.cuda.is_available():
				org = org.cuda()
				dec = dec.cuda()
			
			pyramide_real = pyramid(org)
			pyramide_generated = pyramid(dec)
			total_loss = 0
			x_vgg = vgg(pyramide_generated['prediction_' + str(1)])
			y_vgg = vgg(pyramide_real['prediction_' + str(1)])
			total_loss += torch.abs(x_vgg[3] - y_vgg[3].detach()).mean()
			seq_perceptual_loss.append(total_loss.item())
			'''
			for scale in scales:
				x_vgg = vgg(pyramide_generated['prediction_' + str(scale)])
				y_vgg = vgg(pyramide_real['prediction_' + str(scale)])
				for i, weight in enumerate(loss_weights):
					value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
					total_loss += loss_weights[i] * value
			'''
			#seq_perceptual_loss.append(total_loss.item())
		return seq_perceptual_loss	

def psnr(original, decoded, seq=False):
	'''Peak signal to noise ratio'''
	if seq:
		error_video = []
		seq_psnr = []
		for idx in range(len(decoded)):
			error_img = original[idx]-decoded[idx]
			mse_val = mse(original[idx][:,:,0], decoded[idx][:,:,0])
			psnr_val = 10*np.log10(255**2/mse_val)
			
			error_video.append(error_img)
			seq_psnr.append(psnr_val)
		return seq_psnr, error_video
					
	else:
		mse_val = mse(original[:,:,0], decoded[:,:,0])
		psnr_val = 10*np.log10(255**2/mse_val)
		return psnr_val
	
def ssim(original, decoded, seq=False):
	'''structual similarity index'''
	if seq:
		seq_ssim = []
		for idx in range(len(decoded)):
			ref = img_as_ubyte(original[idx][:,:,0])
			dec = img_as_ubyte(decoded[idx][:,:,0])
			ssim_val = structural_similarity(ref, dec, data_range = np.amax(dec)-np.amin(dec), size_average=False)
			seq_ssim.append(ssim_val)
		return seq_ssim
	else:
		ref = img_as_ubyte(original[:,:,0])
		dec = img_as_ubyte(decoded[:,:,0])
		ssim_val = structural_similarity(ref, dec, data_range = np.amax(dec)-np.amin(dec), size_average=False)
		return ssim_val
		
def mssim(original, decoded, seq=False):
	'''Multiscale structual similarity index'''
	if seq:
		frames = len(decoded)
		org = [frame[:,:,0] for frame in original[0:frames]]
		dec = [frame[:,:,0] for frame in decoded]
		seq_ssim = msssim(org, dec)
		seq_ssim = [x.astype(float) for x in seq_ssim]
		return seq_ssim
	else:
		val = msssim(original,decoded)
		return val.astype(float)
	
def perceptual_loss(original, decoded,seq = False):
	loss_weights = [1, 1, 1, 1, 1]
	scales  = [1, 0.5, 0.25,0.125]
	vgg = Vgg19()
	pyramid = ImagePyramide(scales, 3)
	if torch.cuda.is_available():
		vgg = vgg.cuda()
		pyramid = pyramid.cuda()	
	if seq:
		seq_perceptual_loss = []
		for idx in tqdm(range(len(decoded))):		
			org_frame = deepcopy(original[idx])
			dec_frame = deepcopy(decoded[idx])
			org = torch.tensor(img_as_float(org_frame)[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
			dec = torch.tensor(img_as_float(dec_frame)[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
			
			if torch.cuda.is_available():
				org = org.cuda()
				dec = dec.cuda()
			
			pyramide_real = pyramid(org)
			pyramide_generated = pyramid(dec)
			total_loss = 0
			x_vgg = vgg(pyramide_generated['prediction_' + str(1)])
			y_vgg = vgg(pyramide_real['prediction_' + str(1)])
			total_loss += torch.abs(x_vgg[3] - y_vgg[3].detach()).mean()
			seq_perceptual_loss.append(total_loss.item())
			'''
			for scale in scales:
				x_vgg = vgg(pyramide_generated['prediction_' + str(scale)])
				y_vgg = vgg(pyramide_real['prediction_' + str(scale)])
				for i, weight in enumerate(loss_weights):
					value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
					total_loss += loss_weights[i] * value
			
			seq_perceptual_loss.append(total_loss.item())
			'''
		return seq_perceptual_loss	
	else:
		org_frame = deepcopy(original)
		dec_frame = deepcopy(decoded)
		org = torch.tensor(org_frame[np.newaxis].astype(np.float32)).permute(0,3, 1, 2)
		dec = torch.tensor(dec_frame[np.newaxis].astype(np.float32)).permute(0,3, 1, 2)
		if torch.cuda.is_available():
			org = org.cuda()
			dec = dec.cuda()
		
		pyramide_real = pyramid(org)
		pyramide_generated = pyramid(dec)
		total_loss = 0
		'''
		for scale in scales:
			x_vgg = vgg(pyramide_generated['prediction_' + str(scale)])
			y_vgg = vgg(pyramide_real['prediction_' + str(scale)])
			for i, weight in enumerate(loss_weights):
				value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
				total_loss += loss_weights[i] * value
		'''
		x_vgg = vgg(pyramide_generated['prediction_' + str(1)])
		y_vgg = vgg(pyramide_real['prediction_' + str(1)])
		total_loss += torch.abs(x_vgg[3] - y_vgg[3].detach()).mean()
		return total_loss.item()
	
def bitrate(total_bytes,fps):
	seq_bitrate = total_bytes/(1000*fps)
	return 8 * seq_bitrate
	
	
def find_nearest_hevc(hevc_bitrates, dac_bitrate):
	qp = [30,35,40,45,50]
	array = np.asarray(hevc_bitrates)
	idx = (np.abs(array - dac_bitrate)).argmin()
	return qp[idx], idx
	
def evaluate(original, decoded):
	frame_psnr, error_frame = psnr(original, decoded)
	frame_ssim = compare_ssim(original, decoded)
	frame_perp_loss = perceptual_loss(img_as_float(original), img_as_float(decoded))
	return frame_psnr , frame_ssim, error_frame, frame_perp_loss
	
def analysis(original_video, decoded_sequence,hevc_video, result_video_path, stream_name, fps):
	#create comparison videos and plots
	seq_psnr = []
	seq_ssim = []
	perp_loss = []
	result_video = [] #original_video | decoded_video | error_video | hevc video
	for frame in range(len(original_video)):
		#error_frame = resize(original_video[frame], (256, 256))[..., :3] - decoded_sequence[frame]
		frame_psnr , frame_ssim, error_frame,frame_p_loss= evaluate(original_video[frame], decoded_sequence[frame])
		seq_psnr.append(frame_psnr)
		seq_ssim.append(frame_ssim)
		perp_loss.append(frame_p_loss)
		composite_frame = np.concatenate((original_video[frame],img_as_ubyte(decoded_sequence[frame]),img_as_ubyte(error_frame),hevc_video[frame]), axis=1)
		result_video.append(composite_frame)
	#save video sequence
	imageio.mimsave(result_video_path + stream_name+"_av.mp4", result_video, fps=fps)
	return seq_psnr, seq_ssim,perp_loss
	
