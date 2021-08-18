'''Computing the visual information fidelity
 - Adapted in large part directly from the work of Alex Izvorski <aizvorski@gmail.com>, <alex@ciinow.com>
 - Original implementation modified to fit into the D.A.C test pipeline
'''
import numpy as np
import scipy.signal
import scipy.ndimage
from skimage import img_as_ubyte, img_as_float
from tqdm import tqdm



def mscale(ref, dist):
	sigma_nsq=4
	eps = 1e-10
	num = 0.0
	den = 0.0
	for scale in range(1, 5):
		N = 2**(4-scale+1) + 1
		sd = N/5.0
		if (scale > 1):
			ref = scipy.ndimage.gaussian_filter(ref, sd)
			dist = scipy.ndimage.gaussian_filter(dist, sd)
			ref = ref[::2, ::2]
			dist = dist[::2, ::2]        
		mu1 = scipy.ndimage.gaussian_filter(ref, sd)
		mu2 = scipy.ndimage.gaussian_filter(dist, sd)
		mu1_sq = mu1 * mu1
		mu2_sq = mu2 * mu2
		mu1_mu2 = mu1 * mu2
		sigma1_sq = scipy.ndimage.gaussian_filter(ref * ref, sd) - mu1_sq
		sigma2_sq = scipy.ndimage.gaussian_filter(dist * dist, sd) - mu2_sq
		sigma12 = scipy.ndimage.gaussian_filter(ref * dist, sd) - mu1_mu2
        
		sigma1_sq[sigma1_sq<0] = 0
		sigma2_sq[sigma2_sq<0] = 0
        
		g = sigma12 / (sigma1_sq + eps)
		sv_sq = sigma2_sq - g * sigma12
		g[sigma1_sq<eps] = 0
		sv_sq[sigma1_sq<eps] = sigma2_sq[sigma1_sq<eps]
		sigma1_sq[sigma1_sq<eps] = 0
        
		g[sigma2_sq<eps] = 0
		sv_sq[sigma2_sq<eps] = 0
		sv_sq[g<0] = sigma2_sq[g<0]
		g[g<0] = 0
		sv_sq[sv_sq<=eps] = eps
        
		num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
		den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))
	vifp = num/den
	if np.isnan(vifp):
		return 1.0
	else:
		return vifp



def vif(original, decoded, seq=True):
	'''Multi-scale Visual information fidelity'''    
	if seq:
		seq_vif = []
		for idx in tqdm(range(len(decoded))):
			org = img_as_ubyte(original[idx])
			dec = img_as_ubyte(decoded[idx])
			vif = mscale(org,dec)
			seq_vif.append(vif)
		return seq_vif
	else:
		original = img_as_ubyte(original)
		decoded = img_as_ubyte(decoded)
		vif = mscale(original, decoded)
		return vif
    	
    	


