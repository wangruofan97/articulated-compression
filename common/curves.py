import matplotlib.pyplot as plt
import numpy as np

def plot_psnr(seq_psnr,frames,results_path, filename = None, seq_hevc_psnr = None):
	if seq_hevc_psnr != None:
		plt.plot(range(frames),seq_psnr)
		#plt.plot(range(frames),seq_wpsnr)
		plt.plot(range(frames),seq_hevc_psnr)
		#plt.plot(range(frames),seq_hevc_wpsnr)
		plt.title(":PSNR:: "+filename)
		plt.legend(("PSNR_DAC","PSNR_HEVC"))
		plt.xlabel("Frames")
		plt.ylabel("PSNR(dB)")
		plt.savefig(results_path +'/psnr_curve.png')
		plt.close()	
	else:
		plt.plot(range(frames),seq_psnr)
		plt.title("PSNR Curve")
		plt.legend(("Mean PSNR DAC: {} dB".format(round(np.mean(np.array(seq_psnr)),2))))
		plt.xlabel("Frames")
		plt.ylabel("PSNR(dB)")
		plt.savefig(results_path +'/psnr_curve.png')
		plt.close()	

def plot_psnr_compare(seq_psnr, seq_psnr_jpeg, frames,results_path,  filename):
	plt.plot(range(frames),seq_psnr)
	plt.plot(range(frames),seq_psnr_jpeg)
	plt.title(":PSNR:: "+filename)
	plt.legend(("PSNR_DAC","PSNR_HEVC"))
	plt.xlabel("Frames")
	plt.ylabel("PSNR(dB)")
	plt.savefig(results_path + filename+"/curves/"+filename+"_psnr_compare.png")
	plt.close()	
	
def plot_perp_loss(seq_perp_loss,frames,results_path,  filename = None, perp_loss_hevc = None):
	if perp_loss_hevc != None:
		plt.plot(range(frames),seq_perp_loss)
		plt.plot(range(frames),perp_loss_hevc)
		plt.title(":Perceptual_loss: "+filename)
		plt.legend(("P_loss DAC", "P_Loss_HEVC"))
		plt.xlabel("Frames")
		plt.ylabel("Perceptual")
		plt.savefig(results_path + filename+"/curves/"+filename+"_perp_hevc_seq.png")
		plt.close()	
	else:
		plt.plot(range(frames),seq_perp_loss)
		plt.title(":Perceptual_loss")
		plt.legend(("P_loss DAC"))
		plt.xlabel("Frames")
		plt.ylabel("Perceptual Loss")
		plt.savefig(results_path + "/perceptual_loss.png")
		plt.close()	
	
	
def plot_ssim(seq_ssim,frames,results_path,  filename, seq_hevc_ssim = None):
	if seq_hevc_ssim != None:
		plt.plot(range(frames),seq_ssim)
		plt.plot(range(frames),seq_hevc_ssim)
		plt.title("SSIM:: "+filename)
		plt.legend(("SSIM D.A.C","SSIM HEVC"))
		plt.xlabel("Frames")
		plt.ylabel("SSIM")
		plt.savefig(results_path + filename+"/curves/"+filename+"_ssim.png")
		plt.close()	
	else:
		plt.plot(range(frames),seq_ssim)
		plt.title("SSIM:: "+filename)
		plt.legend(("SSIM"))
		plt.xlabel("Frames")
		plt.ylabel("SSIM-PSNR")
		plt.savefig(results_path + filename+"/curves/"+filename+"_ssim.png")
		plt.close()	
		
def plot_torch_ssim(seq_ssim,frames,results_path,  filename, seq_hevc_ssim = None):
	if seq_hevc_ssim != None:
		plt.plot(range(frames),seq_ssim)
		plt.plot(range(frames),seq_hevc_ssim)
		plt.title("TORCH SSIM")
		plt.legend(("SSIM D.A.C","SSIM HEVC"))
		plt.xlabel("Frames")
		plt.ylabel("SSIM")
		plt.savefig(results_path + filename+"/curves/"+filename+"_torch_ssim.png")
		plt.close()	
	else:
		plt.plot(range(frames),seq_ssim)
		plt.title("SSIM:: "+filename)
		plt.legend(("SSIM"))
		plt.xlabel("Frames")
		plt.ylabel("SSIM-PSNR")
		plt.savefig(results_path + filename+"/curves/"+filename+"_torch_ssim.png")
		plt.close()	
		
def plot_eval_ssim(regular_ssim, torch_ssim,frames,results_path,  filename):
	plt.plot(range(frames),regular_ssim)
	plt.plot(range(frames),torch_ssim)
	plt.title("2-Methods::SSIM")
	plt.ylabel("SSIM")
	plt.xlabel("Frames")
	plt.legend(("Regular SSIM","Torch SSIM"))
	plt.savefig(results_path + filename+"/curves/"+filename+"_torch_vs_regular_ssim.png")
	plt.close()
	

def plot_rd(hevc_bitrate, hevc_psnr,seq_bitrate,mean_psnr,close_hevc_bitrate,mean_psnr_hevc,frames,results_path, filename):
	curve = plt.plot(hevc_bitrate, hevc_psnr)
	point_1 = plt.scatter(seq_bitrate, mean_psnr, s=5, c = '#ff0a2c')
	point_2 = plt.scatter(close_hevc_bitrate, mean_psnr_hevc, s=5, c = '#b9681b')
	plt.legend(("HEVC","D-A-C: Bitrate:{} kbps".format(round(seq_bitrate,2)),"Close HEVC Bitrate:{} kbps".format(round(close_hevc_bitrate,2)))) 
	plt.title("Open Gop::"+filename+ " Frames: "+ str(frames))
	plt.xlabel("Bitrate(kbps)")
	plt.ylabel("PSNR(dB)")
	plt.savefig(results_path + filename+"/curves/"+filename+"_rd.png")
	plt.close()
