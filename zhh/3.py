from skimage.metrics import peak_signal_noise_ratio as psnr_rgb
from skimage.metrics import structural_similarity as cal_ssim
psnr_rgb(SR,HR)
cal_ssim(SR,HR, multichannel=True)