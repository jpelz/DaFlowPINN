import numpy as np
from scipy.fft import fftfreq, fft2, fftn
import matplotlib.pyplot as plt

mag = np.load("/scratch/jpelz/da-challenge/DA_CASE01/Analysis/binned_field_filled_200.npy")

x_ffts = []
y_ffts = []
Nx = mag.shape[0]
Ny = mag.shape[1]
Lx = 0.1/0.2 #m*
Ly = 0.1/0.2 #m*
Nz = mag.shape[2]
Nt = mag.shape[3]
Lz = 0.1/0.2 #m*
T = 0.289884 #s*



fft = fftn(mag)[:(Nx//2),:(Ny//2),:(Nz//2),:(Nt//2)]

x_fft = np.mean(fft, axis=(1,2,3))
xf = fftfreq(Nx, Lx/Nx)[:(Nx//2)]

y_fft = np.mean(fft, axis=(0,2,3))
yf = fftfreq(Ny, Ly/Ny)[:(Ny//2)]

z_fft = np.mean(fft, axis=(0,1,3))
zf = fftfreq(Nz, Lz/Nz)[:(Nz//2)]

t_fft = np.mean(fft, axis=(0,1,2))
tf = fftfreq(Nt, T/Nt)[:(Nt//2)]

np.savez("/scratch/jpelz/da-challenge/DA_CASE01/Analysis/fft_data.npz", x_fft=x_fft, xf=xf, y_fft=y_fft, yf=yf, z_fft=z_fft, zf=zf, t_fft=t_fft, tf=tf)


# fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# # Plot x_fft
# axs[0, 0].plot(xf[1:], np.abs(x_fft)[1:])

# axs[0, 0].set_title('X FFT')
# axs[0, 0].set_xlabel('Frequency (1/m)')
# axs[0, 0].set_ylabel('Amplitude')

# # Plot y_fft
# axs[0, 1].plot(yf[1:], np.abs(y_fft)[1:])
# axs[0, 1].set_title('Y FFT')
# axs[0, 1].set_xlabel('Frequency (1/m)')
# axs[0, 1].set_ylabel('Amplitude')

# # Plot z_fft
# axs[1, 0].plot(zf[1:], np.abs(z_fft)[1:])
# axs[1, 0].set_title('Z FFT')
# axs[1, 0].set_xlabel('Frequency (1/m)')
# axs[1, 0].set_ylabel('Amplitude')

# # Plot t_fft
# axs[1, 1].plot(tf[1:], np.abs(t_fft)[1:])
# axs[1, 1].set_title('T FFT')
# axs[1, 1].set_xlabel('Frequency (1/s)')
# axs[1, 1].set_ylabel('Amplitude')

# plt.tight_layout()
# plt.savefig("/content/drive/MyDrive/MA/DA_CASE01/fft_subplots.png")