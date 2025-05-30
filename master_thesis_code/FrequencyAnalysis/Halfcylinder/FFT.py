import netCDF4 as nc
import numpy as np
from scipy.fft import fftfreq, fft2, fftn
import matplotlib.pyplot as plt

datafile = "/scratch/jpelz/ma-pinns/TrackGen/halfcylinder.nc"

data=nc.Dataset(datafile)

xdim=data['xdim'][:]
ydim=data['ydim'][:]
zdim=data['zdim'][:]
tdim=data['tdim'][:]


x_ffts = []
y_ffts = []
Nx = 640
Ny = 240
Lx = 8 #m
Ly = 3 #m
Nz = 80
Nt = 150
Lz = 1 #m
T = 15 #s


u = data['u'][:,:,:,:]
v = data['v'][:,:,:,:]
w = data['w'][:,:,:,:]


mag = np.sqrt(u**2 + v**2 + w**2)

fft = fftn(mag)#[:(Nt//2),:(Nz//2),:(Ny//2),:(Nx//2)]

x_fft = np.mean(fft, axis=(0,1,2))
xf = fftfreq(Nx, Lx/Nx)#[:(Nx//2)]

y_fft = np.mean(fft, axis=(0,1,3))
yf = fftfreq(Ny, Ly/Ny)#[:(Ny//2)]

z_fft = np.mean(fft, axis=(0,2,3))
zf = fftfreq(Nz, Lz/Nz)#[:(Nz//2)]

t_fft = np.mean(fft, axis=(1,2,3))
tf = fftfreq(Nt, T/Nt)#[:(Nt//2)]

np.savez("fft_data.npz", x_fft=x_fft, xf=xf, y_fft=y_fft, yf=yf, z_fft=z_fft, zf=zf, t_fft=t_fft, tf=tf)


fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot x_fft
axs[0, 0].plot(xf[1:], np.abs(x_fft)[:])

axs[0, 0].set_title('X FFT')
axs[0, 0].set_xlabel('Frequency (1/m)')
axs[0, 0].set_ylabel('Amplitude')

# Plot y_fft
axs[0, 1].plot(yf[1:], np.abs(y_fft)[:])
axs[0, 1].set_title('Y FFT')
axs[0, 1].set_xlabel('Frequency (1/m)')
axs[0, 1].set_ylabel('Amplitude')

# Plot z_fft
axs[1, 0].plot(zf[1:], np.abs(z_fft)[:])
axs[1, 0].set_title('Z FFT')
axs[1, 0].set_xlabel('Frequency (1/m)')
axs[1, 0].set_ylabel('Amplitude')

# Plot t_fft
axs[1, 1].plot(tf[1:], np.abs(t_fft)[:])
axs[1, 1].set_title('T FFT')
axs[1, 1].set_xlabel('Frequency (1/s)')
axs[1, 1].set_ylabel('Amplitude')

plt.tight_layout()
plt.savefig("fft_subplots.png")
