import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, mu, sig):
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )


fft_data = np.load("fft_data.npz")
x_fft = fft_data['x_fft']
xf = fft_data['xf']
y_fft = fft_data['y_fft']
yf = fft_data['yf']
z_fft = fft_data['z_fft']
zf = fft_data['zf']
t_fft = fft_data['t_fft']
tf = fft_data['tf']

sig_x = 5
sig_y = 3
sig_z = 5
sig_t = 10

nd_x = gaussian(xf, 0, sig_x)
nd_y = gaussian(yf, 0, sig_y)
nd_z = gaussian(zf, 0, sig_z)
nd_t = gaussian(tf, 0, sig_t)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot x_fft
axs[0, 0].plot(xf[1:], np.abs(x_fft)[1:])
ax2 = axs[0, 0].twinx()
ax2.plot(xf[1:], nd_x[1:], 'r--', linewidth=0.5)
ax2.get_yaxis().set_visible(False)
axs[0, 0].set_title('X FFT')
axs[0, 0].set_xlabel('Frequency (1/m)')
axs[0, 0].set_ylabel('Amplitude')

# Plot y_fft
axs[0, 1].plot(yf[1:], np.abs(y_fft)[1:])
ax2 = axs[0, 1].twinx()
ax2.plot(yf[1:], nd_y[1:], 'r--', linewidth=0.5)
ax2.get_yaxis().set_visible(False)
axs[0, 1].set_title('Y FFT')
axs[0, 1].set_xlabel('Frequency (1/m)')
axs[0, 1].set_ylabel('Amplitude')

# Plot z_fft
axs[1, 0].plot(zf[1:], np.abs(z_fft)[1:])
ax2 = axs[1, 0].twinx()
ax2.plot(zf[1:], nd_z[1:], 'r--', linewidth=0.5)
ax2.get_yaxis().set_visible(False)
axs[1, 0].set_title('Z FFT')
axs[1, 0].set_xlabel('Frequency (1/m)')
axs[1, 0].set_ylabel('Amplitude')

# Plot t_fft
axs[1, 1].plot(tf[1:], np.abs(t_fft)[1:])
ax2 = axs[1, 1].twinx()
ax2.plot(tf[1:], nd_t[1:], 'r--', linewidth=0.5)
ax2.get_yaxis().set_visible(False)
axs[1, 1].set_title('T FFT')
axs[1, 1].set_xlabel('Frequency (1/s)')
axs[1, 1].set_ylabel('Amplitude')

plt.savefig("/content/drive/MyDrive/MA/DA_CASE01/fft_nd_overlay.png")