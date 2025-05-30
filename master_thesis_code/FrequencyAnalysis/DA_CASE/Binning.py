import numpy as np
import scipy
import scipy.signal
import scipy.fftpack
from scipy import stats

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load the data
    print("loading data")
    data = np.loadtxt("/content/drive/MyDrive/MA/DA_CASE01/velocity_files/DA_CASE01_TR_ppp_0_200_velocities.dat", skiprows=1)
    print("done")
    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]
    t = data[:, 4]

    u = data[:, 5]
    v = data[:, 6]
    w = data[:, 7]

    l_scale = 0.1
    u_scale = 0.2
    t_scale = l_scale / u_scale

    mag = np.sqrt(u**2 + v**2 + w**2)
    mag = mag / u_scale

    x = x / l_scale
    y = y / l_scale
    z = z / l_scale
    t = t / t_scale

    x_max = x.max()
    x_min = x.min()
    y_max = y.max()
    y_min = y.min()
    z_max = z.max()
    z_min = z.min()
    t_max = t.max()
    t_min = t.min()

    n_split = 50
    nt = len(np.unique(t))
    
    x_width = (x_max - x_min) / n_split
    y_width = (y_max - y_min) / n_split
    z_width = (z_max - z_min) / n_split
    t_width = (np.unique(t).max() - np.unique(t).min()) / nt

    x_bins = np.linspace(x_min, x_max, n_split)
    y_bins = np.linspace(y_min, y_max, n_split)
    z_bins = np.linspace(z_min, z_max, n_split)
    t_bins = np.unique(t)

    x_idx = np.digitize(x, x_bins)
    y_idx = np.digitize(y, y_bins)
    z_idx = np.digitize(z, z_bins)

    binning = stats.binned_statistic_dd([x, y, z, t], mag, statistic='mean', bins=[x_bins, y_bins, z_bins, t_bins])
    
    interpolator = scipy.interpolate.NearestNDInterpolator(list(zip(x, y, z, t)), mag)
    x_grid, y_grid, z_grid, t_grid = np.meshgrid(x_bins, y_bins, z_bins, t_bins, indexing='ij')
    mag_grid = interpolator(x_grid, y_grid, z_grid, t_grid)
    

    mag_binned = binning.statistic
    print(mag_binned.shape)

    fig = plt.figure()
    plt.figure(figsize=(8, 6))
    plt.imshow(mag_grid[:,:,5,24].T, origin='lower', cmap='jet', interpolation='nearest', extent=[0, 10, 0, 10])
    plt.colorbar(label='mag (m/s)')
    plt.title('Binned Flow Field with Squares')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    #save binned field

    np.save("/content/drive/MyDrive/MA/DA_CASE01/binned_field_200.npy", mag_binned)
    np.save("/content/drive/MyDrive/MA/DA_CASE01/binned_field_filled_200.npy", mag_grid)