import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.stats as stats


import sys


sys.path.append('G:\Meine Ablage\MA')
sys.path.append('G:\Meine Ablage\MA\_packages')
sys.path.append("G:\Meine Ablage\MA\_SuperResolution")

from compute_tke import compute_tke_spectrum

from SRGAN.data import min_max_normalize
from SRGAN.architectures import Generator

from DaFlowPINN.model.core import load_predictable

def compute_blocks(center, length, points_per_edge):
    all_blocks = []

    for c in center:
        # Create a grid of points
        linspace = np.linspace(-length / 2, length / 2, points_per_edge)
        x, y, z = np.meshgrid(linspace, linspace, linspace, indexing='ij')
        points = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T

 
        # Translate points to the center
        translated_points = points + c

        all_blocks.append(translated_points.reshape(points_per_edge, points_per_edge, points_per_edge, 3))

    return np.array(all_blocks)


def upsample_field(PINN_file, generator_file, res_LR, size_LR=5, upscaling=2, overlap_HR = 4, lb = [0, 0.1E-3, 0.1E-3], ub = [0.1, 99.1E-3, 99.1E-3], t = 24*5.916E-3):
    o_HR = overlap_HR
    length = res_LR * size_LR

    res_HR = res_LR / upscaling
    size_HR = size_LR * upscaling

    nx_LR = int(np.round((ub[0]-lb[0])/res_LR,0)+1)
    ny_LR = int(np.round((ub[1]-lb[1])/res_LR,0)+1)
    nz_LR = int(np.round((ub[2]-lb[2])/res_LR,0)+1)


    nx_HR = int(np.round((ub[0]-lb[0])/res_HR,0)+1)
    ny_HR = int(np.round((ub[1]-lb[1])/res_HR,0)+1)
    nz_HR = int(np.round((ub[2]-lb[2])/res_HR,0)+1)


    x_mids = [lb[0]+length/2-res_HR/2]
    y_mids = [lb[1]+length/2-res_HR/2]
    z_mids = [lb[2]+length/2-res_HR/2]

    while x_mids[-1]+length/2 < ub[0]:
        x_mids.append(x_mids[-1]+length-res_HR*o_HR)
    while y_mids[-1]+length/2 < ub[1]:
        y_mids.append(y_mids[-1]+length-res_HR*o_HR)
    while z_mids[-1]+length/2 < ub[2]:
        z_mids.append(z_mids[-1]+length-res_HR*o_HR)

    x_mids = np.array(x_mids)
    y_mids = np.array(y_mids)
    z_mids = np.array(z_mids)

    y_mids, x_mids, z_mids = np.meshgrid(y_mids, x_mids, z_mids)

    points = np.column_stack((x_mids.flatten(), y_mids.flatten(), z_mids.flatten()))

    blocks_LR = compute_blocks(points, length-res_LR, size_LR)
    blocks_HR = compute_blocks(points, length-res_HR, size_HR)

    time_vec_LR = np.expand_dims(np.ones(blocks_LR.shape[0:4]) * t, axis=-1)
    time_vec_HR = np.expand_dims(np.ones(blocks_HR.shape[0:4]) * t, axis=-1)

    blocks_LR = np.concatenate((blocks_LR, time_vec_LR), axis=-1)
    blocks_HR = np.concatenate((blocks_HR, time_vec_HR), axis=-1)

    x_LR_vec = blocks_LR.reshape(-1, 4)
    x_HR_vec = blocks_HR.reshape(-1, 4)

    with torch.no_grad():
        PINN = load_predictable(PINN_file)
        predict = PINN.get_forward_callable()

        y_LR_vec = predict(torch.from_numpy(x_LR_vec).float())
        y_LR_blocks = torch.moveaxis(y_LR_vec.reshape(blocks_LR.shape), -1, 1)

        #Superresolution
        generator = Generator(n_residual_blocks=3)
        generator.load_state_dict(torch.load(generator_file, map_location='cpu', weights_only=True))
        generator.eval()

        y_LR_blocks_norm, denormalize = min_max_normalize(y_LR_blocks, return_denormalization=True)
        y_HR_blocks = generator(y_LR_blocks_norm.float())
        y_HR_blocks = denormalize(y_HR_blocks).detach().numpy()
        

    y_LR_blocks = torch.moveaxis(y_LR_blocks, 1, -1).reshape(-1, 4)
    #y_LR_blocks = y_LR_blocks[:,o_LR:-o_LR,o_LR:-o_LR,o_LR:-o_LR,:]
    #y_LR_field = y_LR_blocks.reshape(-1, 4)[map_LR,:]

    y_HR_blocks = np.moveaxis(y_HR_blocks, 1, -1).reshape(-1, 4)
    # y_HR_blocks = y_HR_blocks[:,o_HR:-o_HR,o_HR:-o_HR,o_HR:-o_HR,:]
    # y_HR_field = y_HR_blocks.reshape(-1, 4)[map_HR,:]


    x_bins = np.linspace(lb[0]-res_HR/2, ub[0]+res_HR/2, int(nx_HR+1))
    y_bins = np.linspace(lb[1]-res_HR/2, ub[1]+res_HR/2, int(ny_HR+1))
    z_bins = np.linspace(lb[2]-res_HR/2, ub[2]+res_HR/2, int(nz_HR+1))

    binning = stats.binned_statistic_dd(x_HR_vec[:,0:3], [y_HR_blocks[:,0], y_HR_blocks[:,1], y_HR_blocks[:,2], y_HR_blocks[:,3]], statistic='mean', bins=[x_bins, y_bins, z_bins])

    y_HR_field = binning.statistic
    y_HR_field = np.moveaxis(y_HR_field, 0, -1)


    #For reference:
    #LR grid
    y_LR, x_LR, z_LR, t_LR = np.meshgrid(
        np.linspace(lb[1], ub[1], ny_LR),
        np.linspace(lb[0], ub[0], nx_LR),
        np.linspace(lb[2], ub[2], nz_LR),
        t
    )
  
    #HR grid
    y_HR, x_HR, z_HR, t_HR = np.meshgrid(
        np.linspace(lb[1], ub[1], ny_HR),
        np.linspace(lb[0], ub[0], nx_HR),
        np.linspace(lb[2], ub[2], nz_HR),
        t
    )

    x_LR_vec = np.column_stack((x_LR.flatten(), y_LR.flatten(), z_LR.flatten(), t_LR.flatten()))
    x_HR_vec = np.column_stack((x_HR.flatten(), y_HR.flatten(), z_HR.flatten(), t_HR.flatten()))

    with torch.no_grad():
        y_LR_vec = predict(torch.from_numpy(x_LR_vec).float()).detach().numpy()
        y_HR_vec = predict(torch.from_numpy(x_HR_vec).float()).detach().numpy()
        y_LR_field = y_LR_vec.reshape((x_LR.shape[0], x_LR.shape[1], x_LR.shape[2], 4))
        y_HR_ref_field = y_HR_vec.reshape((x_HR.shape[0], x_HR.shape[1], x_HR.shape[2], 4))


    return y_LR_field, y_HR_field, y_HR_ref_field

def upsample_fieldV2(PINN_file, generator_file, res_LR, size_LR=5, upscaling=2, overlap_HR = 4, lb = [0, 0.1E-3, 0.1E-3], ub = [0.1, 99.1E-3, 99.1E-3], t = 24*5.916E-3, avg = False):
    o_HR = int(np.round(overlap_HR,0))
    length = res_LR * size_LR

    if avg:
        avg = 1
    else:
        avg = 0

    res_HR = res_LR / upscaling
    size_HR = size_LR * upscaling

    nx_LR = int(np.round((ub[0]-lb[0])/res_LR,0)+1)
    ny_LR = int(np.round((ub[1]-lb[1])/res_LR,0)+1)
    nz_LR = int(np.round((ub[2]-lb[2])/res_LR,0)+1)


    nx_HR = int(np.round((ub[0]-lb[0])/res_HR,0)+1)
    ny_HR = int(np.round((ub[1]-lb[1])/res_HR,0)+1)
    nz_HR = int(np.round((ub[2]-lb[2])/res_HR,0)+1)


    x_mids = [lb[0]+length/2-res_HR/2]
    y_mids = [lb[1]+length/2-res_HR/2]
    z_mids = [lb[2]+length/2-res_HR/2]

    while x_mids[-1]+length/2 < ub[0]:
        x_mids.append(x_mids[-1]+length-res_HR*(2*o_HR+avg))
    while y_mids[-1]+length/2 < ub[1]:
        y_mids.append(y_mids[-1]+length-res_HR*(2*o_HR+avg))
    while z_mids[-1]+length/2 < ub[2]:
        z_mids.append(z_mids[-1]+length-res_HR*(2*o_HR+avg))

    x_mids = np.array(x_mids)
    y_mids = np.array(y_mids)
    z_mids = np.array(z_mids)

    y_mids, x_mids, z_mids = np.meshgrid(y_mids, x_mids, z_mids)

    points = np.column_stack((x_mids.flatten(), y_mids.flatten(), z_mids.flatten()))

    blocks_LR = compute_blocks(points, length-res_LR, size_LR)
    blocks_HR = compute_blocks(points, length-res_HR, size_HR)

    time_vec_LR = np.expand_dims(np.ones(blocks_LR.shape[0:4]) * t, axis=-1)
    time_vec_HR = np.expand_dims(np.ones(blocks_HR.shape[0:4]) * t, axis=-1)

    blocks_LR = np.concatenate((blocks_LR, time_vec_LR), axis=-1)
    blocks_HR = np.concatenate((blocks_HR, time_vec_HR), axis=-1)

    blocks_HR_cut = blocks_HR[:,o_HR:-o_HR,o_HR:-o_HR,o_HR:-o_HR,:].reshape(-1, 4)


    x_LR_vec = blocks_LR.reshape(-1, 4)
    x_HR_vec = blocks_HR.reshape(-1, 4)

    with torch.no_grad():
        PINN = load_predictable(PINN_file)
        predict = PINN.get_forward_callable()

        y_LR_vec = predict(torch.from_numpy(x_LR_vec).float())
        y_LR_blocks = torch.moveaxis(y_LR_vec.reshape(blocks_LR.shape), -1, 1)

        #Superresolution
        generator = Generator(n_residual_blocks=3)
        generator.load_state_dict(torch.load(generator_file, map_location='cpu', weights_only=True))
        generator.eval()

        y_LR_blocks_norm, denormalize = min_max_normalize(y_LR_blocks, return_denormalization=True)
        y_HR_blocks = generator(y_LR_blocks_norm.float())
        y_HR_blocks = denormalize(y_HR_blocks).detach().numpy()
    
    

    y_HR_blocks = np.moveaxis(y_HR_blocks, 1, -1)
    print(y_HR_blocks.shape)
    

    x_bins = np.linspace(lb[0]-res_HR/2, ub[0]+res_HR/2, int(nx_HR+1))
    y_bins = np.linspace(lb[1]-res_HR/2, ub[1]+res_HR/2, int(ny_HR+1))
    z_bins = np.linspace(lb[2]-res_HR/2, ub[2]+res_HR/2, int(nz_HR+1))

    x_HR_blocks_filt = blocks_HR[:,o_HR:-o_HR,o_HR:-o_HR,o_HR:-o_HR,:].reshape(-1, 4)
    y_HR_blocks_filt = y_HR_blocks[:,o_HR:-o_HR,o_HR:-o_HR,o_HR:-o_HR,:].reshape(-1, 4)

    
    binning = stats.binned_statistic_dd(x_HR_blocks_filt[:,0:3], [y_HR_blocks_filt[:,0], y_HR_blocks_filt[:,1], y_HR_blocks_filt[:,2], y_HR_blocks_filt[:,3]], statistic='mean', bins=[x_bins, y_bins, z_bins])
    y_HR_field = binning.statistic
    nan_mask = np.isnan(y_HR_field)


    #Fill X
    x_HR_blocks_filt = blocks_HR[:,:,o_HR:-o_HR,o_HR:-o_HR,:].reshape(-1, 4)
    y_HR_blocks_filt = y_HR_blocks[:,:,o_HR:-o_HR,o_HR:-o_HR,:].reshape(-1, 4)
    binning = stats.binned_statistic_dd(x_HR_blocks_filt[:,0:3], [y_HR_blocks_filt[:,0], y_HR_blocks_filt[:,1], y_HR_blocks_filt[:,2], y_HR_blocks_filt[:,3]], statistic='mean', bins=[x_bins, y_bins, z_bins])
    y_HR_field[nan_mask] = binning.statistic[nan_mask]
    nan_mask = np.isnan(y_HR_field)

    #Fill Y
    x_HR_blocks_filt = blocks_HR[:,o_HR:-o_HR,:,o_HR:-o_HR,:].reshape(-1, 4)
    y_HR_blocks_filt = y_HR_blocks[:,o_HR:-o_HR,:,o_HR:-o_HR,:].reshape(-1, 4)
    binning = stats.binned_statistic_dd(x_HR_blocks_filt[:,0:3], [y_HR_blocks_filt[:,0], y_HR_blocks_filt[:,1], y_HR_blocks_filt[:,2], y_HR_blocks_filt[:,3]], statistic='mean', bins=[x_bins, y_bins, z_bins])
    y_HR_field[nan_mask] = binning.statistic[nan_mask]
    nan_mask = np.isnan(y_HR_field)

    #Fill Z
    x_HR_blocks_filt = blocks_HR[:,o_HR:-o_HR,o_HR:-o_HR,:,:].reshape(-1, 4)
    y_HR_blocks_filt = y_HR_blocks[:,o_HR:-o_HR,o_HR:-o_HR,:,:].reshape(-1, 4)
    binning = stats.binned_statistic_dd(x_HR_blocks_filt[:,0:3], [y_HR_blocks_filt[:,0], y_HR_blocks_filt[:,1], y_HR_blocks_filt[:,2], y_HR_blocks_filt[:,3]], statistic='mean', bins=[x_bins, y_bins, z_bins])
    y_HR_field[nan_mask] = binning.statistic[nan_mask]
    nan_mask = np.isnan(y_HR_field)

    #Fill XY Edges
    x_HR_blocks_filt = blocks_HR[:,:,:,o_HR:-o_HR,:].reshape(-1, 4)
    y_HR_blocks_filt = y_HR_blocks[:,:,:,o_HR:-o_HR,:].reshape(-1, 4)
    binning = stats.binned_statistic_dd(x_HR_blocks_filt[:,0:3], [y_HR_blocks_filt[:,0], y_HR_blocks_filt[:,1], y_HR_blocks_filt[:,2], y_HR_blocks_filt[:,3]], statistic='mean', bins=[x_bins, y_bins, z_bins])
    y_HR_field[nan_mask] = binning.statistic[nan_mask]
    nan_mask = np.isnan(y_HR_field)

    #Fill XZ Edges
    x_HR_blocks_filt = blocks_HR[:,:,o_HR:-o_HR,:,:].reshape(-1, 4)
    y_HR_blocks_filt = y_HR_blocks[:,:,o_HR:-o_HR,:,:].reshape(-1, 4)
    binning = stats.binned_statistic_dd(x_HR_blocks_filt[:,0:3], [y_HR_blocks_filt[:,0], y_HR_blocks_filt[:,1], y_HR_blocks_filt[:,2], y_HR_blocks_filt[:,3]], statistic='mean', bins=[x_bins, y_bins, z_bins])
    y_HR_field[nan_mask] = binning.statistic[nan_mask]
    nan_mask = np.isnan(y_HR_field)

    #Fill YZ Edges
    x_HR_blocks_filt = blocks_HR[:,:,:,o_HR:-o_HR,:].reshape(-1, 4)
    y_HR_blocks_filt = y_HR_blocks[:,:,:,o_HR:-o_HR,:].reshape(-1, 4)
    binning = stats.binned_statistic_dd(x_HR_blocks_filt[:,0:3], [y_HR_blocks_filt[:,0], y_HR_blocks_filt[:,1], y_HR_blocks_filt[:,2], y_HR_blocks_filt[:,3]], statistic='mean', bins=[x_bins, y_bins, z_bins])
    y_HR_field[nan_mask] = binning.statistic[nan_mask]
    nan_mask = np.isnan(y_HR_field)

    #Fill Outer Edges
    x_HR_blocks_filt = blocks_HR[:,:,:,:,:].reshape(-1, 4)
    y_HR_blocks_filt = y_HR_blocks[:,:,:,:,:].reshape(-1, 4)
    binning = stats.binned_statistic_dd(x_HR_blocks_filt[:,0:3], [y_HR_blocks_filt[:,0], y_HR_blocks_filt[:,1], y_HR_blocks_filt[:,2], y_HR_blocks_filt[:,3]], statistic='mean', bins=[x_bins, y_bins, z_bins])
    y_HR_field[nan_mask] = binning.statistic[nan_mask]
    nan_mask = np.isnan(y_HR_field)





    y_HR_field = np.moveaxis(y_HR_field, 0, -1)


    #For reference:
    #LR grid
    y_LR, x_LR, z_LR, t_LR = np.meshgrid(
        np.linspace(lb[1], ub[1], ny_LR),
        np.linspace(lb[0], ub[0], nx_LR),
        np.linspace(lb[2], ub[2], nz_LR),
        t
    )
  
    #HR grid
    y_HR, x_HR, z_HR, t_HR = np.meshgrid(
        np.linspace(lb[1], ub[1], ny_HR),
        np.linspace(lb[0], ub[0], nx_HR),
        np.linspace(lb[2], ub[2], nz_HR),
        t
    )

    x_LR_vec = np.column_stack((x_LR.flatten(), y_LR.flatten(), z_LR.flatten(), t_LR.flatten()))
    x_HR_vec = np.column_stack((x_HR.flatten(), y_HR.flatten(), z_HR.flatten(), t_HR.flatten()))

    with torch.no_grad():
        y_LR_vec = predict(torch.from_numpy(x_LR_vec).float()).detach().numpy()
        y_HR_vec = predict(torch.from_numpy(x_HR_vec).float()).detach().numpy()
        y_LR_field = y_LR_vec.reshape((x_LR.shape[0], x_LR.shape[1], x_LR.shape[2], 4))
        y_HR_ref_field = y_HR_vec.reshape((x_HR.shape[0], x_HR.shape[1], x_HR.shape[2], 4))


    return y_LR_field, y_HR_field, y_HR_ref_field

def evaluate_field(pinn_file, generator_file, res_LR, size_LR=5, upscaling=2, overlap_HR = 4, lb = [0, 0.1E-3, 0.1E-3], ub = [0.1, 99.1E-3, 99.1E-3], t = 24*5.916E-3, name="", V2 = False, avg = False):

    if V2:
        y_LR_field, y_HR_field, y_HR_ref_field = upsample_fieldV2(pinn_file, generator_file, res_LR, size_LR=size_LR, upscaling=upscaling, overlap_HR=overlap_HR, lb=lb, ub=ub, t=t, avg=avg)
    else:
        y_LR_field, y_HR_field, y_HR_ref_field = upsample_field(pinn_file, generator_file, res_LR, size_LR=size_LR, upscaling=upscaling, overlap_HR=overlap_HR, lb=lb, ub=ub, t=t)
    
    
    mae = np.mean(np.abs(y_HR_field - y_HR_ref_field), axis=(0, 1, 2))
    rmse = np.sqrt(np.mean((y_HR_field - y_HR_ref_field) ** 2, axis=(0, 1, 2)))

    _, wave_LR, tke_LR = compute_tke_spectrum(y_LR_field[:, :, :, 0], y_LR_field[:, :, :, 1], y_LR_field[:, :, :, 2], lx=0.1, ly=0.1, lz=(ub[2]-lb[2]), smooth=False)
    _, wave_HR, tke_HR = compute_tke_spectrum(y_HR_field[:, :, :, 0], y_HR_field[:, :, :, 1], y_HR_field[:, :, :, 2], lx=0.1, ly=0.1, lz=(ub[2]-lb[2]), smooth=False)
    _, wave_HR_ref, tke_HR_ref = compute_tke_spectrum(y_HR_ref_field[:, :, :, 0], y_HR_ref_field[:, :, :, 1], y_HR_ref_field[:, :, :, 2], lx=0.1, ly=0.1, lz=(ub[2]-lb[2]), smooth=False)


    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    pic1 = axs[0].imshow(y_LR_field[:, :, int(y_LR_field.shape[2]/2), 0].T, cmap='jet', origin='lower', extent=[0, 0.1, 0, 0.1], vmin=0, vmax=0.26)
    axs[0].set_title('LR direct', fontsize=10)
    pic2 = axs[1].imshow(y_HR_field[:, :, int(y_HR_field.shape[2]/2), 0].T, cmap='jet', origin='lower', extent=[0, 0.1, 0, 0.1], vmin=0, vmax=0.26)
    axs[1].set_title('HR - Superresolution', fontsize=10)
    pic3 = axs[2].imshow(y_HR_ref_field[:, :, int(y_HR_field.shape[2]/2), 0].T, cmap='jet', origin='lower', extent=[0, 0.1, 0, 0.1], vmin=0, vmax=0.26)
    axs[2].set_title('HR direct', fontsize=10)
    cbar3 = fig.colorbar(pic3, ax=axs[2], fraction=0.046, pad=0.04)
    fig.suptitle('XY-Section of u in m/s', fontsize=16)
    plt.savefig(f'xy_section_LR{res_LR*1000}_{name}.png')

    return mae, rmse, wave_LR, tke_LR, wave_HR, tke_HR, wave_HR_ref, tke_HR_ref

if __name__ == "__main__":
    lb = [0, 0.0001, 0.04]
    ub = [0.1, 0.0991, 0.06]

    t = 24*5.916E-3

    res_LR = [10E-3, 5E-3, 2.5E-3]
    size_LR = 5  # n*n*n
    upscaling = 2

    o_HR = 6

    PINN_files = ['G:\Meine Ablage\MA\_SuperResolution\Models\PINNs\DA_CASE01_p010_21_predictable.pt',
                    'G:\Meine Ablage\MA\_SuperResolution\Models\PINNs\DA_CASE01_p050_21_predictable.pt',
                    'G:\Meine Ablage\MA\_SuperResolution\Models\PINNs\DA_CASE01_p200_21_predictable.pt']
    
    generator_base = "G:\Meine Ablage\MA\_SuperResolution\Models\Generator\generator_{}_990.pt"
    sample_sizes = ["5000", "10000", "50000", "100000", "250000"]

    mae10_list = []
    rmse10_list = []
    mae50_list = []
    rmse50_list = []
    mae200_list = []
    rmse200_list = []

    tke_HR10_list = []
    tke_HR50_list = []
    tke_HR200_list = []

    #sample_size = "250000"
    for sample_size in sample_sizes:
    #Overlapping = [0,2,4,6,8]
    #for o_HR in Overlapping:
        generator_file = generator_base.format(sample_size)
        print(f"Evaluating for {sample_size} samples")


        mae10, rmse10, wave_LR10, tke_LR10, wave_HR10, tke_HR10, wave_HR_ref10, tke_HR_ref10= evaluate_field(PINN_files[0], generator_file, res_LR[0], size_LR=size_LR, upscaling=upscaling, overlap_HR=o_HR, lb=lb, ub=ub, t=t, name=o_HR)
        mae50, rmse50, wave_LR50, tke_LR50, wave_HR50, tke_HR50, wave_HR_ref50, tke_HR_ref50 = evaluate_field(PINN_files[1], generator_file, res_LR[1], size_LR=size_LR, upscaling=upscaling, overlap_HR=o_HR, lb=lb, ub=ub, t=t, name=o_HR)
        mae200, rmse200, wave_LR200, tke_LR200, wave_HR200, tke_HR200, wave_HR_ref200, tke_HR_ref200 = evaluate_field(PINN_files[2], generator_file, res_LR[2], size_LR=size_LR, upscaling=upscaling, overlap_HR=o_HR, lb=lb, ub=ub, t=t, name=o_HR)


        mae10_list.append(mae10)
        rmse10_list.append(rmse10)
        mae50_list.append(mae50)
        rmse50_list.append(rmse50)
        mae200_list.append(mae200)
        rmse200_list.append(rmse200)
        tke_HR10_list.append(tke_HR10)
        tke_HR50_list.append(tke_HR50)
        tke_HR200_list.append(tke_HR200)

#Plot TKE spectra for different PINNs
#10 mm
fig = plt.figure(figsize=(10, 5))
plt.title("TKE Spectrum for LR = 10 mm")
plt.loglog(wave_LR10, tke_LR10, label='LR')
plt.loglog(wave_HR_ref10, tke_HR_ref10, label='HR ref')
for i in range(len(tke_HR10_list)):
    plt.loglog(wave_HR10, tke_HR10_list[i], label=f'SR {sample_sizes[i]}')
plt.xlabel('Wave number')
plt.ylabel('TKE')
plt.ylim(1E-8, 1E-5)
plt.legend()
plt.savefig('tke_spectrum_LR10_sample_size.png')

#5 mm
fig = plt.figure(figsize=(10, 5))
plt.title("TKE Spectrum for LR = 5 mm")
plt.loglog(wave_LR50, tke_LR50, label='LR')
plt.loglog(wave_HR_ref50, tke_HR_ref50, label='HR ref')
for i in range(len(tke_HR50_list)):
    plt.loglog(wave_HR50, tke_HR50_list[i], label=f'SR {sample_sizes[i]}')
plt.xlabel('Wave number')
plt.ylim(1E-8, 1E-5)
plt.ylabel('TKE')
plt.legend()
plt.savefig('tke_spectrum_LR5_sample_size.png')

#2.5 mm
fig = plt.figure(figsize=(10, 5))
plt.title("TKE Spectrum for LR = 2.5 mm")
plt.loglog(wave_LR200, tke_LR200, label='LR')
plt.loglog(wave_HR_ref200, tke_HR_ref200, label='HR ref')
for i in range(len(tke_HR200_list)):
    plt.loglog(wave_HR200, tke_HR200_list[i], label=f'SR {sample_sizes[i]}')
plt.xlabel('Wave number')
plt.ylim(1E-8, 1E-5)
plt.ylabel('TKE')
plt.legend()
plt.savefig('tke_spectrum_LR2.5_sample_size.png')




#Plot MAEs over Sample Size
sample_sizes = np.array([5000, 10000, 50000, 100000, 250000])
fig, axs = plt.subplots(1, 4, figsize=(10, 5))
axs[0].plot(sample_sizes, np.array(mae10_list)[:, 0], "o--",  label='LR10')
axs[0].plot(sample_sizes, np.array(mae50_list)[:, 0], "^--", label='LR5')
axs[0].plot(sample_sizes, np.array(mae200_list)[:, 0], "s--", label='LR2.5')
axs[0].set_title('MAE u')
axs[0].set_xlabel('Dataset Size')
axs[0].set_ylabel('MAE (m/s)')
axs[0].legend()

axs[1].plot(sample_sizes, np.array(mae10_list)[:, 1], "o--", label='LR10')
axs[1].plot(sample_sizes, np.array(mae50_list)[:, 1], "^--", label='LR5')
axs[1].plot(sample_sizes, np.array(mae200_list)[:, 1], "s--", label='LR2.5')
axs[1].set_title('MAE v')
axs[1].set_xlabel('Dataset Size')
axs[1].set_ylabel('MAE (m/s)')
axs[1].legend()

axs[2].plot(sample_sizes, np.array(mae10_list)[:, 2], "o--", label='LR10')
axs[2].plot(sample_sizes, np.array(mae50_list)[:, 2], "^--", label='LR5')
axs[2].plot(sample_sizes, np.array(mae200_list)[:, 2], "s--", label='LR2.5')
axs[2].set_title('MAE w')
axs[2].set_xlabel('Dataset Size')
axs[2].set_ylabel('MAE (m/s)')
axs[2].legend()

axs[3].plot(sample_sizes, np.array(mae10_list)[:, 3], "o--", label='LR10')
axs[3].plot(sample_sizes, np.array(mae50_list)[:, 3], "^--", label='LR5')
axs[3].plot(sample_sizes, np.array(mae200_list)[:, 3], "s--", label='LR2.5')
axs[3].set_title('MAE p')
axs[3].set_xlabel('Dataset Size')
axs[3].set_ylabel('MAE (Pa)')
axs[3].legend()

fig.suptitle('Mean Absolute Error (MAE) vs Sample Size', fontsize=16)
plt.tight_layout()
plt.savefig('mae_vs_overlap.png')

#Plot RMSEs over Sample Size
fig, axs = plt.subplots(1, 4, figsize=(10, 5))
axs[0].plot(sample_sizes, np.array(rmse10_list)[:, 0], "o--", label='LR10')
axs[0].plot(sample_sizes, np.array(rmse50_list)[:, 0], "^--", label='LR5')
axs[0].plot(sample_sizes, np.array(rmse200_list)[:, 0], "s--", label='LR2.5')
axs[0].set_title('RMSE u')
axs[0].set_xlabel('Dataset Size')
axs[0].set_ylabel('RMSE (m/s)')
axs[0].legend()

axs[1].plot(sample_sizes, np.array(rmse10_list)[:, 1], "o--", label='LR10')
axs[1].plot(sample_sizes, np.array(rmse50_list)[:, 1], "^--", label='LR5')
axs[1].plot(sample_sizes, np.array(rmse200_list)[:, 1], "s--", label='LR2.5')
axs[1].set_title('RMSE v')
axs[1].set_xlabel('Dataset Size')
axs[1].set_ylabel('RMSE (m/s)')
axs[1].legend()

axs[2].plot(sample_sizes, np.array(rmse10_list)[:, 2], "o--", label='LR10')
axs[2].plot(sample_sizes, np.array(rmse50_list)[:, 2], "^--", label='LR5')
axs[2].plot(sample_sizes, np.array(rmse200_list)[:, 2], "s--", label='LR2.5')
axs[2].set_title('RMSE w')
axs[2].set_xlabel('Dataset Size')
axs[2].set_ylabel('RMSE (m/s)')
axs[2].legend()

axs[3].plot(sample_sizes, np.array(rmse10_list)[:, 3], "o--", label='LR10')
axs[3].plot(sample_sizes, np.array(rmse50_list)[:, 3], "^--", label='LR5')
axs[3].plot(sample_sizes, np.array(rmse200_list)[:, 3], "s--", label='LR2.5')
axs[3].set_title('RMSE p')
axs[3].set_xlabel('Dataset Size')
axs[3].set_ylabel('RMSE (Pa)')
axs[3].legend()

fig.suptitle('Root Mean Square Error (RMSE) vs Dataset Size', fontsize=16)
plt.tight_layout()
plt.savefig('rmse_vs_overlap.png')


#write maes to file:
with open("mae_results_dataset.txt", "w") as f:
    f.write("Sample Size\tMAE u\tMAE v\tMAE w\tMAE p\n")
    for i in range(len(sample_sizes)):
        f.write(f"{sample_sizes[i]}\t{mae10_list[i][0]}\t{mae10_list[i][1]}\t{mae10_list[i][2]}\t{mae10_list[i][3]}\n")
        f.write(f"{sample_sizes[i]}\t{mae50_list[i][0]}\t{mae50_list[i][1]}\t{mae50_list[i][2]}\t{mae50_list[i][3]}\n")
        f.write(f"{sample_sizes[i]}\t{mae200_list[i][0]}\t{mae200_list[i][1]}\t{mae200_list[i][2]}\t{mae200_list[i][3]}\n")
        f.write("\n")
#write rmse to file:
with open("rmse_results_dataset.txt", "w") as f:
    f.write("Sample Size\tRMSE u\tRMSE v\tRMSE w\tRMSE p\n")
    for i in range(len(sample_sizes)):
        f.write(f"{sample_sizes[i]}\t{rmse10_list[i][0]}\t{rmse10_list[i][1]}\t{rmse10_list[i][2]}\t{rmse10_list[i][3]}\n")
        f.write(f"{sample_sizes[i]}\t{rmse50_list[i][0]}\t{rmse50_list[i][1]}\t{rmse50_list[i][2]}\t{rmse50_list[i][3]}\n")
        f.write(f"{sample_sizes[i]}\t{rmse200_list[i][0]}\t{rmse200_list[i][1]}\t{rmse200_list[i][2]}\t{rmse200_list[i][3]}\n")
        f.write("\n")


