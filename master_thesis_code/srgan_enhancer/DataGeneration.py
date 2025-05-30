import sys


sys.path.append('/scratch/jpelz/ma-pinns/_project/src')

import torch
import numpy as np

from scipy.stats.qmc import Halton, LatinHypercube, scale

from DaFlowPINN.model.core import load_predictable

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os




def compute_block(center, length, points_per_edge, alpha, beta, gamma, lb = None, ub = None):
    all_blocks = []

    if len(length) == 1:
        length = np.repeat(length, len(center))

    for c, a, b, g, l in zip(center, alpha, beta, gamma, length):
        # Create a grid of points
        linspace = np.linspace(-l / 2, l / 2, points_per_edge)
        x, y, z = np.meshgrid(linspace, linspace, linspace)
        points = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T

        # Rotation matrices
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(a), -np.sin(a)],
                       [0, np.sin(a), np.cos(a)]])
        
        Ry = np.array([[np.cos(b), 0, np.sin(b)],
                       [0, 1, 0],
                       [-np.sin(b), 0, np.cos(b)]])
        
        Rz = np.array([[np.cos(g), -np.sin(g), 0],
                       [np.sin(g), np.cos(g), 0],
                       [0, 0, 1]])

        # Combined rotation matrix
        R = Rz @ Ry @ Rx

        # Rotate points
        rotated_points = points @ R.T

        # Translate points to the center
        translated_points = rotated_points + c

        # Move if one point is outside the limits
        for i in range(3):
            if lb is not None and np.min(translated_points[:, i]) < lb[i]:
                translated_points[:, i] = translated_points[:, i] - np.min(translated_points[:, i]) + lb[i]
            if ub is not None and np.max(translated_points[:, i]) > ub[i]:
                translated_points[:, i] = translated_points[:, i] - np.max(translated_points[:, i]) + ub[i]

        all_blocks.append(translated_points.reshape(points_per_edge, points_per_edge, points_per_edge, 3))

    return np.array(all_blocks)

def plot_points(points, lb, ub, velocity = None, HR = False):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    size = 5 if HR else 20

    ax.scatter(points[:,:,:,:,0], points[:,:,:,:,1], points[:,:,:,:,2], c=velocity, cmap='jet', s=size, edgecolors='none', vmin=0, vmax=0.25)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if velocity is not None:
        cbar = plt.colorbar(ax.collections[0], ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('v_mag (m/s)')

    ax.set_xlim([lb[0], ub[0]])
    ax.set_ylim([lb[1], ub[1]])
    ax.set_zlim([lb[2], ub[2]])

    aspect_y = (ub[1] - lb[1]) / (ub[0] - lb[0])
    aspect_z = (ub[2] - lb[2]) / (ub[0] - lb[0])

    ax.set_box_aspect([1,aspect_y,aspect_z])  # Equal scaling


def generate_blocks(n_blocks, res_LR, size_LR, upscaling, lb, ub):

    res_HR = res_LR * upscaling
    size_HR = size_LR * upscaling

    length = res_LR * size_LR

    # lb = [lb[0]+length/2, lb[1]+length/2, lb[2]+length/2, 0, 0, 0]
    # ub = [ub[0]-length/2, ub[1]-length/2, ub[2]-length/2, 2*np.pi, 2*np.pi, 2*np.pi]
    
    lb = [lb[0], lb[1], lb[2], 0, 0, 0]
    ub = [ub[0], ub[1], ub[2], 2*np.pi, 2*np.pi, 2*np.pi]

    sampler = Halton(6)
    points = sampler.random(n_blocks)
    points = scale(points, lb, ub)

    LR_blocks = compute_block(points[:, :3], length-res_LR, size_LR, points[:, 3], points[:, 4], points[:, 5])
    HR_blocks = compute_block(points[:, :3], length-res_HR, size_HR, points[:, 3], points[:, 4], points[:, 5])

    return LR_blocks, HR_blocks


def main(N = 1000, validation = 0.2):

    RES_LR = np.array((9.4E-3, 5.4E-3, 2E-3)) #min
    RES_LR_max = np.array((10E-3, 9.4E-3, 5.4E-3)) #max
    size_LR = 5 #n*n*n

    lb = [0,0,0]
    ub = [0.1, 0.1, 0.1]

    block_length = size_LR * RES_LR
    blocks_per_length = 1/(block_length / 0.1)
    data_ratios = blocks_per_length / np.sum(blocks_per_length)


    upscaling = 2

    file_paths = ["/scratch/jpelz/ma-pinns/DA_Case_Time_Series/DA_CASE01_p010",
                 "/scratch/jpelz/ma-pinns/DA_Case_Time_Series/DA_CASE01_p050",
                 "/scratch/jpelz/ma-pinns/DA_Case_Time_Series/DA_CASE01_p200"]
    
    output_path = "/scratch/jpelz/srgan/TrainingData_DA01/distributed"

    for part in range(2):
        if part == 0:
            n_blocks_total = N * (1-validation)
        else:
            n_blocks_total = N * validation

        y_LR =[]
        y_HR = []

        condition_vector = []

        for res_LR, res_LR_max, file_path, data_ratio in zip(RES_LR, RES_LR_max, file_paths, data_ratios):
            blocks_total = int(n_blocks_total * data_ratio)


            # Search for all files in the folder with the ending _predictable.pt
            pinn_files = glob.glob(f"{file_path}/*_predictable.pt")

            # Print the found files
            print("Found PINN files:", pinn_files)

            # Load all found PINN files and extract the number before _predictable
            trainedPINNs = []
            pinn_numbers = []
            trainedPINNs = [load_predictable(file) for file in pinn_files]
            pinn_numbers = [int(file.split('_')[-2]) for file in pinn_files]


            dt = 5.916E-3
            nt = 3
            t_range = (nt-1)*dt
            t = np.array(pinn_numbers) * dt

            if blocks_total < len(trainedPINNs):
                trainedPINNs = np.random.choice(trainedPINNs, int(blocks_total), replace=False)

            blocks_per_pinn = int(blocks_total / len(trainedPINNs))




            print(f'Generating {blocks_total} Blocks in total...')
            print(f'Generating {blocks_per_pinn} Blocks per PINN...')


            

            import time
            start = time.time()

            for i in range(len(trainedPINNs)):

                res_sampler = LatinHypercube(1)
                res_LR_random = res_sampler.random(blocks_per_pinn)
                res_LR_random = scale(res_LR_random, [res_LR], [res_LR_max])

                condition_vector.append(2E-3 / res_LR_random)

                res_HR = res_LR_random * upscaling
                size_HR = size_LR * upscaling
            

                PINN = trainedPINNs[i]
                predict = PINN.get_forward_callable()

                blocks_LR, blocks_HR = generate_blocks(n_blocks=blocks_per_pinn, res_LR=res_LR_random, size_LR=size_LR, upscaling=upscaling, lb=lb, ub=ub)

                time_sampler = LatinHypercube(1)
                time_points = time_sampler.random(blocks_per_pinn)
                time_points = scale(time_points, [t[i]-t_range/2], [t[i]+t_range/2])

                time_vec_LR = np.ones((blocks_per_pinn, size_LR, size_LR, size_LR, 1)) * time_points[:, np.newaxis, np.newaxis, np.newaxis, :]
                time_vec_HR = np.ones((blocks_per_pinn, size_HR, size_HR, size_HR, 1)) * time_points[:, np.newaxis, np.newaxis, np.newaxis, :]

                x_LR = np.concatenate([blocks_LR, time_vec_LR], axis=-1)
                x_HR = np.concatenate([blocks_HR, time_vec_HR], axis=-1)

                x_LR_vec = x_LR.reshape(-1, 4)
                x_HR_vec = x_HR.reshape(-1, 4)

                with torch.no_grad():
                    
                    y_LR_vec = predict(torch.from_numpy(x_LR_vec).float())
                    y_HR_vec = predict(torch.from_numpy(x_HR_vec).float())

                y_LR_np = y_LR_vec.reshape(x_LR.shape).detach().numpy()
                y_HR_np = y_HR_vec.reshape(x_HR.shape).detach().numpy()

                y_LR.append(np.moveaxis(y_LR_np, -1, 1))
                y_HR.append(np.moveaxis(y_HR_np, -1, 1))
                
                
                print(f"Generated {(i+1)*blocks_per_pinn} of {blocks_total} Blocks...    {(i+1)*blocks_per_pinn*100/blocks_total} % done")

            

        y_LR = np.concatenate(y_LR, axis=0)
        y_HR = np.concatenate(y_HR, axis=0)
        condition_vector = np.concatenate(condition_vector, axis=0)

        if part == 0:
            out_path = f"{output_path}/training"
        else:
            out_path = f"{output_path}/validation"

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        np.save(f"{out_path}/y_n{n_blocks_total}_LR.npy", y_LR)
        np.save(f"{out_path}/y_n{n_blocks_total}_HR.npy", y_HR)
        np.save(f"{out_path}/cond_vec_n{n_blocks_total}.npy", condition_vector)


        print("Finished after ", time.time()-start, "seconds")


def generate_condition_vector(N = 1000, validation = 0.2):

    RES_LR = np.array((10E-3, 5E-3, 2.5E-3))
    size_LR = 5 #n*n*n

    lb = [0,0,0]
    ub = [0.1, 0.1, 0.1]

    block_length = size_LR * RES_LR
    blocks_per_length = 1/(block_length / 0.1)
    data_ratios = blocks_per_length / np.sum(blocks_per_length)


    upscaling = 2

    file_paths = ["/scratch/jpelz/ma-pinns/DA_Case_Time_Series/DA_CASE01_p010",
                 "/scratch/jpelz/ma-pinns/DA_Case_Time_Series/DA_CASE01_p050",
                 "/scratch/jpelz/ma-pinns/DA_Case_Time_Series/DA_CASE01_p200"]
    
    output_path = "/scratch/jpelz/srgan/TrainingData_DA01/mo"

    conditions = 2E-3 / RES_LR

    for part in range(2):
        if part == 0:
            n_blocks_total = N * (1-validation)
        else:
            n_blocks_total = N * validation

        y_LR =[]
        y_HR = []

        condition_vector = []

        for res_LR, file_path, data_ratio, condition in zip(RES_LR, file_paths, data_ratios, conditions):
            blocks_total = int(n_blocks_total * data_ratio)

            # Search for all files in the folder with the ending _predictable.pt
            pinn_files = glob.glob(f"{file_path}/*_predictable.pt")

            # Print the found files
            print("Found PINN files:", pinn_files)

            # Load all found PINN files and extract the number before _predictable
            trainedPINNs = []
            pinn_numbers = []
            trainedPINNs = [load_predictable(file) for file in pinn_files]
            pinn_numbers = [int(file.split('_')[-2]) for file in pinn_files]


            dt = 5.916E-3
            nt = 3
            t_range = (nt-1)*dt
            t = np.array(pinn_numbers) * dt

            if blocks_total < len(trainedPINNs):
                trainedPINNs = np.random.choice(trainedPINNs, int(blocks_total), replace=False)

            blocks_per_pinn = int(blocks_total / len(trainedPINNs))

            condition_vector.append(np.ones(blocks_per_pinn*len(trainedPINNs)) * condition)

        if part == 0:
            out_path = f"{output_path}/training"
        else:
            out_path = f"{output_path}/validation"

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        condition_vector = np.concatenate(condition_vector, axis=0)

        np.save(f"{out_path}/cond_vec_n{n_blocks_total}.npy", condition_vector)




if __name__ == '__main__':

    total_blocks = [1E3, 5E3, 1E4, 5E4, 1E5, 2.5E5]
    for blocks in total_blocks:
        main(N=int(blocks), validation=0.2)
        #generate_condition_vector(N=int(blocks), validation=0.2)