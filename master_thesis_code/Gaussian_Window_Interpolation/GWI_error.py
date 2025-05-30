import numpy as np
from scipy.spatial import cKDTree
from DaFlowPINN.post.evaluation import evaluatePINN
import torch
import os

def gaussian_kernel(particles, velocities, sigma):
    sigma = sigma
    cutoff = 3 * sigma
    def f(points):



        points = points.numpy()
        points = points[:, :3]  # Ensure points are 3D


        U, V, W = velocities[:, 0], velocities[:, 1], velocities[:, 2]



        kdtree = cKDTree(particles)

        u_pred = np.zeros(len(points))
        v_pred = np.zeros(len(points))
        w_pred = np.zeros(len(points))


        for i in range(len(points[:,0])):
            # Find the indices of the points within the cutoff distance
            gp = points[i, :3]  # Get the current point
            indices = kdtree.query_ball_point(gp, cutoff)

            if not indices:
                distances, indices = kdtree.query(gp, k=1)
            
            if np.isscalar(indices):
                indices = [indices]
            else:
                indices = list(indices)

        
            # Compute the weights using a Gaussian function
            distances = np.linalg.norm(particles[indices] - gp, axis=1)
            weights = np.exp(-0.5 * (distances / sigma) ** 2)
            if np.sum(weights) == 0:
                weights = np.ones_like(weights)
            weights /= np.sum(weights)
        
            u_pred[i] = np.average(U[indices], weights=weights)
            v_pred[i] = np.average(V[indices], weights=weights)
            w_pred[i] = np.average(W[indices], weights=weights)

        v_pred= np.column_stack((u_pred, v_pred, w_pred))       

        return torch.from_numpy(v_pred).float()
    return f


if __name__ == "__main__":

    names = ["p001", "p010", "p050", "p200"]
    sigmas = [0.14, 0.09, 0.06, 0.05]
    path = "/scratch/jpelz/ma-pinns/TrackGen/binned_stats"
    os.chdir(path)

    for name, sigma in zip(names, sigmas):
        if not os.path.exists(f"{name}"):
            os.makedirs(f"{name}")
        os.chdir(f"{name}")
        print(f"Processing {name} with sigma {sigma}")
        # Load the data
        data = np.loadtxt(f"/scratch/jpelz/ma-pinns/TrackGen/HalfcylinderTracks_{name}_t14.5-15.dat", skiprows=1, delimiter=' ')

        t = 14.7
        data = data[abs(data[:, 4]-t) < 1e-3,:]
        particles = data[:, 1:4]
        velocities = data[:, 5:8]

        evaluatePINN(
            predict=gaussian_kernel(particles, velocities, sigma),
            datafile="/scratch/jpelz/ma-pinns/TrackGen/halfcylinder.nc",
            outputfile=f"gaussian_error_{name}.txt",
            tmin=14.65,
            tmax=14.75,
            nt_max=1,
            zmin=-0.5,
            zmax=0.5,
            nz_max=30,
            nx_max=240,
            ny_max=60,
            z_plot=0.0,
            t_plot=14.7,
        )
        os.chdir(path)


# if __name__ == "__main__":

#     names = ["p001", "p010", "p050", "p200"]
#     sigmas = [0.005, 0.002, 0.002, 0.0015]
#     times = [23*5.916E-3, 24*5.916E-3, 25*5.916E-3]
#     path = "/scratch/jpelz/da-challenge/DA_CASE01/binned_stats"

#     # names = ["p001", "p010", "p050", "p200"]
#     # sigmas = [0.14, 0.09, 0.06, 0.05]
#     # times = [14.6, 14.7, 14.8]
#     # path = "/scratch/jpelz/ma-pinns/TrackGen/binned_stats"


  

    

#     for name, sigma in zip(names, sigmas):
#         os.chdir(path)
#         if not os.path.exists(f"{name}"):
#             os.makedirs(f"{name}")
#         os.chdir(f"{name}")
#         print(f"Processing {name} with sigma {sigma}")
#         # Load the data
#         #data = np.load(f"/scratch/jpelz/ma-pinns/final_sims/_datasets/HC_t_14.6_14.8_{name}.npz")
#         data = np.load(f"/scratch/jpelz/ma-pinns/final_sims/_datasets/DA_CASE01_t_23_25_{name}.npz")
#         X_test = data['x_test']
#         Y_test = data['y_test']
#         X_train = data['x_train']
#         Y_train = data['y_train']

#         v_pred = np.zeros_like(Y_test)

#         for t in times:
#             mask_test = abs(X_test[:, 3]-t) < 1e-4
#             mask_train = abs(X_train[:, 3]-t) < 1e-4

#             particles = X_train[mask_train,:3]
#             velocities = Y_train[mask_train,:]

#             pred = gaussian_kernel(particles, velocities, sigma)(torch.from_numpy(X_test[mask_test,:]).float())

#             v_pred[mask_test] = pred.numpy()

#         rmse_u = np.sqrt(np.mean((v_pred[:, 0] - Y_test[:, 0])**2))
#         rmse_v = np.sqrt(np.mean((v_pred[:, 1] - Y_test[:, 1])**2))
#         rmse_w = np.sqrt(np.mean((v_pred[:, 2] - Y_test[:, 2])**2))

#         file=open(f"gaussian_error_{name}_t{t:.1f}_testset.txt", "w")
#         file.write(f"RMSE u,v,w: [{rmse_u}, {rmse_v}, {rmse_w}]\n")
#         file.close()

   