import netCDF4 as nc
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import time
from typing import Union, List, Tuple

from .plot import plotField, plotField2
import vtk
from vtkmodules.util import numpy_support

def abs_mae(Y_pred: Union[torch.Tensor, np.ndarray], Y_true: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute the absolute Mean Absolute Error (MAE) between predictions and ground truth.

    Args:
        Y_pred: Predicted values (torch.Tensor or np.ndarray).
        Y_true: Ground truth values (torch.Tensor or np.ndarray).

    Returns:
        The absolute MAE as a float.
    """
    if not isinstance(Y_pred, torch.Tensor):
        Y_pred = torch.tensor(Y_pred, requires_grad=False)
    if not isinstance(Y_true, torch.Tensor):
        Y_true = torch.tensor(Y_true, requires_grad=False)
    return torch.nanmean(torch.abs(Y_pred - Y_true)).item()
    

def rel_mae(Y_pred: Union[torch.Tensor, np.ndarray], Y_true: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute the relative Mean Absolute Error (MAE) in percent between predictions and ground truth.

    Args:
        Y_pred: Predicted values (torch.Tensor or np.ndarray).
        Y_true: Ground truth values (torch.Tensor or np.ndarray).

    Returns:
        The relative MAE as a percentage (float).
    """
    if not isinstance(Y_pred, torch.Tensor):
        Y_pred = torch.tensor(Y_pred, requires_grad=False)
    if not isinstance(Y_true, torch.Tensor):
        Y_true = torch.tensor(Y_true, requires_grad=False)
    rel_error = torch.abs(torch.div((Y_pred - Y_true), Y_true))
    rel_error = rel_error[~torch.isinf(rel_error)]
    return torch.nanmean(rel_error).item() * 100

def abs_rmse(Y_pred: Union[torch.Tensor, np.ndarray], Y_true: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute the absolute Root Mean Squared Error (RMSE) between predictions and ground truth.

    Args:
        Y_pred: Predicted values (torch.Tensor or np.ndarray).
        Y_true: Ground truth values (torch.Tensor or np.ndarray).

    Returns:
        The absolute RMSE as a float.
    """
    if not isinstance(Y_pred, torch.Tensor):
        Y_pred = torch.tensor(Y_pred, requires_grad=False)
    if not isinstance(Y_true, torch.Tensor):
        Y_true = torch.tensor(Y_true, requires_grad=False)
    return torch.sqrt(torch.nanmean((Y_pred - Y_true) ** 2)).item()


def rel_rmse(Y_pred: Union[torch.Tensor, np.ndarray], Y_true: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute the relative Root Mean Squared Error (RMSE) in percent between predictions and ground truth.

    Args:
        Y_pred: Predicted values (torch.Tensor or np.ndarray).
        Y_true: Ground truth values (torch.Tensor or np.ndarray).

    Returns:
        The relative RMSE as a percentage (float).
    """
    if not isinstance(Y_pred, torch.Tensor):
        Y_pred = torch.tensor(Y_pred, requires_grad=False)
    if not isinstance(Y_true, torch.Tensor):
        Y_true = torch.tensor(Y_true, requires_grad=False)
    rel_error = torch.div((Y_pred - Y_true), Y_true) ** 2
    rel_error = rel_error[~torch.isinf(rel_error)]
    return torch.sqrt(torch.nanmean(rel_error)).item() * 100



def detailed_data_err(
    model: torch.nn.Module,
    X: torch.Tensor,
    Y_true: torch.Tensor,
    denormalize: callable,
    redimension: callable
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Compute detailed error metrics (RMSE and MAE, both absolute and relative) for u, v, w components
    between model predictions and ground truth.

    Args:
        model: The trained PyTorch model to evaluate.
        X: Input tensor for the model.
        Y_true: Ground truth tensor.
        denormalize: Function to denormalize the output.
        redimension: Function to reshape or redimension the output.

    Returns:
        Tuple containing:
            - List of absolute RMSE values for u, v, w.
            - List of relative RMSE values for u, v, w.
            - List of absolute MAE values for u, v, w.
            - List of relative MAE values for u, v, w.
    """
    with torch.no_grad():
        Y_pred = model(X)
        Y_pred = redimension(y=denormalize(y=Y_pred))
        Y_true = redimension(y=denormalize(y=Y_true))

        abs_rmse_u = abs_rmse(Y_pred[:, 0], Y_true[:, 0])
        abs_rmse_v = abs_rmse(Y_pred[:, 1], Y_true[:, 1])
        abs_rmse_w = abs_rmse(Y_pred[:, 2], Y_true[:, 2])

        rel_rmse_u = rel_rmse(Y_pred[:, 0], Y_true[:, 0])
        rel_rmse_v = rel_rmse(Y_pred[:, 1], Y_true[:, 1])
        rel_rmse_w = rel_rmse(Y_pred[:, 2], Y_true[:, 2])

        abs_mae_u = abs_mae(Y_pred[:, 0], Y_true[:, 0])
        abs_mae_v = abs_mae(Y_pred[:, 1], Y_true[:, 1])
        abs_mae_w = abs_mae(Y_pred[:, 2], Y_true[:, 2])

        rel_mae_u = rel_mae(Y_pred[:, 0], Y_true[:, 0])
        rel_mae_v = rel_mae(Y_pred[:, 1], Y_true[:, 1])
        rel_mae_w = rel_mae(Y_pred[:, 2], Y_true[:, 2])

        a_rmse = [abs_rmse_u, abs_rmse_v, abs_rmse_w]
        r_rmse = [rel_rmse_u, rel_rmse_v, rel_rmse_w]
        a_mae = [abs_mae_u, abs_mae_v, abs_mae_w]
        r_mae = [rel_mae_u, rel_mae_v, rel_mae_w]

    return a_rmse, r_rmse, a_mae, r_mae


def load_data(
    datafile: str,
    tmin: float,
    tmax: float,
    nt_max: int,
    zmin: float,
    zmax: float,
    nz_max: int,
    nx_max: int = None,
    ny_max: int = None
) -> tuple:
    """
    Load and subsample ground truth velocity fields from a NetCDF (.nc) file.

    Args:
        datafile (str): Path to the NetCDF file containing the data.
        tmin (float): Minimum time value to include.
        tmax (float): Maximum time value to include.
        nt_max (int): Maximum number of time steps to load.
        zmin (float): Minimum z-coordinate value to include.
        zmax (float): Maximum z-coordinate value to include.
        nz_max (int): Maximum number of z slices to load.
        nx_max (int, optional): Maximum number of x slices to load. If None, use all.
        ny_max (int, optional): Maximum number of y slices to load. If None, use all.
    Returns:
        tuple: (xdim_filt, ydim_filt, zdim_filt, tdim_filt, u_data, v_data, w_data)
            - xdim_filt (np.ndarray): Filtered x-dimension coordinates.
            - ydim_filt (np.ndarray): Filtered y-dimension coordinates.
            - zdim_filt (np.ndarray): Filtered z-dimension coordinates.
            - tdim_filt (np.ndarray): Filtered time steps.
            - u_data (np.ndarray): Filtered u-velocity field data.
            - v_data (np.ndarray): Filtered v-velocity field data.
            - w_data (np.ndarray): Filtered w-velocity field data.
    Notes:
        The function applies filtering and subsampling along the time, z, x, and y dimensions
        according to the provided limits and maximums. The returned arrays are suitable for
        use as ground truth fields in PINN evaluation or other post-processing tasks.
    """

    data=nc.Dataset(datafile)

    xdim=data['xdim'][:]
    ydim=data['ydim'][:]
    zdim=data['zdim'][:]
    tdim=data['tdim'][:]

    t_filter=np.bitwise_and(tdim>=tmin, tdim<=tmax)
    z_filter=np.bitwise_and(zdim>=zmin, zdim<=zmax)

    tdim_filt=tdim[t_filter]
    zdim_filt=zdim[z_filter]
    
    if nx_max is not None and len(xdim)>nx_max:
        xdim_filt=xdim[::len(xdim)//nx_max]
    else:
        xdim_filt=xdim

    if ny_max is not None and len(ydim)>ny_max:
        ydim_filt=ydim[::len(ydim)//ny_max]
    else:
        ydim_filt=ydim

    if len(tdim_filt)>nt_max:
        tdim_filt=tdim_filt[::len(tdim_filt)//nt_max]

    if len(zdim_filt)>nz_max:
        zdim_filt=zdim_filt[::len(zdim_filt)//nz_max]
    
    t_filter=np.isin(tdim, tdim_filt)
    z_filter=np.isin(zdim, zdim_filt)
    y_filter=np.isin(ydim, ydim_filt)
    x_filter=np.isin(xdim, xdim_filt)

    u_data=data['u'][t_filter, z_filter, y_filter, x_filter]
    v_data=data['v'][t_filter, z_filter, y_filter, x_filter]
    w_data=data['w'][t_filter, z_filter, y_filter, x_filter]

    return xdim_filt, ydim_filt, zdim_filt, tdim_filt, u_data, v_data, w_data


def evaluatePINN(
    predict: callable,
    datafile: str,
    outputfile: str,
    tmin: float,
    tmax: float,
    nt_max: int,
    zmin: float,
    zmax: float,
    nz_max: int,
    nx_max: int = None,
    ny_max: int = None,
    z_plot: Union[float, List[float]] = 0,
    t_plot: Union[float, List[float]] = 0,
    **kwargs
) -> None:
    """
    Evaluate a Physics-Informed Neural Network (PINN) model against ground truth data, compute error metrics, 
    and generate diagnostic plots.

    Args:
        predict (callable): The PINN model's prediction function, which takes a tensor of input coordinates and returns predicted velocity components.
        datafile (str): Path to the file containing ground truth data.
        outputfile (str): Path to the output file where evaluation results will be written.
        tmin (float): Minimum time value to consider from the data.
        tmax (float): Maximum time value to consider from the data.
        nt_max (int): Maximum number of time steps to load.
        zmin (float): Minimum z-coordinate value to consider from the data.
        zmax (float): Maximum z-coordinate value to consider from the data.
        nz_max (int): Maximum number of z slices to load.
        nx_max (int, optional): Maximum number of x grid points to load. Defaults to None (load all).
        ny_max (int, optional): Maximum number of y grid points to load. Defaults to None (load all).
        z_plot (float or list[float], optional): z-coordinate(s) at which to plot error fields. Defaults to 0.
        t_plot (float or list[float], optional): Time(s) at which to plot error fields. Defaults to 0.
        **kwargs: Additional keyword arguments to be logged in the output file.

    Returns:
        None

    Side Effects:
        - Writes evaluation metrics (MAE, RMSE, relative errors) to the specified output file.
        - Saves error fields as a NumPy .npz file ("error_fields.npz").
        - Generates and saves diagnostic plots for error fields at specified slices.
    """

    file=open(outputfile, "w")

    file.write(time.ctime())

    for key, value in kwargs.items():
        file.write("\n"+"{0} = {1}".format(key, value))

    #Load Ground Truth

    xdim, ydim, zdim, tdim, u, v, w = load_data(datafile, tmin, tmax, nt_max, zmin, zmax, nz_max, nx_max, ny_max)

    dx=xdim[1]-xdim[0]
    dy=ydim[1]-ydim[0]
    if len(zdim)>1: dz=zdim[1]-zdim[0]
    if len(tdim)>1: dt=tdim[1]-tdim[0]


    mag_true=np.sqrt(u**2+v**2+w**2)
    if len(tdim) > 1:
        u_grad_true_t = np.gradient(u, dt, axis=0)
        v_grad_true_t = np.gradient(v, dt, axis=0)
        w_grad_true_t = np.gradient(w, dt, axis=0)
        mag_grad_true_t = np.gradient(mag_true, dt, axis=0)
    else:
        u_grad_true_t = v_grad_true_t = w_grad_true_t = mag_grad_true_t = 0

    if len(zdim) > 1:
        u_grad_true_z = np.gradient(u, dz, axis=1)
        v_grad_true_z = np.gradient(v, dz, axis=1)
        w_grad_true_z = np.gradient(w, dz, axis=1)
        mag_grad_true_z = np.gradient(mag_true, dz, axis=1)
    else:
        u_grad_true_z = v_grad_true_z = w_grad_true_z = mag_grad_true_z = 0

    u_grad_true_y = np.gradient(u, dy, axis=2)
    v_grad_true_y = np.gradient(v, dy, axis=2)
    w_grad_true_y = np.gradient(w, dy, axis=2)
    mag_grad_true_y = np.gradient(mag_true, dy, axis=2)

    u_grad_true_x = np.gradient(u, dx, axis=3)
    v_grad_true_x = np.gradient(v, dx, axis=3)
    w_grad_true_x = np.gradient(w, dx, axis=3)
    mag_grad_true_x = np.gradient(mag_true, dx, axis=3)

    u_grad_true = np.sqrt(u_grad_true_z**2 + u_grad_true_y**2 + u_grad_true_x**2)
    v_grad_true = np.sqrt(v_grad_true_z**2 + v_grad_true_y**2 + v_grad_true_x**2)
    w_grad_true = np.sqrt(w_grad_true_z**2 + w_grad_true_y**2 + w_grad_true_x**2)

    del(u_grad_true_z, v_grad_true_z, w_grad_true_z, u_grad_true_y, v_grad_true_y, w_grad_true_y, u_grad_true_x, v_grad_true_x, w_grad_true_x)


    #Predict
    z_grid,t_grid,y_grid,x_grid=np.meshgrid(zdim,tdim,ydim, xdim)

    X=torch.from_numpy(np.transpose(np.vstack((x_grid.ravel(),y_grid.ravel(),z_grid.ravel(),t_grid.ravel()))))
    del (z_grid,y_grid,x_grid,t_grid)
    with torch.no_grad():
        Y_pred=predict(X)[:,0:3]
        Y_pred=Y_pred.detach().numpy()

    u_pred=Y_pred[:,0].reshape(u.shape)
    v_pred=Y_pred[:,1].reshape(v.shape)
    w_pred=Y_pred[:,2].reshape(w.shape)

    del Y_pred

    mag_pred=np.sqrt(u_pred**2+v_pred**2+w_pred**2)
    if len(tdim) > 1:
        u_grad_pred_t = np.gradient(u_pred, dt, axis=0)
        v_grad_pred_t = np.gradient(v_pred, dt, axis=0)
        w_grad_pred_t = np.gradient(w_pred, dt, axis=0)
        mag_grad_pred_t = np.gradient(mag_pred, dt, axis=0)
    else:
        u_grad_pred_t = v_grad_pred_t = w_grad_pred_t = mag_grad_pred_t = 0

    if len(zdim) > 1:
        u_grad_pred_z = np.gradient(u_pred, dz, axis=1)
        v_grad_pred_z = np.gradient(v_pred, dz, axis=1)
        w_grad_pred_z = np.gradient(w_pred, dz, axis=1)
        mag_grad_pred_z = np.gradient(mag_pred, dz, axis=1)
    else:
        u_grad_pred_z = v_grad_pred_z = w_grad_pred_z = mag_grad_pred_z = 0

    u_grad_pred_y = np.gradient(u_pred, dy, axis=2)
    v_grad_pred_y = np.gradient(v_pred, dy, axis=2)
    w_grad_pred_y = np.gradient(w_pred, dy, axis=2)
    mag_grad_pred_y = np.gradient(mag_pred, dy, axis=2)

    u_grad_pred_x = np.gradient(u_pred, dx, axis=3)
    v_grad_pred_x = np.gradient(v_pred, dx, axis=3)
    w_grad_pred_x = np.gradient(w_pred, dx, axis=3)
    mag_grad_pred_x = np.gradient(mag_pred, dx, axis=3)

    u_grad_pred = np.sqrt(u_grad_pred_z**2 + u_grad_pred_y**2 + u_grad_pred_x**2)
    v_grad_pred = np.sqrt(v_grad_pred_z**2 + v_grad_pred_y**2 + v_grad_pred_x**2)
    w_grad_pred = np.sqrt(w_grad_pred_z**2 + w_grad_pred_y**2 + w_grad_pred_x**2)


    del(u_grad_pred_z, v_grad_pred_z, w_grad_pred_z, u_grad_pred_y, v_grad_pred_y, w_grad_pred_y, u_grad_pred_x, v_grad_pred_x, w_grad_pred_x)

    #error fields
    u_error=u_pred-u
    v_error=v_pred-v
    w_error=w_pred-w
    mag_error=mag_pred-mag_true
    u_grad_error=u_grad_pred-u_grad_true
    v_grad_error=v_grad_pred-v_grad_true
    w_grad_error=w_grad_pred-w_grad_true
    u_t_error=u_grad_pred_t-u_grad_true_t
    v_t_error=v_grad_pred_t-v_grad_true_t
    w_t_error=w_grad_pred_t-w_grad_true_t
    mag_x_error=mag_grad_pred_x-mag_grad_true_x
    mag_y_error=mag_grad_pred_y-mag_grad_true_y
    mag_z_error=mag_grad_pred_z-mag_grad_true_z
    mag_t_error=mag_grad_pred_t-mag_grad_true_t

    np.savez("error_fields.npz", u_error=u_error, v_error=v_error, w_error=w_error, mag_error=mag_error,
             u_grad_error=u_grad_error, v_grad_error=v_grad_error, w_grad_error=w_grad_error, u_t_error=u_t_error,
             v_t_error=v_t_error, w_t_error=w_t_error, mag_x_error=mag_x_error, mag_y_error=mag_y_error,
             mag_z_error=mag_z_error, mag_t_error=mag_t_error)

    # Save separate VTI files for each time slice using VTK

    # error fields shape: (nt, nz, ny, nx)
    nt, nz, ny, nx = u_error.shape

    def to_vtk_order(arr):
        # arr shape: (nz, ny, nx) -> (nx, ny, nz)
        return np.ascontiguousarray(np.transpose(arr, (2,1,0)))

    # Compute cell sizes
    dx = (xdim[1] - xdim[0]) if len(xdim) > 1 else 1.0
    dy = (ydim[1] - ydim[0]) if len(ydim) > 1 else 1.0
    dz = (zdim[1] - zdim[0]) if len(zdim) > 1 else 1.0

    # Compute grid origin so that xdim, ydim, zdim are cell centers
    origin_x = xdim[0] - dx / 2
    origin_y = ydim[0] - dy / 2
    origin_z = zdim[0] - dz / 2

    for t_idx in range(nt):
        # Extract error fields for this time slice
        u_err = u_error[t_idx]
        v_err = v_error[t_idx]
        w_err = w_error[t_idx]
        mag_err = mag_error[t_idx]
        u_grad_err = u_grad_error[t_idx]
        v_grad_err = v_grad_error[t_idx]
        w_grad_err = w_grad_error[t_idx]
        u_t_err = u_t_error[t_idx]
        v_t_err = v_t_error[t_idx]
        w_t_err = w_t_error[t_idx]
        mag_x_err = mag_x_error[t_idx]
        mag_y_err = mag_y_error[t_idx]
        mag_z_err = mag_z_error[t_idx]
        mag_t_err = mag_t_error[t_idx]

        # Create vtkImageData for this time slice
        image = vtk.vtkImageData()
        image.SetDimensions(nx, ny, nz)
        image.SetOrigin(origin_x, origin_y, origin_z)
        image.SetSpacing(dx, dy, dz)

        def add_array(image, arr, name):
            vtk_arr = numpy_support.numpy_to_vtk(num_array=to_vtk_order(arr).ravel(order="F"), deep=True)
            vtk_arr.SetName(name)
            image.GetPointData().AddArray(vtk_arr)

        add_array(image, u_err, "u_error")
        add_array(image, v_err, "v_error")
        add_array(image, w_err, "w_error")
        add_array(image, mag_err, "mag_error")
        add_array(image, u_grad_err, "u_grad_error")
        add_array(image, v_grad_err, "v_grad_error")
        add_array(image, w_grad_err, "w_grad_error")
        add_array(image, u_t_err, "u_t_error")
        add_array(image, v_t_err, "v_t_error")
        add_array(image, w_t_err, "w_t_error")
        add_array(image, mag_x_err, "mag_x_error")
        add_array(image, mag_y_err, "mag_y_error")
        add_array(image, mag_z_err, "mag_z_error")
        add_array(image, mag_t_err, "mag_t_error")

        # Write to VTI file
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(f"error_fields_t{t_idx:03d}.vti")
        writer.SetInputData(image)
        writer.Write()


    #MAE
    mae_u = abs_mae(u_pred.ravel(), u.ravel())
    mae_v = abs_mae(v_pred.ravel(), v.ravel())
    mae_w = abs_mae(w_pred.ravel(), w.ravel())
    mae_mag = abs_mae(mag_pred.ravel(), mag_true.ravel())
    mae_u_grad = abs_mae(u_grad_pred.ravel(), u_grad_true.ravel())
    mae_v_grad = abs_mae(v_grad_pred.ravel(), v_grad_true.ravel())
    mae_w_grad = abs_mae(w_grad_pred.ravel(), w_grad_true.ravel())
    mae_u_t = abs_mae(u_grad_pred_t.ravel(), u_grad_true_t.ravel())
    mae_v_t = abs_mae(v_grad_pred_t.ravel(), v_grad_true_t.ravel())
    mae_w_t = abs_mae(w_grad_pred_t.ravel(), w_grad_true_t.ravel())
    mae_mag_x = abs_mae(mag_grad_pred_x.ravel(), mag_grad_true_x.ravel())
    mae_mag_y = abs_mae(mag_grad_pred_y.ravel(), mag_grad_true_y.ravel())
    mae_mag_z = abs_mae(mag_grad_pred_z.ravel(), mag_grad_true_z.ravel())
    mae_mag_t = abs_mae(mag_grad_pred_t.ravel(), mag_grad_true_t.ravel())

    #Rel MAE
    rel_mae_u = rel_mae(u_pred.ravel(), u.ravel())
    rel_mae_v = rel_mae(v_pred.ravel(), v.ravel())
    rel_mae_w = rel_mae(w_pred.ravel(), w.ravel())
    rel_mae_mag = rel_mae(mag_pred.ravel(), mag_true.ravel())
    rel_mae_u_grad = rel_mae(u_grad_pred.ravel(), u_grad_true.ravel())
    rel_mae_v_grad = rel_mae(v_grad_pred.ravel(), v_grad_true.ravel())
    rel_mae_w_grad = rel_mae(w_grad_pred.ravel(), w_grad_true.ravel())
    rel_mae_u_t = rel_mae(u_grad_pred_t.ravel(), u_grad_true_t.ravel())
    rel_mae_v_t = rel_mae(v_grad_pred_t.ravel(), v_grad_true_t.ravel())
    rel_mae_w_t = rel_mae(w_grad_pred_t.ravel(), w_grad_true_t.ravel())
    rel_mae_mag_x = rel_mae(mag_grad_pred_x.ravel(), mag_grad_true_x.ravel())
    rel_mae_mag_y = rel_mae(mag_grad_pred_y.ravel(), mag_grad_true_y.ravel())
    rel_mae_mag_z = rel_mae(mag_grad_pred_z.ravel(), mag_grad_true_z.ravel())
    rel_mae_mag_t = rel_mae(mag_grad_pred_t.ravel(), mag_grad_true_t.ravel())

    #RMSE
    rmse_u = abs_rmse(u_pred.ravel(), u.ravel())
    rmse_v = abs_rmse(v_pred.ravel(), v.ravel())
    rmse_w = abs_rmse(w_pred.ravel(), w.ravel())
    rmse_mag = abs_rmse(mag_pred.ravel(), mag_true.ravel())
    rmse_u_grad = abs_rmse(u_grad_pred.ravel(), u_grad_true.ravel())
    rmse_v_grad = abs_rmse(v_grad_pred.ravel(), v_grad_true.ravel())
    rmse_w_grad = abs_rmse(w_grad_pred.ravel(), w_grad_true.ravel())
    rmse_u_t = abs_rmse(u_grad_pred_t.ravel(), u_grad_true_t.ravel())
    rmse_v_t = abs_rmse(v_grad_pred_t.ravel(), v_grad_true_t.ravel())
    rmse_w_t = abs_rmse(w_grad_pred_t.ravel(), w_grad_true_t.ravel())
    rmse_mag_x = abs_rmse(mag_grad_pred_x.ravel(), mag_grad_true_x.ravel())
    rmse_mag_y = abs_rmse(mag_grad_pred_y.ravel(), mag_grad_true_y.ravel())
    rmse_mag_z = abs_rmse(mag_grad_pred_z.ravel(), mag_grad_true_z.ravel())
    rmse_mag_t = abs_rmse(mag_grad_pred_t.ravel(), mag_grad_true_t.ravel())

    #Rel RMSE
    rel_rmse_u = rel_rmse(u_pred.ravel(), u.ravel())
    rel_rmse_v = rel_rmse(v_pred.ravel(), v.ravel())
    rel_rmse_w = rel_rmse(w_pred.ravel(), w.ravel())
    rel_rmse_mag = rel_rmse(mag_pred.ravel(), mag_true.ravel())
    rel_rmse_u_grad = rel_rmse(u_grad_pred.ravel(), u_grad_true.ravel())
    rel_rmse_v_grad = rel_rmse(v_grad_pred.ravel(), v_grad_true.ravel())
    rel_rmse_w_grad = rel_rmse(w_grad_pred.ravel(), w_grad_true.ravel())
    rel_rmse_u_t = rel_rmse(u_grad_pred_t.ravel(), u_grad_true_t.ravel())
    rel_rmse_v_t = rel_rmse(v_grad_pred_t.ravel(), v_grad_true_t.ravel())
    rel_rmse_w_t = rel_rmse(w_grad_pred_t.ravel(), w_grad_true_t.ravel())
    rel_rmse_mag_x = rel_rmse(mag_grad_pred_x.ravel(), mag_grad_true_x.ravel())
    rel_rmse_mag_y = rel_rmse(mag_grad_pred_y.ravel(), mag_grad_true_y.ravel())
    rel_rmse_mag_z = rel_rmse(mag_grad_pred_z.ravel(), mag_grad_true_z.ravel())
    rel_rmse_mag_t = rel_rmse(mag_grad_pred_t.ravel(), mag_grad_true_t.ravel())

    file.write(f"\nMAE (m/s): \n u: {mae_u:.4e} \n v: {mae_v:.4e} \n w: {mae_w:.4e} \n mag: {mae_mag:.4e} \n u_grad: {mae_u_grad:.4e} \n v_grad: {mae_v_grad:.4e} \n w_grad: {mae_w_grad:.4e} \n u_t: {mae_u_t:.4e} \n v_t: {mae_v_t:.4e} \n w_t: {mae_w_t:.4e} \n mag_x: {mae_mag_x:.4e} \n mag_y: {mae_mag_y:.4e} \n mag_z: {mae_mag_z:.4e} \n mag_t: {mae_mag_t:.4e}")
    file.write(f"\nRel MAE (%): \n u: {rel_mae_u:.4e} \n v: {rel_mae_v:.4e} \n w: {rel_mae_w:.4e} \n mag: {rel_mae_mag:.4e} \n u_grad: {rel_mae_u_grad:.4e} \n v_grad: {rel_mae_v_grad:.4e} \n w_grad: {rel_mae_w_grad:.4e} \n u_t: {rel_mae_u_t:.4e} \n v_t: {rel_mae_v_t:.4e} \n w_t: {rel_mae_w_t:.4e} \n mag_x: {rel_mae_mag_x:.4e} \n mag_y: {rel_mae_mag_y:.4e} \n mag_z: {rel_mae_mag_z:.4e} \n mag_t: {rel_mae_mag_t:.4e}")
    file.write(f"\nRMSE (m/s): \n u: {rmse_u:.4e} \n v: {rmse_v:.4e} \n w: {rmse_w:.4e} \n mag: {rmse_mag:.4e} \n u_grad: {rmse_u_grad:.4e} \n v_grad: {rmse_v_grad:.4e} \n w_grad: {rmse_w_grad:.4e} \n u_t: {rmse_u_t:.4e} \n v_t: {rmse_v_t:.4e} \n w_t: {rmse_w_t:.4e} \n mag_x: {rmse_mag_x:.4e} \n mag_y: {rmse_mag_y:.4e} \n mag_z: {rmse_mag_z:.4e} \n mag_t: {rmse_mag_t:.4e}")
    file.write(f"\nRel RMSE (%): \n u: {rel_rmse_u:.4e} \n v: {rel_rmse_v:.4e} \n w: {rel_rmse_w:.4e} \n mag: {rel_rmse_mag:.4e} \n u_grad: {rel_rmse_u_grad:.4e} \n v_grad: {rel_rmse_v_grad:.4e} \n w_grad: {rel_rmse_w_grad:.4e} \n u_t: {rel_rmse_u_t:.4e} \n v_t: {rel_rmse_v_t:.4e} \n w_t: {rel_rmse_w_t:.4e} \n mag_x: {rel_rmse_mag_x:.4e} \n mag_y: {rel_rmse_mag_y:.4e} \n mag_z: {rel_rmse_mag_z:.4e} \n mag_t: {rel_rmse_mag_t:.4e}")

    file.close()

    #find matching slices in data
    if not isinstance(t_plot, list):
        t_plot = [t_plot]
    if not isinstance(z_plot, list):
        z_plot = [z_plot]

    # Find matching slices in data
    t_idx = [(np.abs(tdim - t)).argmin() for t in t_plot]
    z_idx = [(np.abs(zdim - z)).argmin() for z in z_plot]

    #Plot
    for i in t_idx:
        for j in z_idx:


            t=tdim[i]
            z=zdim[j]

            plotField2(xdim, ydim, (u_error[i,j,:,:]), None, None, None, f"u_error (m/s)", True, None, None,
                       xdim, ydim, (u_grad_error[i,j,:,:]), None, None, None, f"u_grad_error (1/s^2)", True, None, None,
                       f"Error in u and u_grad at z={zdim[j]:.2f} and t={tdim[i]:.2f} (pred-true)", f"u_error_t{t:.2f}_z{z:.2f}.png")

            plotField2(xdim, ydim, (v_error[i,j,:,:]), None, None, None, f"v_error (m/s)", True, None, None,
                       xdim, ydim, (v_grad_error[i,j,:,:]), None, None, None, f"v_grad_error (1/s^2)", True, None, None,
                       f"Error in v and v_grad at z={zdim[j]:.2f} and t={tdim[i]:.2f} (pred-true)", f"v_error_t{t:.2f}_z{z:.2f}.png")

            plotField2(xdim, ydim, (w_error[i,j,:,:]), None, None, None, f"w_error (m/s)", True, None, None,
                       xdim, ydim, (w_grad_error[i,j,:,:]), None, None, None, f"w_grad_error (1/s^2)", True, None, None,
                       f"Error in w and w_grad at z={zdim[j]:.2f} and t={tdim[i]:.2f} (pred-true)", f"w_error_t{t:.2f}_z{z:.2f}.png")
        


def plotTruthXY(
    datafile: str, 
    z: float, 
    t: float
) -> None:
    """
    Plots the true velocity field components from a NetCDF data file at a specified z and t location.

    This function extracts the velocity components (u, v, w) from the given NetCDF file at the closest available
    z and t indices to the provided values. It computes the velocity magnitude and generates two plots:
    one showing the magnitude and u component, and another showing the v and w components, both as functions of x and y.

    Args:
        datafile (str): Path to the NetCDF data file containing the velocity field data.
        z (float): The z-coordinate at which to extract and plot the data.
        t (float): The time value at which to extract and plot the data.

    Returns:
        None: The function saves the generated plots to files and does not return any value.

    Notes:
        - The function assumes the NetCDF file contains variables 'xdim', 'ydim', 'zdim', 'tdim', 'u', 'v', and 'w'.
        - The output plots are saved as PNG files with filenames indicating the z and t values.
    """
    data=nc.Dataset(datafile)

    xdim=data['xdim'][:]
    ydim=data['ydim'][:]
    zdim=data['zdim'][:]
    tdim=data['tdim'][:]
    tdim=tdim

    x,y=np.meshgrid(xdim,ydim)
    t_idx=(np.abs(tdim-t)).argmin()
    z_idx=(np.abs(zdim-z)).argmin()

    u_data=data['u'][t_idx][z_idx]
    v_data=data['v'][t_idx][z_idx]
    w_data=data['w'][t_idx][z_idx]

    z=zdim[z_idx]
    t=tdim[t_idx]

    mag=np.sqrt(u_data**2+v_data**2+w_data**2)

    mag=np.reshape(mag, (240,640))

    #mag & u
    title=(f'True velocity field (mag & u) at z={z:.2f} and t={t:.2f}')
    label1=('mag (m/s)')
    label2=('u (m/s)')
    filename=(f"True_mag-u-field_z{z:.2f}_t{t:.2f}.png")

    plotField2(x, y, mag, None, None, None, label1,
               x, y, u_data, None, None, None, label2,
               title, filename)

    #v & w

    title=(f'True velocity field (v & w) at z={z:.2f} and t={t:.2f}')
    label1=('v (m/s)')
    label2=('w (m/s)')
    filename=(f"True_v-w-field_z{z:.2f}_t{t:.2f}.png")

    plotField2(x, y, v_data, None, None, None, label1,
               x, y, w_data, None, None, None, label2,
               title, filename)