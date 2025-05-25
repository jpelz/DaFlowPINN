import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import torch


def plotField(
    x: np.ndarray,
    y: np.ndarray,
    field: np.ndarray,
    x_point: np.ndarray,
    y_point: np.ndarray,
    data_point: np.ndarray,
    label: str,
    suptitle: str,
    filename: str,
    dim1_name: str = 'x',
    dim2_name: str = 'y',
    centered: bool = False,
    vmin: float = None,
    vmax: float = None
) -> plt.Figure:
    """
    Plots a 2D field with optional data points overlay and colorbar.

    Args:
        x (np.ndarray): 2D array of x-coordinates (meshgrid).
        y (np.ndarray): 2D array of y-coordinates (meshgrid).
        field (np.ndarray): 2D array of field values to plot.
        x_point (np.ndarray): 1D array of x-coordinates for data points (optional).
        y_point (np.ndarray): 1D array of y-coordinates for data points (optional).
        data_point (np.ndarray): 1D array of data values for scatter points (optional).
        label (str): Label for the colorbar.
        suptitle (str): Figure title.
        filename (str): Path to save the figure.
        dim1_name (str): Name for x-axis (default 'x').
        dim2_name (str): Name for y-axis (default 'y').
        centered (bool): Whether to center the color normalization at zero.
        vmin (float): Minimum value for color normalization (optional).
        vmax (float): Maximum value for color normalization (optional).

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    fig.suptitle(suptitle, fontsize=20)
    
    min1 = field.min()
    max1 = field.max()

    if data_point is not None:
        min1 = np.minimum(min1, data_point.min())
        max1 = np.maximum(max1, data_point.max())

    if vmin is not None:
        min1 = vmin
    if vmax is not None:
        max1 = vmax

    if centered:
        norm = mpl.colors.CenteredNorm(vcenter=0, halfrange=max(abs(min1), abs(max1)))
    else:
        norm = mpl.colors.Normalize(vmin=min1, vmax=max1)

    plt_mag = ax.contourf(x, y, field, 256, cmap='jet', norm=norm)
    if data_point is not None:
        ax.scatter(x_point, y_point, c=data_point, cmap='jet', s=20, edgecolors='black', norm=norm)
    ax.set_xlabel(f"{dim1_name} (m)")
    ax.set_ylabel(f"{dim2_name} (m)")
    ax.axes.set_xlim(left=x.min(), right=x.max())
    ax.axes.set_ylim(bottom=y.min(), top=y.max())
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(plt_mag, cax=cax)
    cbar.set_label(label)

    plt.savefig(filename)
    plt.close()

    return fig

def plotField2(
    x1: np.ndarray,
    y1: np.ndarray,
    field1: np.ndarray,
    x_point1: np.ndarray,
    y_point1: np.ndarray,
    data_point1: np.ndarray,
    label1: str,
    centered1: bool,
    vmin1: float,
    vmax1: float,
    x2: np.ndarray,
    y2: np.ndarray,
    field2: np.ndarray,
    x_point2: np.ndarray,
    y_point2: np.ndarray,
    data_point2: np.ndarray,
    label2: str,
    centered2: bool,
    vmin2: float,
    vmax2: float,
    suptitle: str,
    filename: str,
    dim1_name: str = 'x',
    dim2_name: str = 'y'
) -> plt.Figure:
    """
    Plots two 2D scalar fields side by side with optional overlaid scatter data points, colorbars, and custom normalization.
    Args:
        x1 (np.ndarray): X-coordinates for the first field (2D meshgrid).
        y1 (np.ndarray): Y-coordinates for the first field (2D meshgrid).
        field1 (np.ndarray): Scalar field values for the first plot.
        x_point1 (np.ndarray): X-coordinates for scatter points on the first plot.
        y_point1 (np.ndarray): Y-coordinates for scatter points on the first plot.
        data_point1 (np.ndarray): Data values for scatter points on the first plot.
        label1 (str): Colorbar label for the first plot.
        centered1 (bool): Whether to use a centered normalization (centered at zero) for the first plot.
        vmin1 (float): Minimum value for color normalization of the first plot. If None, uses data minimum.
        vmax1 (float): Maximum value for color normalization of the first plot. If None, uses data maximum.
        x2 (np.ndarray): X-coordinates for the second field (2D meshgrid).
        y2 (np.ndarray): Y-coordinates for the second field (2D meshgrid).
        field2 (np.ndarray): Scalar field values for the second plot.
        x_point2 (np.ndarray): X-coordinates for scatter points on the second plot.
        y_point2 (np.ndarray): Y-coordinates for scatter points on the second plot.
        data_point2 (np.ndarray): Data values for scatter points on the second plot.
        label2 (str): Colorbar label for the second plot.
        centered2 (bool): Whether to use a centered normalization (centered at zero) for the second plot.
        vmin2 (float): Minimum value for color normalization of the second plot. If None, uses data minimum.
        vmax2 (float): Maximum value for color normalization of the second plot. If None, uses data maximum.
        suptitle (str): Supertitle for the entire figure.
        filename (str): Path to save the resulting figure.
        dim1_name (str, optional): Name for the x-axis dimension (default is 'x').
        dim2_name (str, optional): Name for the y-axis dimension (default is 'y').
    Returns:
        plt.Figure: The matplotlib Figure object containing the plots.
    Notes:
        - Both fields are plotted using `contourf` with the 'jet' colormap.
        - If `data_point1` or `data_point2` is not None, scatter points are overlaid on the respective plots.
        - Colorbars are added for both plots, and normalization can be centered or linear.
        - The figure is saved to the specified filename and then closed.
    """
    
    fig, (ax1, ax2)=plt.subplots(2,1, figsize=(10,10))

    fig.suptitle(suptitle, fontsize=20)
    
    min1=field1.min()
    max1=field1.max()

    if data_point1 is not None:
        min1=np.minimum(min1, data_point1.min())
        max1=np.maximum(max1, data_point1.max())

    if vmin1 is not None:
        min1 = vmin1
    if vmax1 is not None:
        max1 = vmax1

    if centered1:
        norm1 = mpl.colors.CenteredNorm(vcenter=0, halfrange=max(abs(min1), abs(max1)))
    else:
        norm1 = mpl.colors.Normalize(vmin=min1, vmax=max1)


    plt_mag=ax1.contourf(x1, y1, field1 ,256, cmap='jet', norm=norm1)
    if data_point1 is not None:
        ax1.scatter(x_point1, y_point1, c=data_point1, cmap='jet', s=20, edgecolors='black',norm = norm1)
    ax1.set_xlabel(f"{dim1_name} (m)")
    ax1.set_ylabel(f"{dim2_name} (m)")
    ax1.axes.set_xlim(left=x1.min(), right=x1.max())
    ax1.axes.set_ylim(bottom=y1.min(), top=y1.max())
    ax1.set_aspect('equal')
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cbar1=fig.colorbar(plt_mag, cax=cax1)
    cbar1.set_label(label1)

    min2=field2.min()
    max2=field2.max()

    if data_point2 is not None:
        min2=np.minimum(min2, data_point2.min())
        max2=np.maximum(max2, data_point2.max())

    if label1 == label2:
      min2 = min1
      max2 = max1

    if vmin2 is not None:
        min2 = vmin2
    if vmax2 is not None:
        max2 = vmax2

    if centered2:
        norm2 = mpl.colors.CenteredNorm(vcenter=0, halfrange=max(abs(min2), abs(max2)))
    else:
        norm2 = mpl.colors.Normalize(vmin=min2, vmax=max2)

    plt_mag2=ax2.contourf(x2, y2, field2,256, cmap='jet', norm=norm2)
    if data_point2 is not None:
        ax2.scatter(x_point2, y_point2, c=data_point2, cmap='jet', s=20, edgecolors='black', norm=norm2)
    ax2.set_xlabel(f"{dim1_name} (m)")
    ax2.set_ylabel(f"{dim2_name} (m)")
    ax2.axes.set_xlim(left=x2.min(), right=x2.max())
    ax2.axes.set_ylim(bottom=y2.min(), top=y2.max())
    ax2.set_aspect('equal')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2=fig.colorbar(plt_mag2, cax=cax2)
    cbar2.set_label(label2)

    plt.savefig(filename)
    plt.close()
    return fig


def plotPINN_2D(
    predict: callable,
    plot_dims: list = [0, 1],
    dim3_slice: float = 0,
    t_slice: float = 0,
    component1: int = 0,
    data1: np.ndarray = None,
    dim3_tolerance: float = 0.05,
    centered1: bool = False,
    vmin1: float = None,
    vmax1: float = None,
    component2: int = None,
    centered2: bool = False,
    vmin2: float = None,
    vmax2: float = None,
    lb: list = [-0.5, -1.5],
    ub: list = [7.5, 1.5],
    resolution: list = [640, 240]
) -> plt.Figure:
    """
    Plots the 2D field(s) predicted by a PINN model along with the true data if available.

    Args:
        predict (callable): The PINN model prediction function. Should accept a torch tensor of shape (N, 4) and return a tensor of predictions.
        plot_dims (list[int], optional): The two spatial dimensions to plot (indices 0, 1, or 2). Default is [0, 1].
        dim3_slice (float, optional): The value at which to slice the third spatial dimension. Default is 0.
        t_slice (float, optional): The time slice value. Default is 0.
        component1 (int, optional): The first field/component to plot (see below for options). Default is 0.
        data1 (np.ndarray, optional): The true data array, shape (N, >=8). Default is None.
        dim3_tolerance (float, optional): Tolerance for selecting the third dimension slice from data1. Default is 0.05.
        centered1 (bool, optional): Whether to center the color normalization at zero for the first field. Default is False.
        vmin1 (float, optional): Minimum value for color normalization of the first field. Default is None.
        vmax1 (float, optional): Maximum value for color normalization of the first field. Default is None.
        component2 (int, optional): The second field/component to plot (optional). Default is None.
        centered2 (bool, optional): Whether to center the color normalization at zero for the second field. Default is False.
        vmin2 (float, optional): Minimum value for color normalization of the second field. Default is None.
        vmax2 (float, optional): Maximum value for color normalization of the second field. Default is None.
        lb (list[float], optional): Lower bounds for the plot dimensions. Default is [-0.5, -1.5].
        ub (list[float], optional): Upper bounds for the plot dimensions. Default is [7.5, 1.5].
        resolution (list[int], optional): Resolution of the plot grid [nx, ny]. Default is [640, 240].

    Returns:
        plt.Figure: The matplotlib figure object.

    Components:
        0: Magnitude of velocity
        1: u component of velocity
        2: v component of velocity
        3: w component of velocity
        4: Pressure (p)
        5: du1/dx
        6: du1/dy
        7: du2/dx
        8: du2/dy
        9: Vorticity
        10: Q-criterion (not implemented)

    Notes:
        - If component2 is provided, both fields are plotted in a single figure.
        - If data1 is provided and component1 < 4, true data points are overlaid.
    """
    comp_names = ['mag', 'u', 'v', 'w', 'p', 'du1/dx', 'du1/dy', 'du2/dx', 'du2/dy', 'vorticity', 'Q-criterion']
    dim_names = ['x', 'y', 'z']

    dim1, dim2 = plot_dims[0], plot_dims[1]
    dim3 = (set([0, 1, 2]) - set(plot_dims)).pop()

    # True Data
    if data1 is not None and component1 < 4:
        # Find the nearest data timestep
        t_idx = (np.abs(data1[:, 4] - t_slice)).argmin()
        t_slice = data1[t_idx, 4]

        t_filter = data1[:, 4] == t_slice
        z_filter = np.bitwise_and(data1[:, dim3 + 1] >= dim3_slice - dim3_tolerance, data1[:, dim3 + 1] <= dim3_slice + dim3_tolerance)
        filter = np.bitwise_and(t_filter, z_filter)

        x_true = data1[filter, dim1 + 1]
        y_true = data1[filter, dim2 + 1]

        u_true = data1[filter, 5]
        v_true = data1[filter, 6]
        w_true = data1[filter, 7]

        mag_true = np.sqrt(u_true**2 + v_true**2 + w_true**2)

        if component1 == 0:
            vel_true = mag_true
        else:
            vel_true = data1[filter, component1 + 4]
    else:
        x_true = None
        y_true = None
        vel_true = None

    # Predicted Data
    xdim = np.linspace(lb[0], ub[0], resolution[0])
    ydim = np.linspace(lb[1], ub[1], resolution[1])

    x, y = np.meshgrid(xdim, ydim)

    X = np.zeros((len(x.ravel()), 4))

    X[:, dim1] = x.ravel()
    X[:, dim2] = y.ravel()
    X[:, dim3] = dim3_slice * np.ones_like(x.ravel())
    X[:, 3] = t_slice * np.ones_like(x.ravel())

    X = torch.tensor(X).float()
    Y = predict(X)

    def get_field(component, xdim, ydim, Y):
        if component == 0:
            mag = np.sqrt(Y[:,0].detach().numpy()**2 + Y[:,1].detach().numpy()**2 + Y[:,2].detach().numpy()**2)
            field = np.reshape(mag, (resolution[1], resolution[0]))
            unit = 'm/s'
        elif component >= 1 and component <= 3:
            field = np.reshape(Y[:, component - 1].detach().numpy(), (resolution[1], resolution[0]))
            unit = 'm/s'
        elif component == 4:
            field = np.reshape(Y[:, 3].detach().numpy(), (resolution[1], resolution[0]))
            unit = 'Pa'
        elif component >= 5:
            u1 = Y[:, dim1].detach().numpy()
            u2 = Y[:, dim2].detach().numpy()

            u1 = np.reshape(u1, (resolution[1], resolution[0]))
            u2 = np.reshape(u2, (resolution[1], resolution[0]))

            dx = xdim[1] - xdim[0]
            dy = ydim[1] - ydim[0]

            grad_u1 = np.gradient(u1, dx, dy, edge_order=2)
            grad_u2 = np.gradient(u2, dx, dy, edge_order=2)

            unit = '1/s'

            if component >= 5 and component <= 6:
                field = grad_u1[component - 5]
            elif component >= 7 and component <= 8:
                field = grad_u2[component - 7]
            elif component == 9:
                vorticity = grad_u2[0] - grad_u1[1]
                field = vorticity
            elif component == 10:
                return #does not work yet
                dx = xdim[1] - xdim[0]
                dy = ydim[1] - ydim[0]

                
                # Compute gradients
                du_dx = np.gradient(u1, dx, axis=0)
                du_dy = np.gradient(u1, dy, axis=1) 
                
                dv_dx = np.gradient(u2, dx, axis=0)
                dv_dy = np.gradient(u2, dy, axis=1)
                
                S = 0.5 * (du_dy + dv_dx)
                O = 0.5 * (dv_dy - du_dx)
                
                field = 0.5 * (O**2 - S**2)
                field[field <= 0.1] = float("nan")
                unit = ' '
                
        return field, unit

    field1, unit1 = get_field(component1, xdim, ydim, Y)
    field2, unit2 = get_field(component2, xdim, ydim, Y) if component2 is not None else (None, None)

    if component2 is None:
        title = (f'Predicted ({comp_names[component1]}) field at {dim_names[dim3]}={dim3_slice:.2f} and t={t_slice:.3f}')
        label1 = (f'{comp_names[component1]} ({unit1})')
        if data1 is not None:
            filename = f"{comp_names[component1]}-field_{dim_names[dim3]}{dim3_slice:.2f}_t{t_slice:.3f}_data.png"
        else:
            filename = f"{comp_names[component1]}-field_{dim_names[dim3]}{dim3_slice:.2f}_t{t_slice:.3f}.png"
        fig = plotField(x, y, field1, x_true, y_true, vel_true, label1, title, filename, dim1_name=dim_names[dim1], dim2_name=dim_names[dim2], centered=centered1, vmin=vmin1, vmax=vmax1)
    else:
        title = (f'Predicted ({comp_names[component1]}) and ({comp_names[component2]}) field at {dim_names[dim3]}={dim3_slice:.2f} and t={t_slice:.3f}')
        label1 = (f'{comp_names[component1]} ({unit1})')
        label2 = (f'{comp_names[component2]} ({unit2})')
        if data1 is not None:
            filename = f"{comp_names[component1]}-{comp_names[component2]}-field_{dim_names[dim3]}{dim3_slice:.2f}_t{t_slice:.3f}_data.png"
        else:
            filename = f"{comp_names[component1]}-{comp_names[component2]}-field_{dim_names[dim3]}{dim3_slice:.2f}_t{t_slice:.3f}.png"
        fig = plotField2(x, y, field1, x_true, y_true, vel_true, label1, centered1, vmin1, vmax1,
                         x, y, field2, None, None, None, label2, centered2, vmin2, vmax2,
                         title, filename, dim1_name=dim_names[dim1], dim2_name=dim_names[dim2])
        

    return fig
    
    
    


  