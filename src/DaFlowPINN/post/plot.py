import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import torch


def plotField(x, y, field, x_point, y_point, data_point, label, suptitle, filename, dim1_name='x', dim2_name='y', centered=False, vmin=None, vmax=None):
    
    fig, ax=plt.subplots(1,1, figsize=(10,5))

    fig.suptitle(suptitle, fontsize=20)
    
    min1=field.min()
    max1=field.max()

    if data_point is not None:
        min1=np.minimum(min1, data_point.min())
        max1=np.maximum(max1, data_point.max())

    if vmin is not None:
        min1 = vmin
    if vmax is not None:
        max1 = vmax

    if centered:
        norm = mpl.colors.CenteredNorm(vcenter=0, halfrange=max(abs(min1), abs(max1)))
    else:
        norm = mpl.colors.Normalize(vmin=min1, vmax=max1)



    plt_mag=ax.contourf(x, y, field ,256, cmap='jet', norm=norm)
    if data_point is not None:
        ax.scatter(x_point, y_point, c=data_point, cmap='jet', s=20, edgecolors='black', norm=norm)
    ax.set_xlabel(f"{dim1_name} (m)")
    ax.set_ylabel(f"{dim2_name} (m)")
    ax.axes.set_xlim(left=x.min(), right=x.max())
    ax.axes.set_ylim(bottom=y.min(), top=y.max())
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar=fig.colorbar(plt_mag, cax=cax)
    cbar.set_label(label)

    plt.savefig(filename)
    plt.close()

    return fig

def plotField2(x1, y1, field1, x_point1, y_point1, data_point1, label1, centered1, vmin1, vmax1,
               x2, y2, field2, x_point2, y_point2, data_point2, label2, centered2, vmin2, vmax2,
               suptitle, filename, dim1_name='x', dim2_name='y'):
    
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


def plotPINN_2D(predict: callable, plot_dims: list = [0, 1], dim3_slice: float = 0, t_slice: float = 0, component1: int = 0, data1: np.ndarray = None, dim3_tolerance: float = 0.05, centered1:bool = False, vmin1:float = None, vmax1:float = None,
                component2: int = None, centered2:bool = False, vmin2:float = None, vmax2:float = None, lb: list = [-0.5, -1.5], ub: list = [7.5, 1.5], resolution: list = [640, 240]) -> plt.Figure:
    """
    Plots the 2D field predicted by a PINN model along with the true data if available.

    Args:
        predict (callable): The PINN model prediction function.
        plot_dims (list): The dimensions to plot.
        dim3_slice (float): The slice value for the third dimension.
        t_slice (float): The time slice value.
        component1 (int): The component to plot (see list of possible components to plot).
        data1 (np.ndarray): The true data array.
        dim3_tolerance (float): The tolerance for the third dimension slice.
        component2 (int): The second component to plot (optional).
        lb (list): The lower bounds for the plot dimensions.
        ub (list): The upper bounds for the plot dimensions.
        resolution (list): The resolution of the plot.

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
        10: Q-criterion # not implemented yet
    """
    comp_names = ['mag', 'u', 'v', 'w', 'p', 'du1/dx', 'du1/dy', 'du2/dx', 'du2/dy', 'vorticity', 'Q-criterion']  # Component names
    dim_names = ['x', 'y', 'z']  # Dimension names

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
    xdim = np.linspace(lb[plot_dims[0]], ub[plot_dims[0]], resolution[0])
    ydim = np.linspace(lb[plot_dims[1]], ub[plot_dims[1]], resolution[1])

    x, y = np.meshgrid(xdim, ydim)

    X = np.zeros((len(x.ravel()), 4))

    X[:, dim1] = x.ravel()
    X[:, dim2] = y.ravel()
    X[:, dim3] = dim3_slice * np.ones_like(x.ravel())
    X[:, 3] = t_slice * np.ones_like(x.ravel())

    X = torch.tensor(X).float()

    X.requires_grad = True
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
    
    
    


  