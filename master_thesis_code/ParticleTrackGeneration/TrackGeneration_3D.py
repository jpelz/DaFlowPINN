#Generation of synthetic particle tracks using numerical integration

import netCDF4 as nc
import numpy as np
from scipy.stats import qmc
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
import matplotlib
from dataclasses import dataclass
import time


def loaddata(path:str):
    ds=nc.Dataset(path)
    data = ds
    return data

@dataclass
class Particles3D:
    """
    Generate Particles and step them forward in time based on a given velocity field
    
    ...
    
    
    Attributes
    ----------
    dim : int
        Spacial dimension
    t : float
        current time step
    nb : int
        current number of particles
    X : list[float]
        current position data (z,y,x)
    """

    nb:int #: Number of initial particles
    lb:float #: Lower boundaries of domain
    ub:float #: Upper Boundaries of domain
    do_cleanup:bool = True
    insert_lb:float = None
    insert_ub:float = None
    geometry:callable = None


    def __post_init__(self):
        
        self.t=0
        self.dim=3

        if self.insert_lb is None:
            self.insert_lb=self.lb
        if self.insert_ub is None:
            self.insert_ub=self.ub

        self.ids=np.arange(0,self.nb,dtype=int)
        self.max_id=self.nb

        
        
        self.__sampler=qmc.LatinHypercube(d=self.dim)
        sample=self.__sampler.random(n=self.nb)
        self.Positions=qmc.scale(sample, l_bounds=self.lb, u_bounds=self.ub)
        

    def __cleanup(self):
        """Removes particles outside of the domain and seeds new ones to keep the required number of particles"""
        outsiders=((np.isnan(self.Positions[:,0])) | (np.isnan(self.Positions[:,1]))) | (np.isnan(self.Positions[:,2]))
        if any(outsiders) & self.do_cleanup:
            new_sample=self.__sampler.random(n=sum(outsiders))
            self.Positions[outsiders,:]=qmc.scale(new_sample, self.insert_lb, self.insert_ub)
            self.ids[outsiders]=np.arange(self.max_id+1,self.max_id+sum(outsiders)+1)
            self.max_id += sum(outsiders)+1

        if self.geometry is not None:
            insiders=self.geometry(self.Positions[:,2], self.Positions[:,1], self.Positions[:,0])
            if any(insiders):
                new_sample=self.__sampler.random(n=sum(insiders))
                self.Positions[insiders,:]=qmc.scale(new_sample, self.insert_lb, self.insert_ub)
                self.ids[insiders]=np.arange(self.max_id+1,self.max_id+sum(insiders)+1)
                self.max_id += sum(insiders)+1

    def data(self, grid_points, u_data, v_data, w_data):
        """Returns current particle data as [id, x, y, z, t, u, v, w]"""

        X=np.c_[self.t*np.ones(self.nb), self.Positions]
        u=interpn(grid_points,u_data,X,bounds_error=False)
        v=interpn(grid_points,v_data,X,bounds_error=False)
        w=interpn(grid_points,w_data,X,bounds_error=False)

        return np.c_[self.ids, self.Positions[:,2], self.Positions[:,1], self.Positions[:,0], self.t*np.ones(self.nb),u,v,w]

    def step(self, grid_points, u_data, v_data, w_data, dt):
        """Move particles forward in time using Runge-Kutta scheme"""
        
        #Predictor
        X1=np.c_[self.t*np.ones(self.nb), self.Positions]
        u1=interpn(grid_points,u_data,X1,bounds_error=False)
        v1=interpn(grid_points,v_data,X1,bounds_error=False)
        w1=interpn(grid_points,w_data,X1,bounds_error=False)
        k1x=u1*dt
        k1y=v1*dt
        k1z=w1*dt

        #Corrector 1
        X2=X1
        X2[:,0] += 0.5*dt
        X2[:,1] += 0.5*k1z
        X2[:,2] += 0.5*k1y
        X2[:,3] += 0.5*k1x
        u2=interpn(grid_points,u_data,X2,bounds_error=False)
        v2=interpn(grid_points,v_data,X2,bounds_error=False)
        w2=interpn(grid_points,w_data,X2,bounds_error=False)
        k2x = u2*dt
        k2y = v2*dt
        k2z = w2*dt    

        #Corrector 2
        X3=X1
        X3[:,0] += 0.5*dt
        X3[:,1] += 0.5*k2z
        X3[:,2] += 0.5*k2y
        X3[:,3] += 0.5*k2x
        u3=interpn(grid_points,u_data,X3,bounds_error=False)
        v3=interpn(grid_points,v_data,X3,bounds_error=False)
        w3=interpn(grid_points,w_data,X3,bounds_error=False)
        k3x = u3*dt
        k3y = v3*dt
        k3z = w3*dt

        #Corrector 3
        X4=X1
        X4[:,0] += dt
        X4[:,1] += k3z
        X4[:,2] += k3y
        X4[:,3] += k3x
        u4=interpn(grid_points,u_data,X4,bounds_error=False)
        v4=interpn(grid_points,v_data,X4,bounds_error=False)
        w4=interpn(grid_points,w_data,X4,bounds_error=False)
        k4x = u4*dt
        k4y = v4*dt
        k4z = w4*dt

        self.t=self.t+dt
        self.Positions[:,0] += (1/6)*(k1z+2*k2z+2*k3z+k4z)
        self.Positions[:,1] += (1/6)*(k1y+2*k2y+2*k3y+k4y)
        self.Positions[:,2] += (1/6)*(k1x+2*k2x+2*k3x+k4x)
        
        #Cleanup particles outside of domain or inside geometry
        self.__cleanup()

    
        

def halfcylinder_3d(x,y,z):
    """Returns True if coordinate is inside half cylinder with R=0.125"""
    r=0.125
    inside=((x**2+y**2)<r**2) * (x<0)
    return inside.astype(bool)
    


if __name__=='__main__':

    #Settings
    nb_list=[50000, 200000]    #number of particles
    nt_list=[150]          #number of saved time steps

    #Load Data
    print(time.ctime()+':   Loading Data:')
    data=loaddata('halfcylinder.nc')

    xdim=data['xdim'][:]
    ydim=data['ydim'][:]
    zdim=data['zdim'][:]
    tdim=data['tdim'][:]
    tdim=tdim
    print(time.ctime()+':    - Dimensions loaded')


    u_data=data['u'][:]
    v_data=data['v'][:]
    w_data=data['w'][:]
    print(time.ctime()+':    - Velocities loaded')

    print(time.ctime()+':   Preparing Data')
    points=(tdim, zdim, ydim, xdim)
    

    #Boundaries [Y,X]
    lb=[min(zdim), min(ydim), min(xdim)]
    ub=[max(zdim), max(ydim), max(xdim)]

    #Insertion boundaries for update particles
    insert_lb=[-0.5,-1.5,-0.5]
    insert_ub=[0.5,1.5,-0.2]

    volume=(ub[0]-lb[0])*(ub[1]-lb[1])*(ub[2]-lb[2])
    
    print(time.ctime()+':   Data prepared')
    
    for nb_parts in nb_list:
        for nt in nt_list:
            mean_dist=(volume/nb_parts)**(1/3)
            #Initialize particles
            print(time.ctime()+':   Initilizing particles, nb='+str(nb_parts)+'    mean_dist='+str(mean_dist))
            parts=Particles3D(nb_parts, lb, ub, True, insert_lb, insert_ub, halfcylinder_3d)
            parts.t=tdim[0]
            t_factor=1 #if stepping is too fast, interpolate also in time by increasing t_factor
            dt=(tdim[1]-tdim[0])/t_factor

            #Prepare history file
            hist=np.empty((0,8))
            hist_freq=(len(tdim)-1)/nt
            filename="History_n"+str(nb_parts)+"_t"+str(nt)+".txt"

            print(time.ctime()+':   Stepping forward in time: ')
            for t in range(0,(len(tdim)-1)*t_factor):
                #print(time.ctime()+':    t: '+str(parts.t))
                if t%hist_freq==0:
                    # Append particles to history
                    hist=np.append(hist,parts.data(points,u_data, v_data, w_data), axis=0)
                #Step particles forward in time
                parts.step(points, u_data, v_data, w_data, dt)
            np.savetxt(filename, hist, delimiter=" ")
            print(time.ctime()+':      - saved '+str(nt)+" time steps")
