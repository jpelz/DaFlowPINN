#Originally from: https://turbulence.utah.edu/codes/turbogenpython/tkespec.py

# -*- coding: utf-8 -*-
"""
Created on Fri May  9 10:14:44 2014

@author: tsaad
"""

import numpy as np
from numpy.fft import fftn
from numpy import sqrt, zeros, conj, pi, arange, ones, convolve

#------------------------------------------------------------------------------

def movingaverage(interval, window_size):
    window= ones(int(window_size))/float(window_size)
    return convolve(interval, window, 'same')

#------------------------------------------------------------------------------

def compute_tke_spectrum_1d(u,lx,ly,lz,smooth):
  """
  Given a velocity field u this function computes the kinetic energy
  spectrum of that velocity field in spectral space. This procedure consists of the 
  following steps:
  1. Compute the spectral representation of u using a fast Fourier transform.
  This returns uf (the f stands for Fourier)
  2. Compute the point-wise kinetic energy Ef (kx, ky, kz) = 1/2 * (uf)* conjugate(uf)
  3. For every wave number triplet (kx, ky, kz) we have a corresponding spectral kinetic energy 
  Ef(kx, ky, kz). To extract a one dimensional spectrum, E(k), we integrate Ef(kx,ky,kz) over
  the surface of a sphere of radius k = sqrt(kx^2 + ky^2 + kz^2). In other words
  E(k) = sum( E(kx,ky,kz), for all (kx,ky,kz) such that k = sqrt(kx^2 + ky^2 + kz^2) ).

  Parameters:
  -----------  
  u: 3D array
    The x-velocity component.
  v: 3D array
    The y-velocity component.
  w: 3D array
    The z-velocity component.    
  lx: float
    The domain size in the x-direction.
  ly: float
    The domain size in the y-direction.
  lz: float
    The domain size in the z-direction.
  smooth: boolean
    A boolean to smooth the computed spectrum for nice visualization.
  """
  nx = len(u[:,0,0])
  ny = len(u[0,:,0])
  nz = len(u[0,0,:])
  
  nt= nx*ny*nz
  n = max(nx,ny,nz) #int(np.round(np.power(nt,1.0/3.0)))
  
  uh = fftn(u)/nt
  
  tkeh = zeros((nx,ny,nz))
  tkeh = 0.5*(uh*conj(uh)).real
  

  l = max(lx,ly,lz)  
  
  knorm = 2.0*pi/l
  
  kxmax = nx/2
  kymax = ny/2
  kzmax = nz/2
  
  wave_numbers = knorm*arange(0,n)
  
  tke_spectrum = zeros(len(wave_numbers))
  
  for kx in range(nx):
    rkx = kx
    if (kx > kxmax):
      rkx = rkx - (nx)
    for ky in range(ny):
      rky = ky
      if (ky>kymax):
        rky=rky - (ny)
      for kz in range(nz):        
        rkz = kz
        if (kz>kzmax):
          rkz = rkz - (nz)
        rk = sqrt(rkx*rkx + rky*rky + rkz*rkz)
        k = int(np.round(rk))
        tke_spectrum[k] = tke_spectrum[k] + tkeh[kx,ky,kz]

  tke_spectrum = tke_spectrum/knorm
  if smooth:
    tkespecsmooth = movingaverage(tke_spectrum, 5) #smooth the spectrum
    tkespecsmooth[0:4] = tke_spectrum[0:4] # get the first 4 values from the original data
    tke_spectrum = tkespecsmooth

  knyquist = knorm*min(nx,ny,nz)/2 

  return knyquist, wave_numbers, tke_spectrum

#------------------------------------------------------------------------------

def compute_tke_spectrum(u,v,w,lx,ly,lz,smooth):
  """
  Given a velocity field u, v, w, this function computes the kinetic energy
  spectrum of that velocity field in spectral space. This procedure consists of the 
  following steps:
  1. Compute the spectral representation of u, v, and w using a fast Fourier transform.
  This returns uf, vf, and wf (the f stands for Fourier)
  2. Compute the point-wise kinetic energy Ef (kx, ky, kz) = 1/2 * (uf, vf, wf)* conjugate(uf, vf, wf)
  3. For every wave number triplet (kx, ky, kz) we have a corresponding spectral kinetic energy 
  Ef(kx, ky, kz). To extract a one dimensional spectrum, E(k), we integrate Ef(kx,ky,kz) over
  the surface of a sphere of radius k = sqrt(kx^2 + ky^2 + kz^2). In other words
  E(k) = sum( E(kx,ky,kz), for all (kx,ky,kz) such that k = sqrt(kx^2 + ky^2 + kz^2) ).

  Parameters:
  -----------  
  u: 3D array
    The x-velocity component.
  v: 3D array
    The y-velocity component.
  w: 3D array
    The z-velocity component.    
  lx: float
    The domain size in the x-direction.
  ly: float
    The domain size in the y-direction.
  lz: float
    The domain size in the z-direction.
  smooth: boolean
    A boolean to smooth the computed spectrum for nice visualization.
  """
  nx = len(u[:,0,0])
  ny = len(v[0,:,0])
  nz = len(w[0,0,:])
  
  nt= nx*ny*nz
  n = nx #int(np.round(np.power(nt,1.0/3.0)))
  
  uh = fftn(u)/nt
  vh = fftn(v)/nt
  wh = fftn(w)/nt
  
  tkeh = zeros((nx,ny,nz))
  tkeh = 0.5*(uh*conj(uh) + vh*conj(vh) + wh*conj(wh)).real
  
  k0x = 2.0*pi/lx
  k0y = 2.0*pi/ly
  k0z = 2.0*pi/lz
  
  knorm = (k0x + k0y + k0z)/3.0
  
  kxmax = nx/2
  kymax = ny/2
  kzmax = nz/2
  
  wave_numbers = knorm*arange(0,n)
  
  tke_spectrum = zeros(len(wave_numbers))
  
  for kx in range(nx):
    rkx = kx
    if (kx > kxmax):
      rkx = rkx - (nx)
    for ky in range(ny):
      rky = ky
      if (ky>kymax):
        rky=rky - (ny)
      for kz in range(nz):        
        rkz = kz
        if (kz>kzmax):
          rkz = rkz - (nz)
        rk = sqrt(rkx*rkx + rky*rky + rkz*rkz)
        k = int(np.round(rk))
        tke_spectrum[k] = tke_spectrum[k] + tkeh[kx,ky,kz]

  tke_spectrum = tke_spectrum/knorm
#  tke_spectrum = tke_spectrum[1:]
#  wave_numbers = wave_numbers[1:]
  if smooth:
    tkespecsmooth = movingaverage(tke_spectrum, 5) #smooth the spectrum
    tkespecsmooth[0:4] = tke_spectrum[0:4] # get the first 4 values from the original data
    tke_spectrum = tkespecsmooth

  knyquist = knorm*min(nx,ny,nz)/2 

  return knyquist, wave_numbers, tke_spectrum

#------------------------------------------------------------------------------
