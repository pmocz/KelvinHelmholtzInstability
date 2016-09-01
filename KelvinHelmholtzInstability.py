#  Compressible Fluid Dynamics - Finite volume
#  Philip Mocz (2016)   pmocz@cfa.harvard.edu
#  Kelvin Helmholtz Instability (KHI)

import matplotlib.pyplot as plt
import numpy as np

# Parameters
Nx = 128
Ny = 128
boxSizeX = 1.
boxSizeY = 1.
dx = boxSizeX / Nx
dy = boxSizeY / Ny
vol = dx*dy
Y, X = np.meshgrid( np.linspace(0.5*dy, boxSizeY-0.5*dy, Ny), np.linspace(0.5*dx, boxSizeX-0.5*dx, Nx) )
courant_fac = 0.4
t = 0
tEnd = 2
tOut = 0.01
useSlopeLimiting = False

# Set initial conditions for KHI
w0 = 0.1
sigma = 0.05/np.sqrt(2.)
gamma = 5/3.
rho = 1. + (np.abs(Y-0.5) < 0.25)
vx = -0.5 + (np.abs(Y-0.5)<0.25)
vy = w0*np.sin(4*np.pi*X) * ( np.exp(-(Y-0.25)**2/(2 * sigma**2)) + np.exp(-(Y-0.75)**2/(2*sigma**2)) )
vz = 0*X
P = 0*X + 2.5

# directions for np.roll() 
R = -1   # right
L = 1    # left

# quick plotting function 
def myPlot():
  plt.clf()
  plt.imshow(rho.T)
  plt.clim(0.8, 2.2)
  ax = plt.gca()
  ax.invert_yaxis()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  plt.draw()
 
myPlot()
outputCount = 1

# get conserved variables
Mass = rho * vol
Momx = rho * vx * vol
Momy = rho * vy * vol
Energy = (P/(gamma-1) + 0.5*rho*(vx**2+vy**2))*vol

# Main loop
while (t < tEnd):
  # get primitive variables
  rho = Mass / vol
  vx = Momx / rho / vol
  vy = Momy / rho / vol
  P = (Energy/vol - 0.5*rho * (vx**2+vy**2)) * (gamma-1)
  
  # get time step (CFL)
  dt = courant_fac * np.min( np.min([dx,dy]) / (np.sqrt( gamma*P/rho ) + np.sqrt(vx**2+vy**2)) )
  plotThisTurn = False
  if t + dt > outputCount*tOut:
    dt = outputCount*tOut - t
    plotThisTurn = True
    
  # calculate gradients
  rho_gradx = ( np.roll(rho,R,axis=0) - np.roll(rho,L,axis=0) ) / (2.*dx)
  rho_grady = ( np.roll(rho,R,axis=1) - np.roll(rho,L,axis=1) ) / (2.*dy)
  vx_gradx  = ( np.roll(vx,R,axis=0) - np.roll(vx,L,axis=0) ) / (2.*dx)
  vx_grady  = ( np.roll(vx,R,axis=1) - np.roll(vx,L,axis=1) ) / (2.*dy)
  vy_gradx  = ( np.roll(vy,R,axis=0) - np.roll(vy,L,axis=0) ) / (2.*dx)
  vy_grady  = ( np.roll(vy,R,axis=1) - np.roll(vy,L,axis=1) ) / (2.*dy)
  P_gradx   = ( np.roll(P,R,axis=0) - np.roll(P,L,axis=0) ) / (2.*dx)
  P_grady   = ( np.roll(P,R,axis=1) - np.roll(P,L,axis=1) ) / (2.*dy)
  
  # slope limit gradients
  if useSlopeLimiting:
    rho_gradx = np.maximum(0., np.minimum(1., ( (rho-np.roll(rho,L,axis=0))/dx)/(rho_gradx + 1.0e-8*(rho_gradx==0)))) * rho_gradx
    rho_gradx = np.maximum(0., np.minimum(1., (-(rho-np.roll(rho,R,axis=0))/dx)/(rho_gradx + 1.0e-8*(rho_gradx==0)))) * rho_gradx
    rho_grady = np.maximum(0., np.minimum(1., ( (rho-np.roll(rho,L,axis=1))/dy)/(rho_grady + 1.0e-8*(rho_grady==0)))) * rho_grady
    rho_grady = np.maximum(0., np.minimum(1., (-(rho-np.roll(rho,R,axis=1))/dy)/(rho_grady + 1.0e-8*(rho_grady==0)))) * rho_grady
    vx_gradx  = np.maximum(0., np.minimum(1., ( (vx-np.roll(vx,L,axis=0))/dx)  /(vx_gradx  + 1.0e-8*(vx_gradx ==0)))) * vx_gradx
    vx_gradx  = np.maximum(0., np.minimum(1., (-(vx-np.roll(vx,R,axis=0))/dx)  /(vx_gradx  + 1.0e-8*(vx_gradx ==0)))) * vx_gradx
    vx_grady  = np.maximum(0., np.minimum(1., ( (vx-np.roll(vx,L,axis=1))/dy)  /(vx_grady  + 1.0e-8*(vx_grady ==0)))) * vx_grady
    vx_grady  = np.maximum(0., np.minimum(1., (-(vx-np.roll(vx,R,axis=1))/dy)  /(vx_grady  + 1.0e-8*(vx_grady ==0)))) * vx_grady
    vy_gradx  = np.maximum(0., np.minimum(1., ( (vy-np.roll(vy,L,axis=0))/dx)  /(vy_gradx  + 1.0e-8*(vy_gradx ==0)))) * vy_gradx
    vy_gradx  = np.maximum(0., np.minimum(1., (-(vy-np.roll(vy,R,axis=0))/dx)  /(vy_gradx  + 1.0e-8*(vy_gradx ==0)))) * vy_gradx
    vy_grady  = np.maximum(0., np.minimum(1., ( (vy-np.roll(vy,L,axis=1))/dy)  /(vy_grady  + 1.0e-8*(vy_grady ==0)))) * vy_grady
    vy_grady  = np.maximum(0., np.minimum(1., (-(vy-np.roll(vy,R,axis=1))/dy)  /(vy_grady  + 1.0e-8*(vy_grady ==0)))) * vy_grady
    P_gradx   = np.maximum(0., np.minimum(1., ( (P-np.roll(P,L,axis=0))/dx)    /(P_gradx   + 1.0e-8*(P_gradx  ==0)))) * P_gradx
    P_gradx   = np.maximum(0., np.minimum(1., (-(P-np.roll(P,R,axis=0))/dx)    /(P_gradx   + 1.0e-8*(P_gradx  ==0)))) * P_gradx
    P_grady   = np.maximum(0., np.minimum(1., ( (P-np.roll(P,L,axis=1))/dy)    /(P_grady   + 1.0e-8*(P_grady  ==0)))) * P_grady
    P_grady   = np.maximum(0., np.minimum(1., (-(P-np.roll(P,R,axis=1))/dy)    /(P_grady   + 1.0e-8*(P_grady  ==0)))) * P_grady

  # extrapolate to cell faces (in time & space)
  rho_prime = rho - 0.5*dt *( vx * rho_gradx + rho * vx_gradx + vy * rho_grady + rho * vy_grady)
  rho_XL = rho_prime - rho_gradx * dx/2.  
  rho_XL = np.roll(rho_XL,R,axis=0)
  rho_XR = rho_prime + rho_gradx * dx/2.
  rho_YL = rho_prime - rho_grady * dy/2.  
  rho_YL = np.roll(rho_YL,R,axis=1)
  rho_YR = rho_prime + rho_grady * dy/2.
  vx_prime = vx - 0.5*dt * ( vx * vx_gradx + vy * vx_grady + (1/rho) * P_gradx )
  vx_XL = vx_prime - vx_gradx * dx/2.  
  vx_XL = np.roll(vx_XL,R,axis=0)
  vx_XR = vx_prime + vx_gradx * dx/2.
  vx_YL = vx_prime - vx_grady * dy/2. 
  vx_YL = np.roll(vx_YL,R,axis=1)
  vx_YR = vx_prime + vx_grady * dy/2.
  vy_prime = vy - 0.5*dt * ( vx * vy_gradx + vy * vy_grady + (1/rho) * P_grady )
  vy_XL = vy_prime - vy_gradx * dx/2.
  vy_XL = np.roll(vy_XL,R,axis=0)
  vy_XR = vy_prime + vy_gradx * dx/2.
  vy_YL = vy_prime - vy_grady * dy/2. 
  vy_YL = np.roll(vy_YL,R,axis=1)
  vy_YR = vy_prime + vy_grady * dy/2.
  P_prime = P - 0.5*dt * ( gamma*P * (vx_gradx + vy_grady)  + vx * P_gradx + vy * P_grady )
  P_XL = P_prime - P_gradx * dx/2.  
  P_XL = np.roll(P_XL,R,axis=0)
  P_XR = P_prime + P_gradx * dx/2.
  P_YL = P_prime - P_grady * dy/2.
  P_YL = np.roll(P_YL,R,axis=1)
  P_YR = P_prime + P_grady * dy/2.

  # compute star (averaged) states
  rho_Xstar = 0.5*(rho_XL + rho_XR)
  rho_Ystar = 0.5*(rho_YL + rho_YR)
  momx_Xstar = 0.5*(rho_XL * vx_XL + rho_XR * vx_XR)
  momx_Ystar = 0.5*(rho_YL * vx_YL + rho_YR * vx_YR)
  momy_Xstar = 0.5*(rho_XL * vy_XL + rho_XR * vy_XR)
  momy_Ystar = 0.5*(rho_YL * vy_YL + rho_YR * vy_YR)
  en_Xstar = 0.5*( P_XL/(gamma-1)+0.5*rho_XL * (vx_XL**2+vy_XL**2) + P_XR/(gamma-1)+0.5*rho_XR * (vx_XR**2+vy_XR**2))
  en_Ystar = 0.5*( P_YL/(gamma-1)+0.5*rho_YL * (vx_YL**2+vy_YL**2) + P_YR/(gamma-1)+0.5*rho_YR * (vx_YR**2+vy_YR**2))

  P_Xstar = (gamma-1)*(en_Xstar-0.5*(momx_Xstar**2+momy_Xstar**2)/rho_Xstar)
  P_Ystar = (gamma-1)*(en_Ystar-0.5*(momx_Ystar**2+momy_Ystar**2)/rho_Ystar)
  
  # compute fluxes (local Lax-Friedrichs/Rusanov)
  flux_rho_X = momx_Xstar
  flux_rho_Y = momy_Ystar
  flux_momx_X = momx_Xstar**2/rho_Xstar + P_Xstar
  flux_momx_Y = momy_Ystar * momx_Ystar/rho_Ystar
  flux_momy_X = momx_Xstar * momy_Xstar/rho_Xstar
  flux_momy_Y = momy_Ystar**2/rho_Ystar + P_Ystar
  flux_en_X = (en_Xstar+P_Xstar) * momx_Xstar/rho_Xstar
  flux_en_Y = (en_Ystar+P_Ystar) * momy_Ystar/rho_Ystar
  
  C = np.sqrt(gamma*P_XL/rho_XL) + np.abs(vx_XL)
  C = np.maximum( C, np.sqrt(gamma*P_XR/rho_XR) + np.abs(vx_XR))
  C = np.maximum( C, np.sqrt(gamma*P_YL/rho_YL) + np.abs(vy_YL))
  C = np.maximum( C, np.sqrt(gamma*P_YR/rho_YR) + np.abs(vy_YR))
  
  flux_rho_X = flux_rho_X - C * 0.5 * (rho_XL - rho_XR)
  flux_rho_Y = flux_rho_Y - C * 0.5 * (rho_YL - rho_YR)
  flux_momx_X = flux_momx_X - C * 0.5 * (rho_XL * vx_XL - rho_XR * vx_XR)
  flux_momx_Y = flux_momx_Y - C * 0.5 * (rho_YL * vx_YL - rho_YR * vx_YR)
  flux_momy_X = flux_momy_X - C * 0.5 * (rho_XL * vy_XL - rho_XR * vy_XR)
  flux_momy_Y = flux_momy_Y - C * 0.5 * (rho_YL * vy_YL - rho_YR * vy_YR)
  flux_en_X = flux_en_X - C * 0.5 * ( P_XL/(gamma-1)+0.5*rho_XL * (vx_XL**2+vy_XL**2) - (P_XR/(gamma-1)+0.5*rho_XR * (vx_XR**2+vy_XR**2)))
  flux_en_Y = flux_en_Y - C * 0.5 * ( P_YL/(gamma-1)+0.5*rho_YL * (vx_YL**2+vy_YL**2) - (P_YR/(gamma-1)+0.5*rho_YR * (vx_YR**2+vy_YR**2)))

  # update solution
  Mass = Mass - dt * dy * flux_rho_X
  Mass = Mass + dt * dy * np.roll(flux_rho_X,L,axis=0)
  Mass = Mass - dt * dx * flux_rho_Y
  Mass = Mass + dt * dx * np.roll(flux_rho_Y,L,axis=1)
  Momx = Momx - dt * dy * flux_momx_X
  Momx = Momx + dt * dy * np.roll(flux_momx_X,L,axis=0)
  Momx = Momx - dt * dx * flux_momx_Y
  Momx = Momx + dt * dx * np.roll(flux_momx_Y,L,axis=1)
  Momy = Momy - dt * dy * flux_momy_X
  Momy = Momy + dt * dy * np.roll(flux_momy_X,L,axis=0)
  Momy = Momy - dt * dx * flux_momy_Y
  Momy = Momy + dt * dx * np.roll(flux_momy_Y,L,axis=1)
  Energy = Energy - dt * dy * flux_en_X
  Energy = Energy + dt * dy * np.roll(flux_en_X,L,axis=0)
  Energy = Energy - dt * dx * flux_en_Y
  Energy = Energy + dt * dx * np.roll(flux_en_Y,L,axis=1)
 
  # advance time
  t += dt
  
  # plot the solution at regular time intervals
  if plotThisTurn:
    print t
    myPlot()
    plt.pause(0.001)
    outputCount += 1
    
plt.show()
