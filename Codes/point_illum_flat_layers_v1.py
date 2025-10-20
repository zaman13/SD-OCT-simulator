#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 19:46:49 2025

@author: Mohammad Asif Zaman

SD_OCT simulator code (point illumination)

All units in SI unless otherwise stated

"""
%reset -f

import numpy as np
import matplotlib.pyplot as py
from scipy.interpolate import interp1d
from scipy.signal import windows


py.close('all')



# =============================================================================
# Inputs
# =============================================================================
# layer_pos = [60e-6, 140e-6,  250e-6]  # in meters
# layer_n = [1.2, 1.4,  1.2]
# layer_d = np.diff(layer_pos)
# layer_d = np.append(layer_d,220e-2)
# # layer_d[1] = 30e-6


layer_pos = [60e-6, 240e-6, 300e-6, 450e-6]  # in meters
layer_n = [1.2, 1.1, 1.25, 1.35]
layer_d = np.diff(layer_pos)
layer_d = np.append(layer_d,220e-2)
# layer_d[1] = 30e-6



n_air = 1

# vertical axis limit in um for plotting
zmin = 10
zmax = 500
N_x = 400   # x axis (line illumination direction) points
N_crop = 10 # crop first N_crop points to avoid DC component
# =============================================================================



# =============================================================================
# Parameters
# =============================================================================

# light source
flag_consider_layer_d = 'n'
lambda_center = 850e-9  # center wavelengt 
lambda_bw = 100e-9      # bandwidth 
lambda_min = lambda_center - lambda_bw
lambda_max = lambda_center + lambda_bw 
lambda_sigma = lambda_bw/2.355
N_lambda = 2048
zp_factor = 16          # zero padding factor
N_full = N_lambda*zp_factor

noise_amp_fct = 0.1


# =============================================================================



# =============================================================================
# Draw geometry function (2D)
# =============================================================================
def draw_geo(N_x, layer_pos):
    x_fill = np.linspace(0, N_x, N_x)
    for m in range(len(layer_pos)):
        py.plot([0,N_x], [layer_pos[m]*1e6, layer_pos[m]*1e6])
        # if m > 0:
            # py.fill_between(x_fill, layer_pos[m]*1e6, layer_pos[m-1]*1e6,  alpha=0.3)
        if flag_consider_layer_d == 'y':
            if m == len(layer_pos)-1:
                py.fill_between(x_fill, layer_pos[m]*1e6, zmax,  alpha=0.05)
            else:
                py.fill_between(x_fill, layer_pos[m]*1e6, layer_pos[m+1]*1e6,  alpha=0.05)
            
    return 0
 # =============================================================================


# =============================================================================
# Illumination spectra with lambda and k definitions, and k linearization
# =============================================================================

lambda_set = np.linspace(lambda_min, lambda_max, N_lambda)
lambda_set_full = np.linspace(lambda_min, lambda_max, N_full)

k_set_nonlinear = 2*np.pi/lambda_set
k_set_linear = np.linspace(np.min(k_set_nonlinear), np.max(k_set_nonlinear), N_full)

illum_spectra = np.exp(-(lambda_set - lambda_center)**2/(2*lambda_sigma**2))

illum_spectra_k_function = interp1d(k_set_nonlinear, illum_spectra, kind = 'linear')

illum_spectra_k  = illum_spectra_k_function(k_set_linear)
# =============================================================================



py.figure(1, figsize = (14,6))

py.subplot(131)
draw_geo(N_x, layer_pos)

py.ylim([zmin, zmax]) 
py.xlim([0, N_x]) 
py.ylabel('Depth, z [$\\mu$m]')
py.xlabel('Line direction, x [$\\mu$m]')
py.title('Input geometry')

py.gca().invert_yaxis()

py.subplot(132)
py.semilogy(lambda_set_full*1e9, illum_spectra_k) 
# py.plot(lambda_set_full*1e9, illum_spectra_k) 
py.xlabel('Wavelength, $\\lambda$ [nm]')
py.ylabel('Illumination Spectra')


# =============================================================================
# Field calculations
# =============================================================================

ref_field = np.ones_like(k_set_linear)
sample_field = np.zeros_like(k_set_linear, dtype = complex)  # initialization
n_prev = n_air

for m in range(len(layer_pos)):
    n_curr = layer_n[m]
    r = (n_curr - n_prev)/(n_curr + n_prev)
    
    if flag_consider_layer_d == 'y':
        path_diff = layer_pos[m]*n_curr * 2.0
    else:
        path_diff = layer_pos[m]*n_air * 2.0   # reflecting planes are infinitely thin. So path difference calculation only takes into account air.
    
    sample_field = sample_field + r* np.exp(1j*k_set_linear*path_diff)
    n_prev = n_curr
    
interference_signal = illum_spectra_k * (np.abs(sample_field + ref_field)**2)


# Adding noise
noise = noise_amp_fct * np.random.randn(len(interference_signal)) * np.max(interference_signal[N_crop:-1])
interference_signal = interference_signal + noise

# apodization
apod_window = windows.hann(N_lambda*zp_factor)
interference_signal_windowed = (interference_signal -np.mean(interference_signal)) * apod_window


py.subplot(133)
py.semilogy(lambda_set_full*1e9, interference_signal) 
# py.plot(lambda_set_full*1e9, interference_signal) 
py.xlabel('Wavelength, $\\lambda$ [nm]')
py.ylabel('Inteference Spectra')

py.tight_layout() 
# =============================================================================




# =============================================================================
# Depth axis (z) formulation
# =============================================================================
delta_k = k_set_linear[1] - k_set_linear[0]   # setp size in k space
ind_seq = np.arange(N_full//2)                # 1, 2, 3... upto N_full/2   

opl_set = ind_seq * np.pi/(delta_k*N_full)
z_set = ind_seq * np.pi/(delta_k*n_air*N_full)    # depth axis (z) positions corresponding to the k space if the sample had refractive index of air. 
                                                  # Use n_avg of the layers for a rough approximation. The loop below is the accurate general approach.

if flag_consider_layer_d == 'y':
    # calculate the cumulative OPL along the layers
    layer_opl = np.multiply(layer_n, layer_d)
    layer_opl_cum = np.cumsum(layer_opl)
    
    
    for m1 in range(len(opl_set)):  # loop over opls
        for m2 in range(len(layer_opl_cum)):
            if opl_set[m1] <= layer_opl_cum[m2]:    # check between which layers the OPL falls
                # print(f'In layer {m2}')
                z_set[m1] = opl_set[m1]/layer_n[m2] # find z by scaling the opl with corresponding refractive index
                break

# note that the calculated z_set is not longer linearly spaced. Define a linearly spaced set here
# this becomes important later. When using imshow to plot the 2D reconstruction, we cannot explicity define the non-uniform vertical axis.
# imshow assumes uniformly spaced vertical axis when using the extent command, hence the results appear wrong. it is necessary to convert all
# signals back to a linear uniformly spaced z axis before plotting
z_set_lin = np.linspace(np.min(z_set), np.max(z_set), len(z_set))        
# =============================================================================




# =============================================================================
# OCT reconstruction
# =============================================================================
fft_signal = np.fft.ifft(interference_signal_windowed)
fft_signal_mag = np.abs( fft_signal[0:N_full//2])
fft_signal_mag = fft_signal_mag/np.max(fft_signal_mag[N_crop:-1])
# fft_signal_mag = fft_signal_mag/np.max(fft_signal_mag)

# if flag_consider_layer_d == 'y':
# interpolating fft_signal_mag along a uniformly spaced depth axis
fft_sig_interp = interp1d(z_set, fft_signal_mag)   # interpolation funciton
fft_signal_mag = fft_sig_interp(z_set_lin)         # interpolated results



# Making 2D reconstruction and adding additional noise
N_tile = N_x
fft_signal_mag_2D = np.tile(fft_signal_mag.reshape(-1,1), N_tile)
noise2D = np.random.randn(*fft_signal_mag_2D.shape)*noise_amp_fct
fft_signal_mag_2D = fft_signal_mag_2D + noise2D
# =============================================================================


# =============================================================================
# plotting geometry, OCT reconstruction and 2D representation
# =============================================================================

py.figure(3, figsize = (14,6))



py.subplot(131)
draw_geo(N_x, layer_pos)

    
py.ylim([zmin, zmax]) 
py.xlim([0, N_x]) 
py.ylabel('Depth, z [$\\mu$m]')
py.xlabel('Line direction, x [$\\mu$m]')
py.title('Input geometry')

py.gca().invert_yaxis()


# 1D reconstruction amplitdue

py.subplot(132)
# py.plot(fft_signal_mag, z_set*1e6)
py.plot(fft_signal_mag, z_set_lin*1e6)
py.ylim([zmin, zmax]) 
py.xlim([0, np.max(fft_signal_mag[N_crop:-1])*1.6])
py.ylabel('Depth, z [$\\mu$m]')
py.xlabel('OCT reconstruction amplitude')
py.title('1D reconstruction')

py.gca().invert_yaxis()

# 2D reconstruction amplitude
py.subplot(133)
py.imshow((fft_signal_mag_2D),extent=[0, N_x, z_set[0]*1e6, z_set[-1]*1e6], aspect='auto', origin='lower', cmap = 'gray')
py.ylim([zmin, zmax]) 
py.clim([0, np.max(fft_signal_mag[N_crop:-1])*1])
py.ylabel('Depth, z [$\\mu$m]')
py.xlabel('Line direction, x [$\\mu$m]')
py.title('2D reconstruction')
# py.plot(fft_signal_mag_2D[:,10])

py.gca().invert_yaxis()
py.tight_layout() 

py.show()
















