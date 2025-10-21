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
layer_pos = [100e-6, 140e-6,  420e-6]  # in meters
layer_r = [0.1, 0.1,  0.1]
n = 1.4

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
lambda_center = 850e-9  # center wavelengt 
lambda_bw = 100e-9      # bandwidth 
lambda_min = lambda_center - lambda_bw/2 
lambda_max = lambda_center + lambda_bw/2 
lambda_sigma = lambda_bw/2.355
N_lambda = 2048
zp_factor = 16          # zero padding factor
N_full = N_lambda*zp_factor

noise_amp_fct = 0.1


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



py.figure(1, figsize = (10,6))
py.subplot(121)
py.semilogy(lambda_set_full*1e9, illum_spectra_k) 
py.xlabel('Wavelength, $\\lambda$ [nm]')
py.ylabel('Illumination Spectra')


# =============================================================================
# Field calculations
# =============================================================================
ref_field = np.ones_like(k_set_linear)
sample_field = np.zeros_like(k_set_linear, dtype = complex)  # initialization

for m in range(len(layer_pos)):
    path_diff = layer_pos[m]*n * 2.0
    sample_field = sample_field + layer_r[m]* np.exp(1j*k_set_linear*path_diff)


interference_signal = illum_spectra_k * (np.abs(sample_field + ref_field)**2)


# Adding noise
noise = noise_amp_fct * np.random.randn(len(interference_signal)) * np.max(interference_signal[N_crop:-1])
interference_signal = interference_signal + noise

# apodization
apod_window = windows.hann(N_lambda*zp_factor)
interference_signal_windowed = (interference_signal -np.mean(interference_signal)) * apod_window


py.subplot(122)
py.semilogy(lambda_set_full*1e9, interference_signal) 
py.xlabel('Wavelength, $\\lambda$ [nm]')
py.ylabel('Inteference Spectra')
# =============================================================================


# =============================================================================
# OCT reconstruction
# =============================================================================
fft_signal = np.fft.ifft(interference_signal_windowed)
fft_signal_mag = np.abs( fft_signal[0:N_full//2])
fft_signal_mag = fft_signal_mag/np.max(fft_signal_mag)


# Making 2D reconstruction and adding additional noise
N_tile = N_x
fft_signal_mag_2D = np.tile(fft_signal_mag.reshape(-1,1), N_tile)
noise2D = np.random.randn(*fft_signal_mag_2D.shape)*noise_amp_fct
fft_signal_mag_2D = fft_signal_mag_2D + noise2D



delta_k = k_set_linear[1] - k_set_linear[0]   # setp size in k space
ind_seq = np.arange(N_full//2)                # 1, 2, 3... upto N_full/2   
z_set = ind_seq * np.pi/(delta_k*n*N_full)    # depth axis (z) positions corresponding to the k space
# =============================================================================




# =============================================================================
# plotting geometry, OCT reconstruction and 2D representation
# =============================================================================

py.figure(3, figsize = (14,6))



py.subplot(131)
for m in range(len(layer_pos)):
    py.plot([0,N_x], [layer_pos[m]*1e6, layer_pos[m]*1e6])

py.ylim([zmin, zmax]) 
py.xlim([0, N_x]) 
py.ylabel('Depth, z [$\\mu$m]')
py.xlabel('Line direction, x [$\\mu$m]')
py.title('Input geometry')
# 1D reconstruction amplitdue

py.subplot(132)
py.plot(fft_signal_mag, z_set*1e6)
py.ylim([zmin, zmax]) 
py.xlim([0, np.max(fft_signal_mag[N_crop:-1])*1.2])
py.ylabel('Depth, z [$\\mu$m]')
py.xlabel('OCT reconstruction amplitude')
py.title('1D reconstruction')


# 2D reconstruction amplitude
py.subplot(133)
py.imshow((fft_signal_mag_2D),extent=[0, N_x, z_set[0]*1e6, z_set[-1]*1e6], aspect='auto', origin='lower', cmap = 'gray')
py.ylim([zmin, zmax]) 
# py.clim([0, np.max(fft_signal_mag[10:-1])*1.2])
py.ylabel('Depth, z [$\\mu$m]')
py.xlabel('Line direction, x [$\\mu$m]')
py.title('2D reconstruction')
# py.plot(fft_signal_mag_2D[:,10])

py.show()
















