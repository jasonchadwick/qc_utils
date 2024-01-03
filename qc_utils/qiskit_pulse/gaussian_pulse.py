import numpy as np
import math

def gaussian_area(duration, sigma, amp=1, dt=2/9*1e-9):
    return dt * amp * (np.sqrt(2*np.pi) * sigma * np.exp((duration+2)**2/(8*sigma**2))*math.erf(duration/(2*np.sqrt(2)*sigma))-duration) / (np.exp((duration+2)**2/(8*sigma**2))-1)

def get_gaussian_amp_for_target_area(duration, sigma, target_area, dt=2/9*1e-9):
    amp_1_area = gaussian_area(duration, sigma, dt=dt)
    return target_area / amp_1_area

def square_gaussian_area(duration, sigma, width, amp=1, dt=2/9*1e-9):
    risefall = duration - width
    return amp * width * dt + gaussian_area(risefall, sigma, amp, dt)

def get_square_gaussian_amp_for_target_area(duration, sigma, width, target_area, dt=2/9*1e-9):
    amp_1_area = square_gaussian_area(duration, sigma, width, dt=dt)
    return target_area / amp_1_area