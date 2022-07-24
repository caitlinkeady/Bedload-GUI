# -*- coding: utf-8 -*-
"""
FUNCTIONS FOR BEDLOAD MODEL
author: Caitlin Keady
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import odr
from scipy.stats import gamma
from scipy.optimize import curve_fit


def power_func(x, a, b, c):
    return a*(x-b)**c

# Stage-Discharge rating curve
def stageQ_RC(bedload, stage_disc, a,b,c):
    stage = []
    for i in np.arange(len(bedload)):
        if a !=0:
            stage = np.append(stage, power_func(bedload.iloc[i,0], a,b,c))
        elif min(stage_disc['Discharge']) < bedload.iloc[i,0] < max(stage_disc['Discharge']):
            stage_disc = stage_disc.sort_values(by='Discharge')
            stage = np.append(stage, np.interp(bedload.iloc[i,0], stage_disc.iloc[:,0], stage_disc.iloc[:,1]))
        else:
            popt, pcov = curve_fit(power_func, stage_disc['Discharge'], stage_disc['Stage'], maxfev=20000)
            stage = np.append(stage, power_func(bedload.iloc[i,0], *popt))
    return stage

# Get average depth and depth dist from stage & XS
def depth_dist(stage, xs):
    depth_alpha = []
    avg_depth = []
    for i in np.arange(len(stage)):
        depths = []
        for j in np.arange(len(xs)):
            if xs.iloc[j,2] == 1:
                datum = xs['Elevation'][xs[xs['Datum'] == 1].index[0]]
                depths = np.append(depths, stage[i] + datum - xs.iloc[j,1])
            else: 
                depths = np.append(depths, 0)
        depths = list(filter(lambda a: a > 0, depths))
        alpha, loc, scale = gamma.fit(depths, floc=0)
        depth_alpha = np.append(depth_alpha, alpha)
        avg_depth = np.append(avg_depth, np.mean(depths))
    stage_depth = pd.DataFrame(data = {'Stage': stage, 'Depth alpha': depth_alpha, 'Avg depth': avg_depth})
    return stage_depth


# Function to compute dimensionless shear stress using specific gravity
def dimensionless_tau(depth, grain_size, slope, Sg = 2.65):
    return (depth * slope) / ((Sg - 1) * grain_size)

# Function to compute dimensionless SS from SS gamma dist
def SS_from_dist(depth_a, depth, slope, SS_val):
    SS_a = 0.275 * depth_a
    SS_beta = 1000 * 9.81 * depth * slope / SS_a
    return gamma.pdf(SS_val, SS_a, loc=0, scale=SS_beta)


# Function to compute dimensionless bedload
def dimensionless_bedload(Qbi, Fi, Pi, depth, slope, Sg = 2.65):
    return ((Sg - 1) * Qbi * Pi) / (Fi * 9.81**(1/2) * (depth * slope)**(3/2))


# Function to process data by removing NANs and 0s
def process(x_data, y_data):
    mask = ~np.isnan(y_data) & ~np.isinf(y_data) & y_data > 0
    x_data = x_data[mask]
    y_data = y_data[mask]
    if len(y_data) > 0:
        x_data, y_data = zip(*sorted(zip(x_data, y_data)))  # Sort data by x value
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)
    return x_data, y_data


# Function to remove outliers
def remove_outliers(x_data, y_data, sd):
    new_x = process(x_data, y_data)[0]
    new_y = process(x_data, y_data)[1]
    x_std = np.std(new_x)
    x_mean = np.mean(new_x)
    mask = new_x > x_mean -sd*x_std
    new_x = new_x[mask]
    new_y = new_y[mask]
    return new_x, new_y

# Linear function for orthogonal regression
def f(B, x):
    return B[0]*x + B[1]

# plotting orthogonal regression on a log-log plot
def orth_regress(xdata, ydata, ref_bedload = 0.002):
    linear = odr.Model(f)
    mydata = odr.Data(*np.log10(process(xdata, ydata)))
    myodr = odr.ODR(mydata, linear, beta0=[1., 2.])
    output = myodr.run()
    if len(process(xdata, ydata)[0]) >4 or len(process(xdata, ydata)[1]) >4:
        if output.beta[0] < 3:
            plt.plot(process(xdata,ydata)[0], 10**f(output.beta, np.log10(process(xdata,ydata)[0])), 'red')
        else: 
            plt.plot(process(xdata,ydata)[0], 10**f(output.beta, np.log10(process(xdata,ydata)[0])), 'k')
    bed_ints = 10**((np.log10(ref_bedload) - output.beta[1]) / output.beta[0])
    return bed_ints, output.beta[0], output.beta[1]

# Plot orthogonal regression on filtered data
def filtered_orthreg(xdata, ydata, ref_bedload = 0.002):
   linear = odr.Model(f)
   mydata = odr.Data(np.log10(xdata), np.log10(ydata))
   myodr = odr.ODR(mydata, linear, beta0=[1., 2.])
   output = myodr.run()
   plt.plot(xdata, 10**f(output.beta, np.log10(xdata)), 'k')
   bed_ints = 10**((np.log10(ref_bedload) - output.beta[1]) / output.beta[0])
   return bed_ints, output.beta[0], output.beta[1]


# plot linear regression on log-log plot with estimated slopes
def slope_estimator(xdata, ydata, slope, intercept, ref_bedload = 0.002):
     new_y = 10**(np.log10(process(xdata,ydata)[0]) * slope + intercept)
     if slope < 3:
        plt.plot(process(xdata,ydata)[0], new_y, 'red')
     else:
        plt.plot(process(xdata,ydata)[0], new_y, 'k')
     bed_ints = 10**((np.log10(ref_bedload) - intercept) / slope)
     return bed_ints