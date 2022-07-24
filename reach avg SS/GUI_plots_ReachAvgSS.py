# -*- coding: utf-8 -*-
"""
FUNCTIONS TO OPERATE BEDLOAD GUI
author: Caitlin Keady
"""

import matplotlib.pyplot as plt
import GUI_functions_ReachAvgSS as fn
import numpy as np
from scipy import stats
import pandas as pd

def run_bedload(all_data, units, a=0,b=0,c=0) :

    # Import, format data, convert units
    bedload_data = pd.read_excel(all_data, 'bedload', skiprows = [1])
    xs = pd.read_excel(all_data, 'xs', skiprows = [1])
    stage_disc = pd.read_excel(all_data, 'stage_discharge', skiprows = [1])
    gsd = pd.read_excel(all_data, 'grain_sizes', skiprows = [1])


    if units == 1:        # Convert English units to SI units for all calcs in GUI
        bedload_data['Discharge'] = bedload_data['Discharge'] * 0.0283
        bedload_data['Bedload'] = bedload_data['Bedload'] * 907.185 / (bedload_data['Width'] * 86400 * 0.3048 * 2650)
        bedload_data['Width'] = bedload_data['Width'] * 0.3048
        
        xs['Distance'] = xs['Distance'] * 0.3048
        xs['Elevation'] = xs['Elevation'] * 0.3048

        stage_disc['Discharge'] = stage_disc['Discharge'] * 0.0283
        stage_disc['Stage'] = stage_disc['Stage'] * 0.3048
        
        gsd['Bins'] = round(gsd['Bins'] * 25.4, 3)
        
        gs_cols  = gsd['Bins'][0:len(gsd)].astype(str)
        new_cols = np.append(['Discharge', 'Bedload', 'Slope', 'Width'], gs_cols)
        bedload_data.columns = new_cols
        
    if units == 2:        # Convert kg/s to m2/s for all calcs in GUI
        bedload_data['Bedload'] = bedload_data['Bedload'] / (2650 * bedload_data['Width'])
        gs_cols = [float(x) for x in list(bedload_data.columns)[4:len(bedload_data.columns)]]
        new_cols = np.append(['Discharge', 'Bedload', 'Slope', 'Width'], gs_cols)
        bedload_data.columns = new_cols

    
    ref_bedload = 0.002
    
    gsd_prop = gsd['Prop']      # grain size distribution percent finer
    gsd_bins = gsd['Bins']      # grain size distribution bins
    
    grain_sizes = []
    for i in np.arange(len(gsd)-2):
        grain_sizes = np.append(grain_sizes, (gsd_bins[i] * gsd_bins[i+1]) ** (1/2) ).tolist()
                    
    bedload_data = bedload_data.dropna()
    
    # Get stage, average depth, and shear stress from rating curve
    stage = fn.stageQ_RC(bedload_data, stage_disc, a,b,c)
    stage_depth = fn.depth_dist(stage, xs)  
    stage_depth['Shear stress'] = 1000 * 9.81 * bedload_data['Slope'] * stage_depth['Avg depth']
    
    bedload_data['Depth'] = stage_depth['Avg depth']
    bedload_data['Shear stress'] = stage_depth['Shear stress']
    
    # Add columns for dimensionless shear stress
    for i in np.arange(len(gsd)-2):
        bedload_data[f'Dimensionless Tau ({str(gsd_bins[i+1])})'] = fn.dimensionless_tau(bedload_data['Depth'], 
                     grain_sizes[i]/1000, bedload_data['Slope'])
        
    # Add columns for dimensionless bedload
    for i in np.arange(len(gsd)-2):
        bedload_data[f'Dimensionless bedload ({str(gsd_bins[i+1])})'] = fn.dimensionless_bedload(bedload_data['Bedload'], 
                     gsd_prop[i+1], bedload_data[f'{str(gsd_bins[i+1])}'], bedload_data['Depth'], bedload_data['Slope'])

   
    # Plot dimensionless shear stress & bedload
    for i in np.arange(len(gsd)-2):                
        globals()[f'x{i+1}'] = bedload_data[f'Dimensionless Tau ({str(gsd_bins[i+1])})']
        globals()[f'y{i+1}'] = bedload_data[f'Dimensionless bedload ({str(gsd_bins[i+1])})']

    points_count = []
    for i in np.arange(len(grain_sizes)):
        points_count = np.append(points_count, sum(globals()[f'y{i+1}'] > 0))
    
    # Generate error messages to warn user of issues with plot 
    if any(points_count <= 4):
        error0 = f'Red points: not enough points to determine best fit line'
        file = open("error0","w")
        file.write(error0)
        file.close()
    else:
        error0 = f' '
        file = open("error0","w")
        file.write(error0)
        file.close()
    
    plt.clf()
    
    colors = ['blue', 'royalblue', 'forestgreen', 'slateblue', 'cyan', 'yellowgreen', 'steelblue', 'aquamarine', 'indigo', 'dodgerblue',
              'blue', 'royalblue', 'forestgreen', 'slateblue', 'cyan', 'yellowgreen', 'steelblue', 'aquamarine', 'indigo', 'dodgerblue']
    
    # Plot dimensionless bedload and shear stress
    for i in np.arange(len(grain_sizes)):
        mask = ~np.isnan(globals()[f'y{i+1}']) & ~np.isinf(globals()[f'y{i+1}']) & globals()[f'y{i+1}'] > 0
        
        if len(globals()[f'y{i+1}'][mask]) <= 4 and len(globals()[f'y{i+1}'][mask]) > 0:
            globals()[f'a{i}'] = plt.scatter(*fn.process(globals()[f'x{i+1}'], globals()[f'y{i+1}']), s=10, facecolors='white', edgecolors='r')
        elif len(globals()[f'y{i+1}'][mask]) == 0:
            globals()[f'a{i}'] = plt.scatter(*fn.process(globals()[f'x{i+1}'], globals()[f'y{i+1}']), s=10, facecolors='white', edgecolors='white')
        else: globals()[f'a{i}'] = plt.scatter(*fn.process(globals()[f'x{i+1}'], globals()[f'y{i+1}']), s=10, color = colors[i])

    plt.yscale('log')
    plt.xscale('log')
    
    all_y = []
    for i in np.arange(len(grain_sizes)):
        all_y = np.concatenate([all_y, globals()[f'y{i+1}']])
        
    ymin = np.nanmin(all_y[np.nonzero(all_y)])
    ymax = np.nanmax(all_y[all_y != np.inf])
    plt.ylim(ymin-0.5*ymin, ymax+0.5*ymax)
    
    
    # Plot line for reference bedload = 0.002
    plt.axhline(y = ref_bedload, color = 'black')
    
    plt.title('Dimensionless Bedload Transport')
    plt.xlabel('Dimensionless shear stress (\u03C4i*)')
    plt.ylabel('Dimensionless bedload transport rate (Wi*)')
    
    handles = []
    labels = []
    for i in np.arange(len(gsd)-2):
        handles = np.append(handles, globals()[f'a{i}'])
        labels = np.append(labels, f'{gsd_bins[i+1]}')
    
    plt.legend(handles, labels, fontsize = 'small', loc=4, ncol=2)
    
    # Intercepts with reference bedload = 0.002
    bed_ints = np.empty([1,0])
    for i in np.arange(len(grain_sizes)):
        bed_ints = np.append(bed_ints, fn.orth_regress(globals()[f'x{i+1}'], globals()[f'y{i+1}'])[0])
        
    
    # Check slopes for outliers
    slopes = np.empty([1,0])
    for i in np.arange(len(grain_sizes)):
        slopes = np.append(slopes, fn.orth_regress(globals()[f'x{i+1}'], globals()[f'y{i+1}'])[1])
        

    # Array of y intercepts of best fit lines
    y_ints = np.empty([1,0])
    for i in np.arange(len(grain_sizes)):
        y_ints = np.append(y_ints, fn.orth_regress(globals()[f'x{i+1}'], globals()[f'y{i+1}'])[2])
        
    # Generate error messages for GUI
    if any(slopes < 3):
        error1 = f'Red line: slope is out of expected range'
        file = open("error1","w")
        file.write(error1)
        file.close()
    else:
        error1 = f' '
        file = open("error1","w")
        file.write(error1)
        file.close()
        
    plt.savefig('bedload', dpi=70)
    plt.clf()
    
    # Save all new parameters
    np.savetxt("bed_ints.csv", bed_ints, delimiter=",")
    np.savetxt("slopes.csv", slopes, delimiter=",")
    np.savetxt("y_ints.csv", y_ints, delimiter=",")
    np.savetxt("grain_sizes.csv", grain_sizes, delimiter=",", header='GS', comments='')
    names = list(stage_depth.columns)
    np.savetxt("stage_depth.csv", stage_depth, delimiter=",", header = ','.join(names), comments='')
    
    names = list(stage_disc.columns)
    np.savetxt("stage_disc.csv", stage_disc, delimiter=",", header = ','.join(names), comments='')
    
    names = list(xs.columns)
    np.savetxt("xs.csv", xs, delimiter=",", header = ','.join(names), comments='')
    
    np.savetxt("gsd_bins.csv", gsd_bins, delimiter=",", header = 'Bins', comments='', fmt='%1.3f')
    np.savetxt("gsd_prop.csv", gsd_prop, delimiter=",", header = 'Prop', comments='', fmt='%1.6f')
    
    names = list(bedload_data.columns)
    np.savetxt("bedload_data.csv", bedload_data, delimiter=",", header = ','.join(names) , comments='')
    
def estimate_slopes(sd) :
    # Import and format all parameters
    bedload_data = pd.read_csv('bedload_data.csv')
    grain_sizes = pd.read_csv('grain_sizes.csv')
    gsd_bins = pd.read_csv('gsd_bins.csv')
    ref_bedload = 0.002
        
    slopes = np.asarray(pd.read_csv('slopes.csv', header = None)).tolist()
    slopes = [i[0] for i in slopes]
    
    y_ints = np.asarray(pd.read_csv('y_ints.csv', header = None)).tolist()
    y_ints = [i[0] for i in y_ints]
  
    # Plot bedload and dimensionless shear stress with estimated slopes
    plt.clf()
    
    colors = ['blue', 'royalblue', 'forestgreen', 'slateblue', 'cyan', 'yellowgreen', 'steelblue', 'aquamarine', 'indigo', 'dodgerblue',
              'blue', 'royalblue', 'forestgreen', 'slateblue', 'cyan', 'yellowgreen', 'steelblue', 'aquamarine', 'indigo', 'dodgerblue']
    
    # Plot dimensionless bedload and shear stress   
    for i in np.arange(len(gsd_bins)-2):                
        globals()[f'x{i+1}'] = bedload_data[f"Dimensionless Tau ({str(gsd_bins['Bins'][i+1])})"]
        globals()[f'y{i+1}'] = bedload_data[f"Dimensionless bedload ({str(gsd_bins['Bins'][i+1])})"]
        globals()[f'a{i}'] = plt.scatter(*fn.process(globals()[f'x{i+1}'], globals()[f'y{i+1}']), s=10, color = colors[i])
    

    y_concat = []
    for i in np.arange(len(grain_sizes)):
        y_concat = np.append(y_concat, globals()[f'y{i+1}'])
 

    plt.yscale('log')
    plt.xscale('log')
    
    ymin = np.nanmin(y_concat[np.nonzero(y_concat)])
    ymax = np.nanmax(y_concat[y_concat != np.inf])
    plt.ylim(ymin-0.5*ymin, ymax+0.5*ymax)
    
    
    # Plot reference bedload rate = 0.002
    plt.axhline(y = ref_bedload, color = 'black')
    
    plt.title('Dimensionless Bedload Transport')
    plt.xlabel('Dimensionless shear stress (\u03C4i*)')
    plt.ylabel('Dimensionless bedload transport rate (Wi*)')
    
    handles = []
    labels = []
    for i in np.arange(len(grain_sizes)):
        handles = np.append(handles, globals()[f'a{i}'])
        labels = np.append(labels, f"{gsd_bins['Bins'][i+1]}")
    
    plt.legend(handles, labels, fontsize = 'small', loc=4, ncol=2)
    
    for i in np.arange(len(slopes))[::-1]:
        if slopes[i] < 3:
            slopes[i] = np.mean([slopes[i-1], slopes[i-2]])
            y_ints[i] = np.mean(np.log10(fn.process(globals()[f'x{i+1}'], 
                  globals()[f'y{i+1}'])[1])) -slopes[i]*np.mean(np.log10(fn.process(globals()[f'x{i+1}'], globals()[f'y{i+1}'])[0]))

    # Estimate slope using Bakke 2017 method
    bed_ints = np.empty([1,0])
    for i in np.arange(len(grain_sizes)):
        bed_ints = np.append(bed_ints, fn.slope_estimator(globals()[f'x{i+1}'], globals()[f'y{i+1}'], slopes[i], y_ints[i]))
        
    # Generate error messages for outliers  
    for i in np.arange(1, len(grain_sizes)):
        global error2
        mask = globals()[f'x{i}'].isin(np.setdiff1d(fn.process(globals()[f'x{i}'], 
                       globals()[f'y{i}'])[0], fn.remove_outliers(globals()[f'x{i}'], globals()[f'y{i}'], sd)[0]))
        plt.scatter(globals()[f'x{i}'][mask], globals()[f'y{i}'][mask], s=10, facecolors='white', edgecolors='r')        
        if len(globals()[f'x{i}']) > 0:
            error2 = f'Red points: outliers'
            file = open("error2","w")
            file.write(error2)
            file.close()
        else:
            error2 = f''
            file = open("error2","w")
            file.write(error2)
            file.close()
        
    plt.savefig('bedload', dpi=70)
    plt.clf()
    
    # Save new intercepts and slopes
    np.savetxt("bed_ints.csv", bed_ints, delimiter=",")
    np.savetxt("slopes.csv", slopes, delimiter=",")
    np.savetxt("y_ints.csv", y_ints, delimiter=",")

    
def filter_outliers(sd) :
    # Import and format data
    bedload_data = pd.read_csv('bedload_data.csv')
    grain_sizes = pd.read_csv('grain_sizes.csv')
    gsd_bins = pd.read_csv('gsd_bins.csv')
    ref_bedload = 0.002
    
            
    for i in np.arange(len(gsd_bins)-2):                
        globals()[f'x{i+1}'] = bedload_data[f"Dimensionless Tau ({str(gsd_bins['Bins'][i+1])})"]
        globals()[f'y{i+1}'] = bedload_data[f"Dimensionless bedload ({str(gsd_bins['Bins'][i+1])})"]
    
    colors = ['blue', 'royalblue', 'forestgreen', 'slateblue', 'cyan', 'yellowgreen', 'steelblue', 'aquamarine', 'indigo', 'dodgerblue',
              'blue', 'royalblue', 'forestgreen', 'slateblue', 'cyan', 'yellowgreen', 'steelblue', 'aquamarine', 'indigo', 'dodgerblue']
    
    # Check slope values for outliers and make array of y intercepts of best fit lines
    slopes = np.empty([1,0])
    y_ints = np.empty([1,0])
    
    for i in np.arange(len(grain_sizes)): 
        slopes = np.append(slopes, fn.filtered_orthreg(*fn.remove_outliers(globals()[f'x{i+1}'], globals()[f'y{i+1}'], sd))[1])
        y_ints = np.append(y_ints, fn.filtered_orthreg(*fn.remove_outliers(globals()[f'x{i+1}'], globals()[f'y{i+1}'], sd))[2])
    
    plt.clf() 
  
    plt.yscale('log')
    plt.xscale('log')
    
    y_concat = []
    for i in np.arange(len(grain_sizes)):
        y_concat = np.append(y_concat, globals()[f'y{i+1}'])
        
    ymin = np.nanmin(y_concat[np.nonzero(y_concat)])
    ymax = np.nanmax(y_concat[y_concat != np.inf])
    plt.ylim(ymin-0.5*ymin, ymax+0.5*ymax)
    
    # Plot line for reference bedload rate = 0.002
    plt.axhline(y = ref_bedload, color = 'black')
    
    plt.title('Dimensionless Bedload Transport')
    plt.xlabel('Dimensionless shear stress (\u03C4i*)')
    plt.ylabel('Dimensionless bedload transport rate (Wi*)')
    
    handles = []
    labels = []
    for i in np.arange(len(gsd_bins)-2):
        handles = np.append(handles, globals()[f'a{i}'])
        labels = np.append(labels, f"{gsd_bins['Bins'][i+1]}")
    
    plt.legend(handles, labels, fontsize = 'small', loc=4, ncol=2)
    
    for i in np.arange(len(slopes))[::-1]:
        if slopes[i] < 3:
            slopes[i] = np.mean([slopes[i-1], slopes[i-2]])
            y_ints[i] = np.mean(np.log10(fn.process(globals()[f'x{i+1}'], 
                  globals()[f'y{i+1}'])[1])) - slopes[i] * np.mean(np.log10(fn.process(globals()[f'x{i+1}'], globals()[f'y{i+1}'])[0]))
    
    bed_ints = np.empty([1,0])
    
    # Remove outliers from bedload data
    for i in np.arange(len(grain_sizes)):
        if len(fn.remove_outliers(globals()[f'x{i+1}'], globals()[f'y{i+1}'], sd)[0]) != 0 or len(fn.remove_outliers(globals()[f'x{i+1}'], globals()[f'y{i+1}'], sd)[1]) != 0:
            bed_ints = np.append(bed_ints, fn.slope_estimator(pd.Series(fn.remove_outliers(globals()[f'x{i+1}'], globals()[f'y{i+1}'], sd)[0]), 
                                                              pd.Series(fn.remove_outliers(globals()[f'x{i+1}'], globals()[f'y{i+1}'], sd)[1]), 
                                                              slopes[i], y_ints[i]))
            globals()[f'a{i}'] = plt.scatter(*fn.remove_outliers(globals()[f'x{i+1}'], globals()[f'y{i+1}'], sd), s=10, color = colors[i])
        else: bed_ints = np.append(bed_ints, 0)


    plt.savefig('bedload', dpi=70)
    plt.clf()
    
    np.savetxt("bed_ints.csv", bed_ints, delimiter=",")
    
def hiding_function() :
    # Import and format data
    intercepts = np.asarray(pd.read_csv('bed_ints.csv', header = None)).tolist()
    intercepts = [i[0] for i in intercepts]
    
    grain_sizes = pd.read_csv('grain_sizes.csv')
    gsd_bins = pd.read_csv('gsd_bins.csv')
    gsd_prop = pd.read_csv('gsd_prop.csv')
    
            
    perc_finer = []
    for i in np.arange(len(gsd_bins)-2):
        perc_finer = np.append(perc_finer, gsd_prop[i+1:len(gsd_prop)].sum())
    
    D50 = np.interp(0.5, np.sort(perc_finer), np.array(gsd_bins['Bins'][0:len(gsd_bins)-2])[np.argsort(perc_finer)])
    
    # Plot hiding factor relation
    grain_data = pd.DataFrame(data = {'Grain size': grain_sizes['GS'], 'Grain size ratio': grain_sizes['GS'] / D50, 'Ref Tau': intercepts})
    
    ratio = grain_data['Grain size ratio']
    ref_tau = grain_data['Ref Tau']
    
    plt.clf()
    plt.yscale('log')
    plt.xscale('log')
    
    mask = np.ma.masked_where(ref_tau > 0, ref_tau)
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(ratio[mask.mask]), np.log10(ref_tau[mask.mask]))
    new_tau = 10**(np.log10(ratio[mask.mask]) * slope + intercept)
    plt.plot(ratio[mask.mask], new_tau, 'k')
    plt.scatter(ratio, ref_tau, color='grey')
    
    
    plt.title('Hiding Factor Relation')
    plt.xlabel('Grain size ratio (Di / D50)')
    plt.ylabel('Reference shear stress (\u03C4*ri)')
    
    plt.savefig('hiding_function', dpi=70)
    plt.clf()
    
    # Get hiding factor function parameters
    HF_a = round(10**intercept, 3) 
    HF_b = round(slope, 3)
    
    np.savetxt("vars.csv", [HF_a, HF_b], delimiter=",")
    
       
def optimize_beta(): 
    # Import and format data
    bedload_data = pd.read_csv('bedload_data.csv')
    stage_depth = pd.read_csv('stage_depth.csv')

    grain_sizes = pd.read_csv('grain_sizes.csv')
    gsd_bins = pd.read_csv('gsd_bins.csv')
    gsd_prop = pd.read_csv('gsd_prop.csv')
    
    perc_finer = []
    for i in np.arange(len(gsd_bins)-2):
        perc_finer = np.append(perc_finer, gsd_prop[i+1:len(gsd_prop)].sum())
    
    D50 = np.interp(0.5, np.sort(perc_finer), np.array(gsd_bins['Bins'][0:len(gsd_bins)-2])[np.argsort(perc_finer)])

    # Import hiding function parameters
    vars = np.asarray(pd.read_csv('vars.csv', header = None)).tolist()
    vars = [i[0] for i in vars]
    
    HF_a = vars[0]
    HF_b = vars[1]
    
    ref_bedload = 0.002
    
    # Get fractional content of bed as a table
    F_i = pd.DataFrame()
    
    for i in np.arange(len(gsd_bins)-2):
        F_i[f"{gsd_bins['Bins'][i+1]}"] = np.repeat(gsd_prop['Prop'][i+1], len(bedload_data))
        
    # Optimize beta
    beta = 5600
    meas_sum = bedload_data['Bedload'].sum()
    phi_table = pd.DataFrame()
    
    # First, get phi = tao_i / tao_ri for every grain size and every discharge
    for i in np.arange(len(grain_sizes)):
        tau_i = stage_depth['Shear stress'] / (1000 * 9.81 * (2.65-1) * (grain_sizes['GS'][i]/1000))
        tau_ri = HF_a *(grain_sizes['GS'][i] / D50)**HF_b
        phi_table[f"{str(gsd_bins['Bins'][i+1])}"] = tau_i / tau_ri
    
    # Make table of dimensionless bedload values
    Wi_table = ref_bedload * beta * (1 - (0.853 / phi_table))**4.5        # Bakke Eq. 7

    # Solve for q_bi for each grain size and discharge (Bakke Eq. 4)
    q_bi = Wi_table.mul(F_i.values).mul(stage_depth['Shear stress']**(3/2) / ((2.65 - 1) * 9.81 * 1000**(3/2)), axis=0)
        
    ratio = meas_sum / q_bi.sum().sum()
    optimized_beta = beta * ratio
    
    print("Optimized Beta =", optimized_beta)
    np.savetxt("opt_beta.csv", [optimized_beta], delimiter=",")     # Save optimized beta


def run_model(all_data, units1, units2, a=0, b=0, c=0):
    # Import and format data
    bedload_data = pd.read_csv('bedload_data.csv') 
    optimized_beta= pd.read_csv('opt_beta.csv', header=None) 
    stage_depth = pd.read_csv('stage_depth.csv')
    grain_sizes = pd.read_csv('grain_sizes.csv')
    gsd_bins = pd.read_csv('gsd_bins.csv')
    gsd_prop = pd.read_csv('gsd_prop.csv')
    stage_disc = pd.read_csv('stage_disc.csv')
    xs = pd.read_csv('xs.csv')
    flow_freq = pd.read_excel(all_data, 'flow_freq', skiprows = [1])

    if units1 == 1:
            flow_freq['Discharge'] = flow_freq['Discharge'] * 0.0283
            flow_freq['Width'] = flow_freq['Width'] * 0.3048
            
    
    ref_bedload = 0.002
    
            
    perc_finer = []
    for i in np.arange(len(gsd_bins)-2):
        perc_finer = np.append(perc_finer, gsd_prop[i+1:len(gsd_prop)].sum())
    
    D50 = np.interp(0.5, np.sort(perc_finer), np.array(gsd_bins['Bins'][0:len(gsd_bins)-2])[np.argsort(perc_finer)])
    
    # Import hiding function parameters
    vars = np.asarray(pd.read_csv('vars.csv', header = None)).tolist()
    vars = [i[0] for i in vars]
    
    HF_a = vars[0]
    HF_b = vars[1]
    
    # Use optimized beta and hiding function to model bedload for all measured flows
    F_i = pd.DataFrame()
    
    for i in np.arange(len(gsd_bins)-2):
        F_i[f"{gsd_bins['Bins'][i+1]}"] =  np.repeat(gsd_prop['Prop'][i+1], len(bedload_data))
    
    flow_freq = flow_freq.sort_values(by='Discharge')
    
    meas_model_q = pd.DataFrame()
    phi_table = pd.DataFrame()

    for i in np.arange(len(grain_sizes)):
        tau_i = stage_depth['Shear stress'] / (1000 * 9.81 * (2.65-1) * (grain_sizes['GS'][i]/1000))
        tau_ri = HF_a *(grain_sizes['GS'][i] / D50)**HF_b
        phi_table[f"{str(gsd_bins['Bins'][i+1])}"] = tau_i / tau_ri
        
    model_dimless_bedload = ref_bedload * optimized_beta.iloc[0,0] * (1 - (0.853/phi_table))**4.5
    model_bedload = model_dimless_bedload.mul(F_i.values).mul(stage_depth['Shear stress']**(3/2),axis=0) / ((2.65 - 1) * 9.81 * 1000**(3/2))
    meas_model_q = model_bedload
    
    
    meas_model_q['Total'] = meas_model_q.sum(axis=1) 
    meas_model_q['Shear stress'] = stage_depth['Shear stress']
    meas_model_q['Discharge'] = bedload_data['Discharge']
    

    # Use optimized beta and hiding function to model bedload for all flows in flow_freq    
    stage = fn.stageQ_RC(flow_freq, stage_disc, a,b,c)
    stage_depth = fn.depth_dist(stage, xs)
    stage_depth['Shear stress'] = 1000 * 9.81 * flow_freq['Slope'] * stage_depth['Avg depth']
    
    F_i = pd.DataFrame()
    
    for i in np.arange(len(gsd_bins)-2):
        F_i[f"{gsd_bins['Bins'][i+1]}"] =  np.repeat(gsd_prop['Prop'][i+1], len(flow_freq))
        
    total_model_load = pd.DataFrame()
    phi_table = pd.DataFrame()
    
    for i in np.arange(len(grain_sizes)):
        tau_i = stage_depth['Shear stress'] / (1000 * 9.81 * (2.65-1) * (grain_sizes['GS'][i]/1000))
        tau_ri = HF_a *(grain_sizes['GS'][i] / D50)**HF_b
        phi_table[f"{str(gsd_bins['Bins'][i+1])}"] = tau_i / tau_ri
        
    model_dimless_bedload = ref_bedload * optimized_beta.iloc[0,0] * (1 - (0.853/phi_table))**4.5
    model_bedload = model_dimless_bedload.mul(F_i.values).mul(stage_depth['Shear stress']**(3/2),axis=0) / ((2.65 - 1) * 9.81 * 1000**(3/2))
    total_model_load = model_bedload

  
    total_model_load['Total'] = total_model_load.sum(axis=1) 
    total_model_load['Shear stress'] = stage_depth['Shear stress']
    total_model_load['Discharge'] = flow_freq['Discharge']

    # Calculate effective discharge
    if ~np.isnan(flow_freq['Prop']).all() :
        bin_freq = flow_freq.iloc[0, 1]
        for i in np.arange(len(flow_freq) -1):
            bin_freq = np.append(bin_freq, flow_freq.iloc[i+1, 1] - flow_freq.iloc[i, 1])
        
        effQ = []
        total_load = [0]
        total_load = np.append(total_load, total_model_load['Total'])
        for i in np.arange(len(flow_freq)):
            effQ = np.append(effQ, bin_freq[i] * (total_load[i] + total_load[i+1]) /2)
        max_eff_Q = flow_freq.iloc[list(effQ).index(max(effQ)),0]
        
        effQ_table = pd.DataFrame({'Discharge': total_model_load['Discharge'], 'Bedload*Freq': effQ})
        
        
        # Plot effective discharge, flow frequency, and model bedload
        # First plot in terms of bedload
        plt.clf()
        fig, ax = plt.subplots()
        fig.subplots_adjust(right=0.75)
        
        twin = ax.twinx()
        
        ax.plot(flow_freq['Discharge'], bin_freq, label="Freq")            
        twin.plot(flow_freq['Discharge'], total_model_load['Total'], "r-", label="Bedload flux")
        twin.plot(flow_freq['Discharge'], effQ, "black", label="Bedload flux * freq")
        
        ax.set_xlabel("Discharge (m3/s)")
        ax.set_ylabel('Flow frequency')
        twin.set_ylabel('Bedload flux (m2/s) OR\nBedload flux * frequency (m2/s)')
        twin.set_yscale('log')
        
        ax.legend(loc='upper left')
        twin.legend(loc='upper right')
        plt.title('Effective Discharge')
        plt.savefig('effective_Q', dpi=70)
        plt.clf()
        
        # Second plot in terms of shear stress
        fig, ax = plt.subplots()
        fig.subplots_adjust(right=0.75)
        
        twin = ax.twinx()
        
        ax.plot(stage_depth['Shear stress'], bin_freq, label="Freq")            
        twin.plot(stage_depth['Shear stress'], total_model_load['Total'], "r-", label="Bedload flux")
        twin.plot(stage_depth['Shear stress'], effQ, "black", label="Bedload flux * freq")
        
        ax.set_xlabel("Shear stress (Pa)")
        ax.set_ylabel('Flow frequency')
        twin.set_ylabel('Bedload flux (m2/s) OR\nBedload flux * frequency (m2/s)')
        twin.set_yscale('log')
        
        ax.legend(loc='upper left')
        twin.legend(loc='upper right')
        plt.title('Effective Discharge')
        plt.savefig('effective_Q_SS', dpi=70)
        plt.clf()
    
    # Plot measured bedload vs model bedload by discharge 
    plt.yscale('log')
    plt.xlabel('Discharge (m3/s)')
    plt.ylabel('Bedload flux (m2/s)')
    meas = plt.scatter(bedload_data['Discharge'], bedload_data['Bedload'])
    
    indices = np.argsort(stage_depth['Shear stress']).to_numpy()
    mod = plt.plot(flow_freq['Discharge'], total_model_load['Total'].to_numpy()[indices], color='red')
    
    handles = [meas, mod[0]]
    labels = ['Measured', 'Model']

    plt.legend(handles, labels)
    plt.title('Results: Model vs. Measured Bedload flux')
    plt.savefig('bedload_results_Q', dpi=70)
    
    plt.clf()
    
    # Plot measured bedload vs model bedload by shear stress
    plt.yscale('log')
    plt.xlabel('Shear stress (Pa)')
    plt.ylabel('Bedload flux (m2/s)')
    meas = plt.scatter(bedload_data['Shear stress'], bedload_data['Bedload'])
    
    indices = np.argsort(stage_depth['Shear stress']).to_numpy()
    mod = plt.plot(stage_depth['Shear stress'].to_numpy()[indices], total_model_load['Total'].to_numpy()[indices], color='red')
    
    handles = [meas, mod[0]]
    labels = ['Measured', 'Model']

    plt.legend(handles, labels)
    plt.title('Results: Model vs. Measured Bedload flux')
    plt.savefig('bedload_results_SS', dpi=70)
    plt.clf()
    
    # Plot bedload results by grain size and a 1-1 line
    colors = ['blue', 'royalblue', 'forestgreen', 'slateblue', 'cyan', 'yellowgreen', 'steelblue', 'aquamarine', 'indigo', 'dodgerblue',
              'blue', 'royalblue', 'forestgreen', 'slateblue', 'cyan', 'yellowgreen', 'steelblue', 'aquamarine', 'indigo', 'dodgerblue']
    plt.yscale('log')
    plt.xscale('log')
        
    x_vals = []
    y_vals = []
    for i in np.arange(len(gsd_bins)-2):
        globals()[f'a{i}'] = plt.scatter(bedload_data[f"{str(gsd_bins['Bins'][i+1])}"] * bedload_data['Bedload'], meas_model_q[f"{str(gsd_bins['Bins'][i+1])}"], 
                                         color=colors[i], s=10)
        x_vals = np.append(x_vals, bedload_data[f"{str(gsd_bins['Bins'][i+1])}"] * bedload_data['Bedload'])
        y_vals = np.append(y_vals, meas_model_q[f"{str(gsd_bins['Bins'][i+1])}"])
    
    one_to_one = plt.plot([x_vals[np.nonzero(x_vals)].min(), x_vals.max()], [x_vals[np.nonzero(x_vals)].min(), x_vals.max()], color='k',linewidth=2)
    plt.xlabel('Measured Bedload flux (m2/s)')
    plt.ylabel('Model Bedload flux (m2/s)')
    plt.title('Results: Bedload flux by Grain Size')
    
    handles = [one_to_one]
    labels = ['1-to-1']
    for i in np.arange(len(gsd_bins)-2):
        handles = np.append(handles, globals()[f'a{i}'])
        labels = np.append(labels, f"{str(gsd_bins['Bins'][i+1])}")
    
    plt.legend(handles, labels, fontsize='small')
    plt.xlim(np.nanmin(x_vals[np.nonzero(x_vals)]), 2*np.nanmax(x_vals))
    plt.ylim(np.nanmin(y_vals[np.nonzero(y_vals)]), 2*np.nanmax(y_vals))
    plt.savefig('bedload_results_gs', dpi=70)
    plt.clf()

    # Plot total bedload results and a 1-1 line    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Measured Bedload flux (m2/s)')
    plt.ylabel('Model Bedload flux (m2/s)')    
    plt.title('Results: Total Bedload flux')
    
    bedloadplot = plt.scatter(bedload_data['Bedload'], meas_model_q['Total'], s=10)
    one_to_one, = plt.plot([bedload_data['Bedload'].min(), bedload_data['Bedload'].max()], [bedload_data['Bedload'].min(), 
                           bedload_data['Bedload'].max()], color='k',linewidth=2)
    
    handles = [bedloadplot, one_to_one]
    labels = ['Total Bedload', '1-to-1']
    
    plt.legend(handles, labels, fontsize='small')
    plt.savefig('bedload_results_total', dpi=70)
    plt.clf()

    # Save results to excel file
    if units2 == 1:
        total_model_load['Discharge'] = total_model_load['Discharge'] / 0.0283
        total_model_load.iloc[:, 0:len(grain_sizes)+1] = total_model_load.iloc[:, 0:len(grain_sizes)+1].mul(flow_freq['Width'], axis=0) * 2650 * 0.3048 * 86400 / 907.185
                
        results_cols = [float(x) for x in list(total_model_load.columns)[0:len(grain_sizes)]]
        new_cols = [round(x / 25.4, 3) for x in results_cols]
        total_model_load.columns = np.append(new_cols, list(total_model_load.columns)[len(grain_sizes):])
        
        units_row = np.append(np.repeat('tons/day', len(grain_sizes)+1), ['Pa', 'cfs'])
        total_model_load.loc[-1] = units_row
        total_model_load.index = total_model_load.index + 1  # shifting index
        total_model_load = total_model_load.sort_index()  # sorting by index
        
        units_row = ['cfs', 'tons/day']
        effQ_table.loc[-1] = units_row
        effQ_table.index = effQ_table.index + 1  # shifting index
        effQ_table = effQ_table.sort_index()  # sorting by index
        
        if ~np.isnan(flow_freq['Prop']).all() :
            max_eff_Q = max_eff_Q / 0.0283
        
    if units2 == 2:
        total_model_load.iloc[:, 0:len(grain_sizes)+1] = (total_model_load.iloc[:, 0:len(grain_sizes)+1] * 2650).multiply(flow_freq['Width'], axis=0)
        
        units_row = np.append(np.repeat('kg/s', len(grain_sizes)+1), ['Pa', 'm3/s'])
        total_model_load.loc[-1] = units_row
        total_model_load.index = total_model_load.index + 1  # shifting index
        total_model_load = total_model_load.sort_index()  # sorting by index
        
        units_row = ['m3/s', 'kg/s']
        effQ_table.loc[-1] = units_row
        effQ_table.index = effQ_table.index + 1  # shifting index
        effQ_table = effQ_table.sort_index()  # sorting by index
        
    writer = pd.ExcelWriter('RESULTS.xlsx', engine='xlsxwriter')

    total_model_load.to_excel(writer, sheet_name='bedload', index = False, header=True)
    
    if ~np.isnan(flow_freq['Prop']).all() :
        effQ_table.to_excel(writer, sheet_name='effective_Q', index = False, header=True)
    
    writer.save()
