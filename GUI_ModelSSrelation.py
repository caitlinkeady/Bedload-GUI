# -*- coding: utf-8 -*-
"""
BEDLOAD MODEL GUI
author: Caitlin Keady
"""

from GUI_plots_ModelSSrelation import run_bedload
from GUI_plots_ModelSSrelation import hiding_function
from GUI_plots_ModelSSrelation import optimize_beta
from GUI_plots_ModelSSrelation import estimate_slopes
from GUI_plots_ModelSSrelation import filter_outliers
from GUI_plots_ModelSSrelation import run_model
import tkinter as tk
from tkinter.filedialog import askopenfilename
import pandas as pd
import numpy as np

window=tk.Tk()
window.configure(bg = 'white')
window.title('Sediment Transport GUI')
window.iconify()

#icon = tk.PhotoImage(file = "rock_icon.ico")
#window.iconphoto(False, icon)

datafile = tk.StringVar()
stdev = tk.DoubleVar()

a = tk.DoubleVar()
b = tk.DoubleVar()
c = tk.DoubleVar()

units1 = tk.IntVar()
units2 = tk.IntVar()

def new_win(): # new window with instructions
    newwin = tk.Toplevel(window)
    newwin.configure(bg = 'white')
    display = tk.Label(newwin, text=
                    '''Step 1: Upload an Excel file containing all data in the format described below. The Excel file should have 5 tabs: 'bedload', 
            'grain_sizes', 'flow_freq', 'xs', and 'stage_discharge'. The bedload data tab must contain discharge in column 1, total bedload 
            in column 2, slope in column 3, channel width in column 4, followed by fractional content of the bedload by grain size (largest to 
            smallest grain size). Headers for the first four columns are 'Discharge', 'Bedload', 'Slope', and 'Width', respectively. Fractional 
            content of the bedload columns should be titled '{grain size}' in whichever unit you choose. For example '64.0' or '0.25'. You can 
            choose to use SI units or English units. Bedload units should either be English tons/day or kg/m/s, discharge units should be cfs or 
            m3/s, and width units should be ft or m. 
            
            The grain size tab must have 1 column containing grain size bins from largest to smallest (mm or inches) and a second column 
            containing fractional content of the bed by grain size (largest to smallest grain size). Headers are 'Bins' and 'Prop'. 
            
            The flow frequency tab must contain discharge (m3/s or cfs) in column 1 and the proportion of occurence in column 2. This file should 
            have headers of 'Discharge' and 'Prop', respectively. You can choose to leave the 'Prop' column blank if you do not want to calculate 
            effective discharge. 
            
            The cross section tab must have cross stream location (m or ft) in column 1, elevation (m or ft) in column 2, a third column containing 
            0s or 1s indicating where the active channel is, and a fourth column containing 0s or 1s indicating the location of the datum. For 
            example, 1s in the 3rd column should be placed at every row where bedload can move and a single 1 should be placed in the fourth column 
            to indicate which elevation is the datum for the stage measurements. The headers in this sheet must be 'Distance', 'Elevation', 'Channel', 
            and 'Datum' respectively.
            
            The stage-discharge tab must have discharge (m3/s or cfs) in column 1 and stage (m or ft) in column 2. Headers should be 'Discharge' 
            and 'Stage', respectively. If desired, you can opt to provide a rating curve to describe the stage discharge relation instead of uploading 
            these data. If you choose to input a rating curve, please leave the stage-discharge tab blank. 
            
Step 2: Inspect the dimensionless bedload plot and edit slopes to best fit your data. Errors will be highlighted in red and explained. Remove outliers 
            as desired by choosing how many standard deviations (3 is recommended) of bedload data to include for each grain size. Points outside of 
            your chosen standard deviation will be highlighted in red. You can choose to keep or omit these data. 

Step 3: Once you are satisfied with your plot, click "Calibrate model". The model will calibrate the bedload equation to best fit your measured data. 
            
Step 4: Now you can view your results and export your modeled bedload data and effective discharge. You will find a file named "RESULTS.csv" in the 
            folder containing your input files. You may also go back and further edit the dimensionless bedload plot if you are not satisfied with 
            your results.''', font=('calibri', 12), justify='left')
    
    display.configure(bg = 'white')
    display.pack()

def import_csv_data():  # import CSV file with bedload data
    global datafile
    csv_file_path = askopenfilename()
    datafile.set(csv_file_path)

     
def load_img1():   # load regression image and add new buttons
    global image1
    run_bedload(pd.ExcelFile(datafile.get()), units1.get(), a=a.get(), b=b.get(), c=c.get())
    for label in window.grid_slaves():
        if int(label.grid_info()["row"]) >= 9:
            label.grid_forget()
    image1 = tk.PhotoImage(file = 'bedload.png')
    tk.Label(image = image1).grid(row = 10, column = 0, columnspan=3)
    tk.Label(window, text=open('error0', 'r').read(), font=('calibri', 12), bg='white', fg='red').grid(row = 12, column = 0, columnspan = 3)
    tk.Label(window, text=open('error1', 'r').read(), font=('calibri', 12), bg='white', fg='red').grid(row = 13, column = 0, columnspan = 3)
    tk.Button(window, text='Estimate best fit line(s) and highlight outliers',command=SD, font=('calibri', 10), bg='#87CEEB', 
              relief='groove').grid(row=11, column=1) 

def SD(): # standard deviation frame
    f0 = tk.Frame(window)
    f0.config(bg="lightgray")
    tk.Label(f0, text = 'Keep all points within x standard\ndeviations of the mean for each\ngrain size. 3 is recommended.', 
             font = ('calibri', 10), bg="lightgray").pack(side = 'bottom')
    tk.Label(f0, text = 'Standard deviation:', font = ('calibri', 12), bg="lightgray").pack(side = 'left', expand ='true')
    tk.Entry(f0, textvariable = stdev, relief = 'groove', bd = 2, width=3).pack(side = 'left', expand ='true')
    tk.Button(f0, text='Go',command=new_img1, font = ('calibri', 12), bg = '#87CEEB', relief = 'groove').pack(side = 'left', expand ='true')

    f0.grid(row=10, column=3, ipadx=25, columnspan=6, rowspan=3)
    
    
def new_img1(): # load new plot with estimated slope(s)
    global image1
    estimate_slopes(stdev.get())
    image1 = tk.PhotoImage(file = 'bedload.png')
    tk.Label(image = image1).grid(row = 10, column = 0, columnspan=3)
    for label in window.grid_slaves():
        if int(label.grid_info()["row"]) > 10:
            label.grid_forget()
    tk.Label(window, text=open('error2', 'r').read(), font=('calibri', 10), bg='white', fg='red').grid(row = 12, column = 0, columnspan = 3)
    f1 = tk.Frame(window)
    f1.config(bg="white")
    tk.Button(f1, text='Remove outliers',command=final_img1, font = ('calibri', 12), bg = '#87CEEB', relief = 'groove').pack(side="left")
    tk.Button(f1, text='Keep and continue',command=keep_outliers, font = ('calibri', 12), bg = '#87CEEB', relief = 'groove').pack(side="right")
    f1.grid(row=11, column=1, ipadx=10)

    
def final_img1():  # load plot with estimated slope(s) and removed outlier(s)
    filter_outliers(stdev.get())
    for label in window.grid_slaves():
        if int(label.grid_info()["row"]) >= 10:
            label.grid_forget()
    image1 = tk.PhotoImage(file = 'bedload.png')
    tk.Label(image = image1).grid(row = 10, column = 0, columnspan=3)
    tk.Button(window, text='Calibrate model',command=load_img2, font = ('calibri', 12), bg = '#87CEEB', relief = 'groove').grid(row=11, column = 1)

def keep_outliers():  # load plot with estimated slope(s)
    global image1
    for label in window.grid_slaves():
        if int(label.grid_info()["row"]) >= 10:
            label.grid_forget()
    image1 = tk.PhotoImage(file = 'bedload.png')
    tk.Label(image = image1).grid(row = 10, column = 0, columnspan=3)
    tk.Button(window, text='Calibrate model',command=load_img2, font = ('calibri', 12), bg = '#87CEEB', relief = 'groove').grid(row=11, column = 1)

def load_img2():   # load hiding function image and add new buttons
    global image2
    hiding_function()
    optimize_beta()
    for label in window.grid_slaves():
        if int(label.grid_info()["row"]) >= 10:
            label.grid_forget()
    image2 = tk.PhotoImage(file = 'hiding_function.png')
    tk.Label(image = image2).grid(row = 10, column = 0, columnspan=3)
    tk.Label(window, text = 'Done!', font = ('calibri', 14), bg = 'white').grid(row = 10, column = 3)
    tk.Button(window, text='Back',command=load_img1, font = ('calibri', 12), bg = '#87CEEB', relief = 'groove').grid(row=11, column = 0)
    tk.Button(window, text='Export & view results',command=calc_effQ, font = ('calibri', 12), bg = '#87CEEB', relief = 'groove').grid(row=11, column = 1)


def calc_effQ():
    global image3
    run_model(pd.ExcelFile(datafile.get()), units1.get(), units2.get(), a=a.get(), b=b.get(), c=c.get())
    for label in window.grid_slaves():
        if int(label.grid_info()["row"]) >= 10:
            label.grid_forget()
    image3 = tk.PhotoImage(file = 'bedload_results_SS.png')
    tk.Label(image = image3).grid(row = 10, column = 0, columnspan=3)
    
    image4 = tk.PhotoImage(file = 'bedload_results_Q.png')
    tk.Label(image = image4).grid(row = 10, column = 3, columnspan=3)
    
    image5 = tk.PhotoImage(file = 'bedload_results_gs.png')
    tk.Label(image = image5).grid(row = 11, column = 0, columnspan=3)
    
    image6 = tk.PhotoImage(file = 'bedload_results_total.png')
    tk.Label(image = image6).grid(row = 11, column = 3, columnspan=3)
    
    all_data = pd.ExcelFile(datafile.get())
    flow_freq = pd.read_excel(all_data, 'flow_freq')
    
    if ~np.isnan(flow_freq['Prop']).all() :
        image7 = tk.PhotoImage(file = 'effective_Q.png')
        tk.Label(image = image7).grid(row = 11, column = 12, columnspan=3)
        
        image8 = tk.PhotoImage(file = 'effective_Q_SS.png')
        tk.Label(image = image8).grid(row = 10, column = 12, columnspan=3)
    
    tk.Button(window, text='Back',command=load_img1, font = ('calibri', 12), bg = '#87CEEB', relief = 'groove').grid(row=12, column = 3)

# delete lower rows to reset    
def reset():
    for label in window.grid_slaves():
        if int(label.grid_info()["row"]) > 9:
            label.grid_forget()    

tk.Button(window, text = 'Instructions', command =new_win, font = ('calibri', 12), bg = 'seagreen', fg = 'white', 
          relief = 'groove').grid(row = 0, column = 3)
tk.Label(window, text = 'Enter values', font = ('calibri', 12), bg = 'white').grid(row = 1, column = 1)
tk.Label(window, text = 'Input file', font = ('calibri', 12), bg = 'white', relief = 'flat').grid(row = 2, column = 0)

tk.Entry(window, textvariable = datafile, width = 25, relief = 'groove', bd = 2).grid(row = 2, column = 1)

tk.Button(window, text='Browse files',command=import_csv_data, font = ('calibri', 12), bg = '#87CEEB', relief = 'groove').grid(row=2, column=2)
          
tk.Label(window, text = 'Input file units', font = ('calibri', 12), bg = 'white', relief = 'flat').grid(row = 4, column = 0)

f2 = tk.Frame(window)
f2.config(bg="white")
tk.Radiobutton(f2, text="English", variable = units1, font = ('calibri', 12), bg = 'white', value = 1).pack(side='right', anchor='w')
tk.Radiobutton(f2, text="SI", variable = units1, font = ('calibri', 12), bg = 'white', value = 2).pack(side='right', anchor='w')
f2.grid(row=4, column=1)

tk.Label(window, text = 'Desired output units', font = ('calibri', 12), bg = 'white', relief = 'flat').grid(row = 5, column = 0)

f3 = tk.Frame(window)
f3.config(bg="white")
tk.Radiobutton(f3, text="English", variable = units2, font = ('calibri', 12), bg = 'white', value = 1).pack(side='right', anchor='w')
tk.Radiobutton(f3, text="SI", variable = units2, font = ('calibri', 12), bg = 'white', value = 2).pack(side='right', anchor='w')
f3.grid(row=5, column=1)

tk.Button(window, text='Begin',command=load_img1, font = ('calibri', 12), bg = '#87CEEB', relief = 'groove').grid(row=6, column = 1)    
tk.Button(window, text='Clear All', command = reset,
          font = ('calibri', 12), bg = 'darkred', fg = 'white', relief = 'groove').grid(row=0, column=2)          

# Stage discharge rating curve parameters
f1 = tk.Frame(window)
f1.config(bg="white")
tk.Label(f1, text = 'Optional: stage-discharge rating curve: y=', font = ('calibri', 12), bg="white").pack(side = 'left')
tk.Entry(f1, textvariable = a, relief = 'groove', bd = 2, width=6).pack(side = 'left')
tk.Label(f1, text = '(x -', font = ('calibri', 10), bg="white").pack(side = 'left')
tk.Entry(f1, textvariable = b, relief = 'groove', bd = 2, width=6).pack(side = 'left')
tk.Label(f1, text = ')^', font = ('calibri', 10), bg="white").pack(side = 'left')
tk.Entry(f1, textvariable = c, relief = 'groove', bd = 2, width=6).pack(side = 'left')

f1.grid(row=3, column=0, ipadx=25, columnspan=4)
    
window.mainloop()
