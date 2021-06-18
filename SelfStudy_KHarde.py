"""
### University of St. Gallen 
### Data Analytics II: Self Study Project 
### Kevin Hardegger (12-758-785)
### kevin.hardegger@student.unisg.ch

"""

# Import Modules
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import seed
from numpy.random import multivariate_normal
import statsmodels.api as sm
import scipy.stats

# Set Working Directory
PATH = "/Users/Dada/Desktop/Programming/Data 2/Self Studies/HandIn"
sys.path.append(PATH)

# Load Own Functions
import SSFunctions_KHarde as pc

#**** ***** ***** ***** ***** ***** ***** ***** ***** ***** *****

# Defining Parameters for Data Genereating Process:
# -------------------- -------------------- -------------------- 

Mean = [5, -5]                  # Mean Of Two Covariates X1 and X2
Cov = [[2, 0.5], [0.5, 3]]      # Variance-Covariance Matrix 
betas = np.array([2,1,5,8])     # Betas of: Treatment[ATE_TRUE], Intercept, X1, X2
n = 1000                        # Population

# Setting Seed
seed(200)

##### ##### ##### Sample Monte Carlo Simulation ##### ##### #####
#**** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** 

"""
# Defining Sample Monte Carlo Specification:
# -------------------- -------------------- 
DGP = 2                         # Data Generating Process (1=DGP1, 2=DGP2, 3=DGP3)                    
Estimator = "B"                 # Estimation Method (A=IPW, B=Doubly Robust)
num_simulations = 100           # Number of Monte Carlo Simulations
BS_Reps = 100                   # Number of Bootstrap Repitions
SampleSize = 100                # Size of Sample


# Perform Sample Simulation
ATE_Est, ATE_Comp, SE, ATE_True = pc.Simulation(Mean, Cov, betas, 
                                                DGP, Estimator, SampleSize, 
                                                BS_Reps, n, num_simulations)

Bias, Variance, MSE = pc.Performance(ATE, SE, ATE_True)
print(ATE_Est, SE, Bias, Variance, MSE) # works for every DGP & Estimator Combination --> Good 2 GO
print("Sample Monte Carlo Simulation Finished.")
"""

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
#**** ***** *****  Main Monte Carlo Simulations ***** ***** ***** 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 

# Defining Main Monte Carlo Specifications:
# -------------------- -------------------- 
Sequence = list(range(100,1050,50))  # Monte Carlo Sequence for Sample Sizes
num_simulations = 100                # Number of Monte Carlo Simulations
BS_Reps = 100                        # Number of Bootstrap Repitions

# DGP 1 & Estimator A:
# -------------------- 
DGP = 1
Estimator = "A"
DGPEST1A = "DGP1_Estimator1A"

# Predefine Lists for Saving Results:
ATE1A=[]                         
ATE_Comp1A = []             
SE1A =[]
Bias1A=[]
Variance1A=[]
MSE1A=[]

# Perform Sample Simulation 
for i in Sequence:
    SampleSize = i
    # Monte Carlo Simulation
    ATE, ATE_Comp, SE, ATE_True = pc.Simulation(Mean, Cov, betas, DGP, 
                                              Estimator, SampleSize, BS_Reps, 
                                              n, num_simulations)
    # Performance
    Bias, Variance, MSE = pc.Performance(ATE, SE, ATE_True)
    
    # Save Results
    ATE1A= np.append(ATE1A, ATE)
    ATE_Comp1A = np.append(ATE_Comp1A, ATE_Comp)
    SE1A = np.append(SE1A, SE)
    Bias1A = np.append(Bias1A, Bias)
    Variance1A = np.append(Variance1A, Variance)
    MSE1A= np.append(MSE1A, MSE)

# Save Performance in Dicitonary and Transform to DataFrame   
Perf1A = {"Bias": Bias1A, "Variance": Variance1A, "MSE": MSE1A}
Perf1A = pd.DataFrame(Perf1A, index = Sequence)


# Inform If Finished 
print("\nMonte Carlo Simulation 1A Finished:")
print("-------------------------------------")
# Print Mean ATE and Mean SE
print("MEAN ATE1A: " + str(round(np.mean(ATE1A),3)))
print("MEAN SE1A: " + str(round(np.mean(SE1A),3)))
print("MEAN Bias1A: " + str(round(np.mean(Bias1A),3)))
print("MEAN MSE1A: " + str(round(np.mean(MSE1A),3)))
      
# Histograms of Estimated ATE
pc.histPlot(ATE_Comp1A, DGPEST1A, PATH)
# Line Chart of Performance 
pc.linePlot(Perf1A, DGPEST1A, PATH)

#**** ***** ***** ***** ***** ***** ***** ***** ***** ***** *****
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 


# DGP 1 & Estimator B:
# -------------------- 
DGP = 1
Estimator = "B"
DGPEST1B = "DGP1_EstimatorB"

# Predefine Lists for Saving Results:
ATE1B=[]                         
ATE_Comp1B = []                       
SE1B =[]
Bias1B=[]
Variance1B=[]
MSE1B=[]

# Perform Sample Simulation 
for i in Sequence:
    SampleSize = i
    # Monte Carlo Simulation
    ATE, ATE_Comp, SE, ATE_True = pc.Simulation(Mean, Cov, betas, DGP, 
                                              Estimator, SampleSize, BS_Reps, 
                                              n, num_simulations)
    # Performance
    Bias, Variance, MSE = pc.Performance(ATE, SE, ATE_True)
    
    # Save Results
    ATE1B = np.append(ATE1B, ATE)
    ATE_Comp1B = np.append(ATE_Comp1B, ATE_Comp)
    SE1B = np.append(SE1B, SE)
    Bias1B = np.append(Bias1B, Bias)
    Variance1B = np.append(Variance1B, Variance)
    MSE1B= np.append(MSE1B, MSE)

# Save Performance in Dicitonary and Transform to DataFrame   
Perf1B = {"Bias": Bias1B, "Variance": Variance1B, "MSE": MSE1B}
Perf1B = pd.DataFrame(Perf1B, index = Sequence)
   
# Inform If Finished 
print("\nMonte Carlo Simulation 1B Finished:")
print("-------------------------------------")
# Print Mean ATE and Mean SE
print("MEAN ATE1B: " + str(round(np.mean(ATE1B),3)))
print("MEAN SE1B: " + str(round(np.mean(SE1B),3)))
print("MEAN Bias1B: " + str(round(np.mean(Bias1B),3)))
print("MEAN MSE1B: " + str(round(np.mean(MSE1B),3)))

# Histograms of Estimated ATE
pc.histPlot(ATE_Comp1B, DGPEST1B, PATH)
# Line Chart of Performance 
pc.linePlot(Perf1B, DGPEST1B, PATH)

#**** ***** ***** ***** ***** ***** ***** ***** ***** ***** *****
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 

# DGP 2 & Estimator A:
# -------------------- 
DGP = 2
Estimator = "A"
DGPEST2A = "DGP2_EstimatorA"

# Predefine Lists for Saving Results:
ATE2A=[]                         
ATE_Comp2A = []                     
SE2A =[]
Bias2A=[]
Variance2A=[]
MSE2A=[]

# Perform Sample Simulation 
for i in Sequence:
    SampleSize = i
    # Monte Carlo Simulation
    ATE, ATE_Comp, SE, ATE_True = pc.Simulation(Mean, Cov, betas, DGP, 
                                              Estimator, SampleSize, BS_Reps, 
                                              n, num_simulations)
    # Performance
    Bias, Variance, MSE = pc.Performance(ATE, SE, ATE_True)
    
    # Save Results
    ATE2A = np.append(ATE2A, ATE)
    ATE_Comp2A = np.append(ATE_Comp2A, ATE_Comp)
    SE2A = np.append(SE2A, SE)
    Bias2A = np.append(Bias2A, Bias)
    Variance2A = np.append(Variance2A, Variance)
    MSE2A= np.append(MSE2A, MSE)

# Save Performance in Dicitonary and Transform to DataFrame   
Perf2A = {"Bias": Bias2A, "Variance": Variance2A, "MSE": MSE2A}
Perf2A = pd.DataFrame(Perf2A, index = Sequence)

# Inform If Finished 
print("\nMonte Carlo Simulation 2A Finished:")
print("-------------------------------------")
# Print Mean ATE and Mean SE
print("MEAN ATE2A: " + str(round(np.mean(ATE2A),3)))
print("MEAN SE2A: " + str(round(np.mean(SE2A),3)))
print("MEAN Bias2A: " + str(round(np.mean(Bias2A),3)))
print("MEAN MSE2A: " + str(round(np.mean(MSE2A),3)))

# Histograms of Estimated ATE
pc.histPlot(ATE_Comp2A, DGPEST2A, PATH)
# Line Chart of Performance 
pc.linePlot(Perf2A, DGPEST2A, PATH) 

#**** ***** ***** ***** ***** ***** ***** ***** ***** ***** *****
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 

# DGP 2 & Estimator B:
# -------------------- 
DGP = 2
Estimator = "B"
DGPEST2B = "DGP2_EstimatorB"

# Predefine Lists for Saving Results:
ATE2B=[]                         
ATE_Comp2B = []                     
SE2B =[]
Bias2B=[]
Variance2B=[]
MSE2B=[]

# Perform Sample Simulation 
for i in Sequence:
    SampleSize = i
    # Monte Carlo Simulation
    ATE, ATE_Comp, SE, ATE_True = pc.Simulation(Mean, Cov, betas, DGP, 
                                              Estimator, SampleSize, BS_Reps, 
                                              n, num_simulations)
    # Performance
    Bias, Variance, MSE = pc.Performance(ATE, SE, ATE_True)
    
    # Save Results
    ATE2B = np.append(ATE2B, ATE)
    ATE_Comp2B = np.append(ATE_Comp2B, ATE_Comp)
    SE2B = np.append(SE2B, SE)
    Bias2B = np.append(Bias2B, Bias)
    Variance2B = np.append(Variance2B, Variance)
    MSE2B = np.append(MSE2B, MSE)

# Save Performance in Dicitonary and Transform to DataFrame   
Perf2B = {"Bias": Bias2B, "Variance": Variance2B, "MSE": MSE2B}
Perf2B = pd.DataFrame(Perf2B, index = Sequence)
   
# Inform If Finished 
print("\nMonte Carlo Simulation 2B Finished:")
print("-------------------------------------")
# Print Mean ATE and Mean SE
print("MEAN ATE2B: " + str(round(np.mean(ATE2B),3)))
print("MEAN SE2B: " + str(round(np.mean(SE2B),3)))
print("MEAN Bias2B: " + str(round(np.mean(Bias2B),3)))
print("MEAN MSE2B: " + str(round(np.mean(MSE2B),3)))

# Histograms of Estimated ATE
pc.histPlot(ATE_Comp2B, DGPEST2B, PATH)
# Line Chart of Performance 
pc.linePlot(Perf2B, DGPEST2B, PATH)

#**** ***** ***** ***** ***** ***** ***** ***** ***** ***** *****
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 

# DGP 3 & Estimator A:
# -------------------- 
DGP = 3
Estimator = "A"
DGPEST3A = "DGP3_EstimatorA"

# Predefine Lists for Saving Results:
ATE3A=[]                         
ATE_Comp3A = []                  
SE3A =[]
Bias3A=[]
Variance3A=[]
MSE3A=[]

# Perform Sample Simulation 
for i in Sequence:
    SampleSize = i
    # Monte Carlo Simulation
    ATE, ATE_Comp, SE, ATE_True = pc.Simulation(Mean, Cov, betas, DGP, 
                                              Estimator, SampleSize, BS_Reps, 
                                              n, num_simulations)
    # Performance
    Bias, Variance, MSE = pc.Performance(ATE, SE, ATE_True)
    
    # Save Results
    ATE3A = np.append(ATE3A, ATE)
    ATE_Comp3A = np.append(ATE_Comp3A, ATE_Comp)
    SE3A = np.append(SE3A, SE)
    Bias3A = np.append(Bias3A, Bias)
    Variance3A = np.append(Variance3A, Variance)
    MSE3A = np.append(MSE3A, MSE)

# Save Performance in Dicitonary and Transform to DataFrame   
Perf3A = {"Bias": Bias3A, "Variance": Variance3A, "MSE": MSE3A}
Perf3A = pd.DataFrame(Perf3A, index = Sequence)
      

# Inform If Finished 
print("\nMonte Carlo Simulation 3A Finished:")
print("-------------------------------------")
# Print Mean ATE and Mean SE
print("MEAN ATE3A: " + str(round(np.mean(ATE3A),3)))
print("MEAN SE3A: " + str(round(np.mean(SE3A),3)))
print("MEAN Bias3A: " + str(round(np.mean(Bias3A),3)))
print("MEAN MSE3A: " + str(round(np.mean(MSE3A),3)))


# Histograms of Estimated ATE
pc.histPlot(ATE_Comp3A, DGPEST3A, PATH)
# Line Chart of Performance 
pc.linePlot(Perf3A, DGPEST3A, PATH)

#**** ***** ***** ***** ***** ***** ***** ***** ***** ***** *****
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 

# DGP 3 & Estimator B:
# -------------------- 
DGP = 3
Estimator = "B"
DGPEST3B = "DGP3_EstimatorB"

# Predefine Lists for Saving Results:
ATE3B=[]                         
ATE_Comp3B = []                       
SE3B =[]
Bias3B=[]
Variance3B=[]
MSE3B=[]

# Perform Sample Simulation 
for i in Sequence:
    SampleSize = i
    # Monte Carlo Simulation
    ATE, ATE_Comp, SE, ATE_True = pc.Simulation(Mean, Cov, betas, DGP, 
                                              Estimator, SampleSize, BS_Reps, 
                                              n, num_simulations)
    # Performance
    Bias, Variance, MSE = pc.Performance(ATE, SE, ATE_True)
    
    # Save Results
    ATE3B = np.append(ATE3B, ATE)
    ATE_Comp3B = np.append(ATE_Comp3B, ATE_Comp)
    SE3B = np.append(SE3B, SE)
    Bias3B = np.append(Bias3B, Bias)
    Variance3B = np.append(Variance3B, Variance)
    MSE3B = np.append(MSE3B, MSE)

# Save Performance in Dicitonary and Transform to DataFrame   
Perf3B = {"Bias": Bias3B, "Variance": Variance3B, "MSE": MSE3B}
Perf3B = pd.DataFrame(Perf3B, index = Sequence)

# Inform If Finished 
print("\nMonte Carlo Simulation 3B Finished:")
print("-------------------------------------")
# Print Mean ATE and Mean SE
print("MEAN ATE3B: " + str(round(np.mean(ATE3B),3)))
print("MEAN SE3B: " + str(round(np.mean(SE3B),3)))
print("MEAN Bias3B: " + str(round(np.mean(Bias3B),3)))
print("MEAN MSE3B: " + str(round(np.mean(MSE3B),3)))

# Histograms of Estimated ATE
pc.histPlot(ATE_Comp3B, DGPEST3B, PATH)
# Line Chart of Performance 
pc.linePlot(Perf3B, DGPEST3B, PATH)

#**** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** 
#**** ***** ***** ***** End of Simulation ***** ***** ***** *****
#**** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** 



