"""
### University of St. Gallen 
### Data Analytics II: Self Study Project 
### Kevin Hardegger (12-758-785)
### kevin.hardegger@student.unisg.ch

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from numpy.random import seed
from numpy.random import multivariate_normal
import statsmodels.api as sm
import scipy.stats

# Define parameters 
#Mean = [5, -5]                  # mean of multivariate normal
#Cov = [[2, 0.5], [0.5, 3]]      # diagonal covariance matrix of MVN
#betas = np.array([2,1,5,8])     # Treatment,Intercept, X1, X2
#num_simulations = 100           # number of simulations
#n = 1000                        # population
#BS_Reps = 100                   # number of Bootstrap Repetitions
#BS_Size = 100                   # sample size for bootstrao
#Sequence= range(100, 1050, 50)  # Sequence of sample sizes
#ConfLevel =0.95                 # Confidence Level 
#seed(200)
# covar[:,0] # first column
# covar[:,1] # second column


def DGP1(Mean, Cov, betas, n):
    """
    Data Generating Process 1 With Random Treatment Assignment.
    
    Parameter:
    ------------
    Mean: Vector of Means (1D array, float)
    Cov: Variance-Covariance Matrix(2D array, float)
    betas: Vector of Betas (1D array, float)
    n: Population (integer)
    ------------------------------------------------
    Returns: 
    -------
    Y: Outcome (1D array, float)
    X: Covariates (2D array, float)
    D: Treatment (1D array, float)
    ATE_True: True ATE (integer)
    """
    seed(200)
    # Create Covariate Matrix
    X = multivariate_normal(Mean, Cov, n)
    # Create Dummy Treatment Array With Random Condition
    Condition = np.random.normal(0,1,n)
    D = np.zeros(n)
    D[Condition >= 0] = 1
    # Create Complete Variable/Covariate Matrix 
    X_comp = np.c_[D,np.ones(n),X]
    X_const = np.c_[np.ones(n),X]
    # Random Noise 
    u = np.random.normal(0,3,n)
    # Create Outcome
    Y = X_comp @ betas + u
    ATE_True = betas[0]
    return Y, X_const, D, ATE_True


def DGP2(Mean, Cov, betas, n):
    """
    Data Generating Process 2 With Non-Random Treatment Assignment
    
    Parameter:
    ------------
    Mean: Vector of Means (1D list, float)
    Cov: Variance-Covariance Matrix(2D array, float)
    betas: Vector of Betas (1D array, float)
    n: Population (integer)
    ------------------------------------------------
    Returns: 
    -------
    Y: Outcome (1D array, float)
    X: Covariates (2D array, float)
    D: Treatment (1D array, float)
    ATE_True: True ATE (integer)
    """
    seed(200)
    # Create Covariate Matrix
    X = multivariate_normal(Mean, Cov, n)
    # Create Dummy Treatment Array With Non-Random Condition
    e = np.random.normal(0,20,n)
    Condition = X @ betas[2:] + e
    D = np.zeros(n)
    D[Condition >= 0] = 1
    # Create Complete Variable/Covariate Matrix 
    X_comp = np.c_[D,np.ones(n),X]
    X_const = np.c_[np.ones(n),X]
    # Random Noise 
    u = np.random.normal(0,3,n)
    # Create Outcome
    Y = X_comp @ betas + u
    ATE_True = betas[0]
    return Y, X_const, D, ATE_True


def DGP3(Mean, Cov, betas, n):
    """
    Data Generating Process 3 With Non-Random Treatment Assignment Dependent on Exogen Confounder V,
    Which Affects Both Treatment Process and Outcome Y.
    
    Parameter:
    ------------
    Mean: Vector of Means (1D array, float)
    Cov: Variance-Covariance Matrix(2D array, float)
    betas: Vector of Betas (1D array, float)
    n: Population (integer)
    ------------------------------------------------
    Returns: 
    -------
    Y: Outcome (1D array, float)
    X: Covariates (2D array, float)
    D: Treatment (1D array, float)
    ATE_True: True ATE (integer)
    """
    seed(200)
    # Create Covariate Matrix
    X = multivariate_normal(Mean, Cov, n)
    # Create Dummy Treatment Array With Non-Random Condition Dependent of Exogen Confounder
    V = np.random.normal(-1,3,n)
    Condition = V
    D = np.zeros(n)
    D[Condition >= 0] = 1
    # Create Complete Variable/Covariate Matrix 
    X_comp = np.c_[D,np.ones(n),X]
    X_const = np.c_[np.ones(n),X]
    # Random Noise 
    u = np.random.normal(0,3,n)
    # Create Outcome
    Y = X_comp @ betas + u + V
    ATE_True = betas[0]
    return Y, X_const, D, ATE_True


def OLS_Est(Y,X):
    """
    Estimates Expected Outcome Y_hat With OLS.
    
    Parameter:
    ----------
    Y: Outcome (1D array, float)
    X: Covariates (2D array, float)
    ----------
    Returns:
    
    Y_hat: Expected Outcome Values for (Y|X) (1D array, float)
    """
    # Calculate Betas in Matrix Form
    OLS_Beta = np.linalg.inv(X.T @ X) @ X.T @ Y
    # Create Estimated Outcome
    Y_hat = X @ OLS_Beta
    return Y_hat


def IPW(Y,D,X):
    """
    Calculates the Average Treatment Effect With Inverse Probability Weighting Estimation.
    
    Parameter:
    ----------
    Y: Outcome (1D array, float)
    D: Treatment (1D array, float)
    X: Covariates (2D array, float)
    ----------
    Returns:
    ATE_IPW: Estimated IPW Average Treatment Effect (float)
    """
    # Remove Constant For PS Calculation
    XPS=X[:,1:]
    # Calculate Propensity Score with Logit Maximum Likelihood Estimation
    Pscore = sm.Logit(endog=D, exog=XPS).fit(disp=0).predict()
    # Calculate Average Treatment Effect ATE of IPW Estimation
    ATE_IPW = np.mean(((D*Y)/Pscore) - ((1-D)*Y)/(1-Pscore))
    return ATE_IPW


def DoublyR (Y,D,X):
    """
    Calculates the Average Treatment Effect With Doubly Robust Estimation.
    
    Parameter:
    ----------
    Y: Outcome (1D array, float)
    D: Treatment (1D array, float)
    X: Covariates (2D array, float)
    ----------
    Returns:
    ATE_DR: Estimated Doubly Robust Average Treatment Effect (float)
    
    """
    # Remove Constant For PS Calculation
    XPS=X[:,1:]
    # Calculate Propensity Score with Logit Maximum Likelihood Estimation
    Pscore = sm.Logit(endog=D, exog=XPS).fit(disp=0).predict()
    # Split Outcome and Covariates According to Treated/Untreated
    Y1 = Y[D==1]
    X1 = X[D==1]
    Y0 = Y[D==0]
    X0 = X[D==0]
    # Estimate Expected Outcome Y_hat With OLS 
    Y1_hat = OLS_Est(Y1, X1)
    Y0_hat = OLS_Est(Y0, X0)
    # Calculte Average Treatment Effect on Treated (ATET): mu1, mu0 is Untreated 
    mu1 = np.mean(Y1_hat)
    mu0 = np.mean(Y0_hat)
    # Calculate Average Treatment Effect ATE of Doubly Robust Estimator
    ATE_DR = mu1-mu0 + np.mean((D*(Y-mu1))/Pscore - ((1-D)*(Y-mu0))/(1-Pscore))
    return ATE_DR
    

def ATE_Estimation(Y, D, X, Estimator):
    """
    Estimates Average Treatment Effect ATE According to Given Estimator.
    
    Parameter:
    ----------
    Y: Outcome (1D array, float)
    D: Treatment (1D array, float)
    X: Covariates (2D array, float)
    Estimator: Estimation Method (A = Inverse Probability Weighting IPW, B = Doubly Robust, string)
    ----------
    Returns:
    
    ATE_Est: Estimated Average Treatment Effect ATE (float)
    """
    #
    if Estimator == "A": 
        ATE_Es = IPW(Y,D,X)
    elif Estimator == "B":
        ATE_Es = DoublyR(Y,D,X)
    return ATE_Es


def ATE_SE_Bootstrap(Y,D,X, BS_Reps, SampleSize, Estimator):
    """
    Caculates Standard Erros for Estimated Average Treatment Effects by Using Bootstrap.
    
    Parameter:
    ----------
    Y: Outcome (1D array, float)
    D: Treatment (1D array, float)
    X: Covariates (2D array, float)
    BS_Reps: Number of Repetitions for Bootstrap (integer)
    SampleSize: Size of Sample (integer)
    Estimator: Estimation Method (A = Inverse Probability Weighting IPW, B = Doubly Robust, string)
    ----------
    Returns:
    
    ATE_SE: Standard Error for Estimated Bootstrap Average Treatment Effect (float)
    """
    # Define array ATE Estimates & Create Data for Bootstrap Sampling
    ATE_Estimates = []
    Data = np.c_[Y,D,X]
    # for Loop Bootstrap
    for i in range(BS_Reps):
        # Take Sample with Replacement
        sample = Data[np.random.choice(Data.shape[0], len(Data), replace=True)]
        # Split Outcome, Treatment & Covariates to insert them in ATE_Estimation Function
        Y_sample = sample[:,0]
        D_sample = sample[:,1]
        X_sample = sample[:,2:]
        # Save ATE Estimation after estimating
        ATE_Est = ATE_Estimation(Y_sample,D_sample,X_sample, Estimator)
        ATE_Estimates.append(ATE_Est)
    # Calculate Standard Deviation for ATE Estimations     
    ATE_SE = np.std(ATE_Estimates)
    return  ATE_SE    
    

def Performance(ATE_Est, SE, ATE_True):
    """
    Captures Performance of Estimators by Calculating Bias, Variance, and MSE.
    
    Parameter:
    ----------
    ATE_Est: Estimated Average Treatment Effect (1D array, float)
    ATE_True: True Average Treatment Effect (integer)
    ----------
    Returns:
    
    Performance: Bias, Variance, and MSE Combined (2D array, float)
    
    """
    # Calculate Bias, Variance, MSE & Save them as one array "Performance"
    Bias = np.mean(ATE_Est - ATE_True)
    Variance = SE**2
    MSE = Bias**2 + Variance  # Performance = [Bias, Variance, MSE]
    return Bias, Variance, MSE


def Simulation(Mean, Cov, betas, DGP, Estimator, SampleSize, BS_Reps, n, num_simulations):
    """
    Performs Monte Carlo Simulation for Given Specifications and Estimates ATE.
    
    Parameter:
    ----------
    Mean: Vector of Means (1D array, float)
    Cov: Variance-Covariance Matrix(2D array, float)
    betas: Vector of Betas (1D array, float)
    DGP: Data Generating Process (1=DGP1, 2=DGP2, 3=DGP3, integer)
    Estimator: Estimation Method (A = Inverse Probability Weighting IPW, B = Doubly Robust, string)
    BS_Size: Sample Size for Bootstrap (integer)
    BS_Reps: Number of Repetitions for Bootstrap (integer)
    n: Population (integer)
    num_simulations: number of simulations (integer)
    
    ----------
    Returns:
    ATE_Est_List: Results of ATE Estimation (1D array, float)
    ATE_SE_List: Results of ATE Bootstrap Standard Errors(1D array, float)
    ATE_True: True Average Treatment Effect (integer)
    """ 
    # IF Statement for Data Generating Process
    if DGP == 1:
        Y, X, D, ATE_True = DGP1(Mean, Cov, betas, n)
    elif DGP == 2:
        Y, X, D, ATE_True = DGP2(Mean, Cov, betas, n)
    elif DGP == 3:
        Y, X, D, ATE_True = DGP3(Mean, Cov, betas, n)
    # Prepare Data for Sampling
    Data = np.c_[Y,D,X]
    # Create List for Results
    ATE_Est_List=[]
    ATE_SE_List=[]
    # Start for loop for Monte Carlo Simulation
    for i in range(1, num_simulations):
        sample= Data[np.random.choice(Data.shape[0], SampleSize, replace=False)]
        Y_sample = sample[:,0]
        D_sample = sample[:,1]
        X_sample = sample[:,2:]
        # Estimate ATE and then Standard Errors With Bootstrap
        ATE_Est = ATE_Estimation(Y_sample, D_sample, X_sample, Estimator)
        ATE_SE = ATE_SE_Bootstrap(Y_sample, D_sample, X_sample, BS_Reps, SampleSize, Estimator)
        ATE_Est_List.append(ATE_Est)
        ATE_SE_List.append(ATE_SE)
    # Calculate Average ATE And Standard Deviation
    ATE = np.mean(ATE_Est_List)
    SE = np.mean(ATE_SE_List)
          
    return ATE, ATE_Est_List, SE, ATE_True


def histPlot(data, DGPEST , PATH):
    """
    Plots Histogram and Saves on PATH.

    Parameter:
    ----------
    data: Data To Plot (1D array, float)
    DGPEST: Specified DGP and Estimator (string)
    PATH: PATH
    ----------
    Returns:

    Hisogramm
    """
    # My General Specifications
    plot.hist(data, bins = 35, rwidth = 0.8, color="lightcoral")
    plot.xlabel("ATE_Estimation " + DGPEST)
    plot.ylabel('Counts')
    plot.title('Histogram of ' + DGPEST)
    plot.grid(axis='y', alpha=0.75)
    plot.savefig(PATH + '/histogram_of_' + DGPEST + '.png')
    plot.show()


def linePlot(Perf, DGPEST, PATH):
    """
    Plots Line Chart of Performance

    Parameter:
    ----------
    Perf: Data To Plot (pandas DataFrame)
    DGPEST: Specified DGP and Estimator (string)
    PATH: PATH
    ----------
    Returns:

    Line Chart

    """
    Perf.plot(kind="line", xlabel="SampleSize", ylabel="Value", title="Performance of "+DGPEST)
    plot.savefig(PATH + '/LinePlot_' + DGPEST + '.png')
    plot.show()

