from Package import StaircaseSimulation

import matplotlib.pyplot as plt
import numpy as np
# from scipy.stats import norm
# import random

''' plotting the psychometric function, 
calculating the stimulus intensity at which p(correct response) = 0.5 in a 2AFC 3up1down staircase '''

stimulus_range, pr_correct = StaircaseSimulation.GetPsychometricFunction(PsychometricCurveMu = 50,
                                 PsychometricCurveSigma = 10,
                                 StimulusIntensityStart = 0, 
                                 StimulusIntensityStop = 100)

target_probability, NumAFC, Criterion = StaircaseSimulation.GetStaircaseConvergenceTarget(NumAFC = 2, 
                                    Criterion = (3,1))

target_intensity = StaircaseSimulation.GetStaircaseConvergenceIntensity(stimulus_range, pr_correct, target_probability)

StaircaseSimulation.PlotPsychometricFunctionTarget(stimulus_range, pr_correct, target_probability, target_intensity, Criterion, NumAFC)

''' running the staircase simulation '''

Trial_Amplitude_History, Reversion_Amplitude_History, Threshold_History, Detection_History = StaircaseSimulation.SimulateTransformedStaircase(NumSimulations = 1000, 
                                 PsychometricCurveMu = 50,
                                 PsychometricCurveSigma = 10,
                                 StimulusIntensityStart = 0, # start of stimulus intensity range 
                                 StimulusIntensityStop = 100, # end of stimulus intensity range 
                                 MaxNumTrials = 1000, 
                                 MaxReversions = 10,  
                                 NumAFC = 2, 
                                 Criterion = (3,1), 
                                 InitialStepSize = 10, 
                                 StepFactor = 0.8,
                                 NumInitialReversionsSkipped = 0)

reversions_counted_thresholds, reversions_skipped_thresholds, NumReversions = StaircaseSimulation.CalculateReversionThresholds(Reversion_Amplitude_History)

# example staircase plot 






''' Optimizing parameters for the 3up1down staircase in a 2AFC task'''

''' 1) Number of reversions '''

# plotting 
rc_mean_threshold = np.mean(reversions_counted_thresholds, axis=0)
rc_threshold_sd = np.std(reversions_counted_thresholds, axis=0)
rs_mean_threshold = np.mean(reversions_skipped_thresholds, axis=0)
rs_threshold_sd = np.std(reversions_skipped_thresholds, axis=0)

# number of reversions counted plot 
plt.errorbar(list(range(NumReversions)), rc_mean_threshold, yerr=rc_threshold_sd, fmt='o-', color='b', capsize=5, label='Mean Threshold Estimate ± SD')
plt.axhline(y=target_intensity, color='green', linestyle='--', label=f'Staircase Convergence Threshold: {target_intensity:.2f}')
plt.xlabel('Number of Reversions')
plt.ylabel('Mean Threshold Estimate ± SD')
plt.legend()
plt.show()

# plot error 

''' 2) Number of Initial Reversions Skipped '''
# number of initial reversions skipped plot 
plt.errorbar(list(range(NumReversions-1)), rs_mean_threshold[:-1], yerr=rs_threshold_sd[:-1], fmt='o-', color='b', capsize=5, label='Mean Threshold Estimate ± SD')
plt.axhline(y=target_intensity, color='green', linestyle='--', label=f'Staircase Convergence Threshold: {target_intensity:.2f}')
plt.xlabel('Number of Initial Reversions Skipped')
plt.ylabel('Mean Threshold Estimate ± SD')
plt.xlim(0, NumReversions-2) 
plt.legend()
plt.show()

# plot error 

## plot number of reversions skipped vs number of reversions - heatmap? 

''' 3) Initial step size '''

# plot initial step size vs number of reversions 
# plot error 


''' 4) Step factor (keeping the initial step size, number of reversions/skipped,  '''

# plot initial step size vs number of reversions 
# plor error 

## plot initial step size initial vs factor 





