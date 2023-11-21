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

Trial_Amplitude_History, Reversion_Amplitude_History, Threshold_History, Detection_History, Num_Trials_History, Reversion_Trials_History = StaircaseSimulation.SimulateTransformedStaircase(NumSimulations = 1000, 
                                 PsychometricCurveMu = 50,
                                 PsychometricCurveSigma = 10,
                                 StimulusIntensityStart = 0, # start of stimulus intensity range 
                                 StimulusIntensityStop = 100, # end of stimulus intensity range 
                                 MaxNumTrials = 1000, 
                                 MaxReversions = 100,  
                                 NumAFC = 2, 
                                 Criterion = [3,1], 
                                 InitialStepSize = 10, 
                                 StepFactor = 0.5,
                                 NumInitialReversionsSkipped = 0)

# # example staircase plot 
# StaircaseSimulation.PlotExampleStaircase(Trial_Amplitude_History = Trial_Amplitude_History,
#                          Reversion_Amplitude_History = Reversion_Amplitude_History,
#                          Detection_History = Detection_History, 
#                          Num_Trials_History = Num_Trials_History,
#                          Threshold_History = Threshold_History,
#                          Reversion_Trials_History = Reversion_Trials_History,
#                          SimulationNumber = 1)

''' Optimizing parameters for the 3up1down staircase in a 2AFC task'''

# For 1) and 2), calculate the reversion thresholds 
reversions_counted_thresholds, reversions_skipped_thresholds, NumReversions = StaircaseSimulation.CalculateReversionThresholds(Reversion_Amplitude_History) 

''' 1) Number of reversions '''

# plotting 
rc_mean_threshold = np.mean(reversions_counted_thresholds, axis=0)
rc_threshold_sd = np.std(reversions_counted_thresholds, axis=0)
rs_mean_threshold = np.mean(reversions_skipped_thresholds, axis=0)
rs_threshold_sd = np.std(reversions_skipped_thresholds, axis=0)

# number of reversions vs threshold calulcated 
plt.errorbar(list(range(NumReversions)), rc_mean_threshold, yerr=rc_threshold_sd, fmt='o-', color='b', capsize=5, label='Mean Threshold Estimate ± SD')
plt.axhline(y=target_intensity, color='green', linestyle='--', label=f'Staircase Convergence Threshold: {target_intensity:.2f}')
plt.xlabel('Number of Reversions')
plt.ylabel('Mean Threshold Estimate ± SD')
plt.legend()
plt.show()

# plot number of reversions vs mean threshold error (sd) 
plt.plot(list(range(NumReversions)), rc_threshold_sd)
plt.xlabel('Number of Reversions')
plt.ylabel('Error')
plt.title('Staircase Accuracy as a Function of # Reversions')

plt.show()

# plot number of reversions vs num of trials the staircase goes on until 



''' 2) Number of Initial Reversions Skipped '''
# number of initial reversions skipped vs threshold calulcated  
plt.errorbar(list(range(NumReversions-1)), rs_mean_threshold[:-1], yerr=rs_threshold_sd[:-1], fmt='o-', color='b', capsize=5, label='Mean Threshold Estimate ± SD')
plt.axhline(y=target_intensity, color='green', linestyle='--', label=f'Staircase Convergence Threshold: {target_intensity:.2f}')
plt.xlabel('Number of Initial Reversions Skipped')
plt.ylabel('Mean Threshold Estimate ± SD')
plt.xlim(0, NumReversions-2) 
plt.legend()
plt.show()

# plot number of reversions skipped vs mean threshold error (sd) 
plt.plot(list(range(NumReversions-1)), rs_threshold_sd[:-1])
plt.xlabel('Number of Initial Reversions Skipped')
plt.ylabel('Error')
plt.title('Staircase Accuracy as a Function of # Initial Reversions Skipped')
plt.show()

# plot number of reversions vs num of trials the staircase goes on until 

''' 3) Initial step size '''

initial_step_range = range(0, 15)

initial_step_thresholds = np.full((len(initial_step_range), 1), 0)
initial_step_errors = np.full((len(initial_step_range), 1), 0)

for step_sizes in initial_step_range: 
    Threshold_History = StaircaseSimulation.SimulateTransformedStaircase(NumSimulations = 1000, 
                                 PsychometricCurveMu = 50,
                                 PsychometricCurveSigma = 10,
                                 StimulusIntensityStart = 0, 
                                 StimulusIntensityStop = 100, 
                                 MaxNumTrials = 1000, 
                                 MaxReversions = 8,  
                                 NumAFC = 2, 
                                 Criterion = [3,1], 
                                 InitialStepSize = step_sizes, 
                                 StepFactor = 0.725,
                                 NumInitialReversionsSkipped = 0)[2]
    threshold = np.mean(Threshold_History) # calculate threshold for each initial step size 
    error = np.std(Threshold_History) # calculate the error 
    initial_step_thresholds[step_sizes, 0] = threshold 
    initial_step_errors[step_sizes, 0] = error 
    
plt.plot(list(initial_step_range), initial_step_thresholds)
plt.xlabel('Initial Step Size')
plt.ylabel('Estimated Threshold')
plt.show()

plt.plot(list(initial_step_range), initial_step_errors)
plt.xlabel('Initial Step Size')
plt.ylabel('Error')
plt.title('Staircase Accuracy as a Function of Initial Step Size')
plt.show()

''' 4) Step factor '''

step_factor_range = np.arange(0, 1, 0.2)

step_factor_thresholds = np.full((len(step_factor_range), 1), 0)
step_factor_errors = np.full((len(step_factor_range), 1), 0)

for i, step_factors in enumerate(step_factor_range): 
    Threshold_History = StaircaseSimulation.SimulateTransformedStaircase(NumSimulations = 1000, 
                                 PsychometricCurveMu = 50,
                                 PsychometricCurveSigma = 10,
                                 StimulusIntensityStart = 0,  
                                 StimulusIntensityStop = 100, 
                                 MaxNumTrials = 1000, 
                                 MaxReversions = 8,  
                                 NumAFC = 2, 
                                 Criterion = [3,1], 
                                 InitialStepSize = 10, 
                                 StepFactor = step_factors,
                                 NumInitialReversionsSkipped = 0)[2]
    threshold = np.mean(Threshold_History) # calculate threshold for each step factor 
    error = np.std(Threshold_History) # calculate the error 
    step_factor_thresholds[i, 0] = threshold 
    step_factor_errors[i, 0] = error 
    
plt.plot(list(step_factor_range), step_factor_thresholds)
plt.xlabel('Step Factor')
plt.ylabel('Estimated Threshold')
plt.show()

plt.plot(list(step_factor_range), step_factor_errors)
plt.xlabel('Step Factor')
plt.ylabel('Error')
plt.title('Staircase Accuracy as a Function of Step Factor')
plt.show()






