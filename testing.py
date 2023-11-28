from Package import StaircaseSimulation

import matplotlib.pyplot as plt
import numpy as np

# from scipy.stats import norm
# import random

#%%
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
#%% 

'''Example Plot''' 
(Trial_Amplitude_History, Reversion_Amplitude_History, Threshold_History,
 Detection_History, Num_Trials_History, Reversion_Trials_History) = StaircaseSimulation.SimulateTransformedStaircase(NumSimulations = 1,
    PsychometricCurveMu = 50,
    PsychometricCurveSigma = 10,
    StimulusIntensityStart = 0, # start of stimulus intensity range 
    StimulusIntensityStop = 100, # end of stimulus intensity range 
    MaxNumTrials = 100, 
    MaxReversions = 8,  
    NumAFC = 2, 
    Criterion = [3,1], 
    InitialStepSize = 10, 
    StepFactor = 0.725,
    NumInitialReversionsSkipped = 2, 
    use_MLE = True)
   
NumTrials = Num_Trials_History[0][0]
Trial_Amplitude_History = Trial_Amplitude_History[:NumTrials,0]
Detection_History = Detection_History[:NumTrials,0]
Reversion_Trials_History = Reversion_Trials_History[:,0]
Reversion_Amplitude_History = Reversion_Amplitude_History[:,0]
    
fig, ax = plt.subplots()
x_detect = np.where(Detection_History == 1)[0]
x_miss = np.where(Detection_History == 0)[0]

ax.plot(range(NumTrials), Trial_Amplitude_History, color = 'gray', linestyle = ':')
ax.scatter(x_detect, Trial_Amplitude_History[x_detect], color = 'green')
ax.scatter(x_miss, Trial_Amplitude_History[x_miss], color = 'red')
ax.scatter(Reversion_Trials_History, Reversion_Amplitude_History, marker = '*', color = 'gold')
plt.ylim(0, 100)

print(np.mean(Reversion_Amplitude_History))
print(np.std(Reversion_Amplitude_History, axis = 0))

# example plot specifying the simulation number 

# StaircaseSimulation.PlotExampleStaircase(Trial_Amplitude_History = Trial_Amplitude_History,
#                          Reversion_Amplitude_History = Reversion_Amplitude_History,
#                          Detection_History = Detection_History, 
#                          Num_Trials_History = Num_Trials_History,
#                          Threshold_History = Threshold_History,
#                          Reversion_Trials_History = Reversion_Trials_History,
#                          SimulationNumber = 1)
#%%

''' running the staircase simulation '''

Trial_Amplitude_History, Reversion_Amplitude_History, Threshold_History, Detection_History, Num_Trials_History, Reversion_Trials_History = StaircaseSimulation.SimulateTransformedStaircase(NumSimulations = 1000, 
                                 PsychometricCurveMu = 50,
                                 PsychometricCurveSigma = 10,
                                 StimulusIntensityStart = 0, # start of stimulus intensity range 
                                 StimulusIntensityStop = 100, # end of stimulus intensity range 
                                 MaxNumTrials = 1000, 
                                 MaxReversions = 30,  
                                 NumAFC = 2, 
                                 Criterion = [3,1], 
                                 InitialStepSize = 10, 
                                 StepFactor = 0.725,
                                 NumInitialReversionsSkipped = 0)
#%%

''' Optimizing parameters for the 3up1down staircase in a 2AFC task'''

# For 1) and 2), calculate the reversion thresholds 
reversions_counted_thresholds, reversions_skipped_thresholds, NumReversions = StaircaseSimulation.CalculateReversionThresholds(Reversion_Amplitude_History = Reversion_Amplitude_History) 

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
plt.ylabel('Mean Threshold Estimate ± σ')
plt.legend()
plt.show()

# plot number of reversions vs mean threshold error (sd) 
plt.plot(list(range(NumReversions)), rc_threshold_sd)
plt.xlabel('Number of Reversions')
plt.ylabel('Error (σ)')
plt.title('Staircase Accuracy as a Function of # Reversions')

plt.show()


# obviously, now we know that as the number of reversions increases, the staircase becomes more accurate 
# however, the number of trials also increases (time taken for the procedure increases)

''' num of trials vs number of reversions + error '''

num_reversions_range = range(1, 10)

# initialize 1d array to store mean num of trials for maxreversions 
numtrials_by_maxreversions = np.zeros((len(num_reversions_range)))
threshold_error_by_maxreversions = np.zeros(len(num_reversions_range))

for i, reversion_number in enumerate(num_reversions_range):
    _, _, Threshold_History, _, Num_Trials_History, _ = StaircaseSimulation.SimulateTransformedStaircase(
        NumSimulations=10, 
        PsychometricCurveMu=50,
        PsychometricCurveSigma=10,
        StimulusIntensityStart=0, 
        StimulusIntensityStop=100, 
        MaxNumTrials=1000, 
        MaxReversions=reversion_number,  
        NumAFC=2, 
        Criterion=[3,1], 
        InitialStepSize=5, 
        StepFactor=0.725,
        NumInitialReversionsSkipped=0
    )

    # calculate mean number of trials and standard deviation of the threshold
    num_trials = np.mean(Num_Trials_History)
    threshold_error = np.std(Threshold_History)

    # store the values in the arrays
    numtrials_by_maxreversions[i] = num_trials
    threshold_error_by_maxreversions[i] = threshold_error

# plot for mean number of trials
plt.plot(num_reversions_range, numtrials_by_maxreversions)
plt.xlabel('Number of Reversions')
plt.ylabel('Mean Number of Trials')
plt.xticks(range(1, len(num_reversions_range) + 1))  # adjust the x-axis ticks
plt.show()

# plot for standard deviation of threshold estimates vs number of trials 
plt.scatter(numtrials_by_maxreversions, threshold_error_by_maxreversions)
plt.xlabel('Mean Number of Trials')
plt.ylabel('Threshold Estimate Error (σ)')
plt.show()


''' 2) Number of Initial Reversions Skipped '''
# number of initial reversions skipped vs threshold calulcated  
plt.errorbar(list(range(NumReversions-1)), rs_mean_threshold[:-1], yerr=rs_threshold_sd[:-1], fmt='o-', color='b', capsize=5, label='Mean Threshold Estimate ± SD')
plt.axhline(y=target_intensity, color='green', linestyle='--', label=f'Staircase Convergence Threshold: {target_intensity:.2f}')
plt.xlabel('Number of Initial Reversions Skipped')
plt.ylabel('Mean Threshold Estimate ± σ')
plt.xlim(0, NumReversions-2) 
plt.legend()
plt.show()

# plot number of reversions skipped vs mean threshold error (sd) 
plt.plot(list(range(NumReversions-1)), rs_threshold_sd[:-1])
plt.xlabel('Number of Initial Reversions Skipped')
plt.ylabel('Error (σ)')
plt.title('Staircase Accuracy as a Function of # Initial Reversions Skipped')
plt.show()

''' number of initial reversions skipped vs number of reversions - error heatmap'''

# define the ranges 

max_reversions_range = range(5, 11)  # range for MaxReversions
num_initial_reversions_skipped_range = range(0, 5)  # range for NumInitialReversionsSkipped

# initialize the 2D array for standard deviations
error_matrix = np.zeros((len(max_reversions_range), len(num_initial_reversions_skipped_range)))

# run simulations for each combination
for i, max_reversions in enumerate(max_reversions_range):
    for j, num_initial_reversions_skipped in enumerate(num_initial_reversions_skipped_range):
        Threshold_History = StaircaseSimulation.SimulateTransformedStaircase(NumSimulations = 1000, 
                                     PsychometricCurveMu = 50,
                                     PsychometricCurveSigma = 10,
                                     StimulusIntensityStart = 0, 
                                     StimulusIntensityStop = 100, 
                                     MaxNumTrials = 1000, 
                                     MaxReversions = max_reversions,  
                                     NumAFC = 2, 
                                     Criterion = [3,1], 
                                     InitialStepSize = 5, 
                                     StepFactor = 0.725,
                                     NumInitialReversionsSkipped = num_initial_reversions_skipped)[2]
        error_matrix[i, j] = np.std(Threshold_History) # calculate the standard deviation for this combination
    
# plotting the heatmap
plt.imshow(error_matrix, cmap='hot', interpolation='nearest')
plt.colorbar()  # add color bar to show the scale of errors
plt.xlabel('Number of Reversions')
plt.ylabel('Number of Initial Reversions Skipped')
plt.title('Error (σ)')

plt.xticks(range(len(max_reversions_range)), max_reversions_range)  # replace with MaxReversions values
plt.yticks(range(len(num_initial_reversions_skipped_range)), num_initial_reversions_skipped_range)  # replace with NumInitialReversionsSkipped values

plt.show()


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
plt.ylabel('Error (σ)')
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
plt.ylabel('Error (σ)')
plt.title('Staircase Accuracy as a Function of Step Factor')
plt.show()

#%% 
# using MLE for threshold estimation 





















