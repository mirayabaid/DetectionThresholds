#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:55:09 2023

@author: mirayabaid
"""

# Transformed Staircase Simulation 

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import random

def GetPsychometricFunction(PsychometricCurveMu = 50,
                                 PsychometricCurveSigma = 10,
                                 StimulusIntensityStart = 0, 
                                 StimulusIntensityStop = 100):
    stimulus_range = np.linspace(StimulusIntensityStart, StimulusIntensityStop, 100)
    pr_correct = norm.cdf(stimulus_range, loc = PsychometricCurveMu, scale = PsychometricCurveSigma) 
    return stimulus_range, pr_correct

stimulus_range, pr_correct = GetPsychometricFunction()

 # get the staircase convergence point when the observer's performance can be attributed to ONLY their sensitivity and not guessing by chance 
 
def GetStaircaseConvergenceTarget(NumAFC = 2, 
                                    Criterion = (3,1)): 
    chance = 1/NumAFC 
    unadjusted_target= 0.5**(1/(Criterion[0]/Criterion[1]))
    # adjusting the convergence point to account for chance 
    target = (unadjusted_target - chance) 
    # scaling this difference to the proportion of the range that's above chance 
    target_probability = target/(1 - chance)
    return target_probability, NumAFC, Criterion
                                        
target_probability, NumAFC, Criterion = GetStaircaseConvergenceTarget()

# find the stimulus amplitude at which the target staircase convergence target is reached 
# basically the amplitude at which p(correctly detecting) = 0.5 (point of subjective equality)

def GetStaircaseConvergenceIntensity(stimulus_range, pr_correct, target_probability): # values 
    
    # find value in pr_correct that is closest to the target, then index by that value to find a stimulus intensity value 
    # first find the absolute difference between pr_correct values and the target 
    # whichever value from the resulting array is the smallest, find the index of that value, then use to find the target intensity 
    
    target_index = np.abs(pr_correct - target_probability).argmin()
    target_intensity = stimulus_range[target_index]
    return target_intensity 

target_intensity = GetStaircaseConvergenceIntensity(stimulus_range, pr_correct, target_probability)
    
def PlotPsychometricFunctionTarget(stimulus_range, pr_correct, target_probability, target_intensity):
    plt.plot(stimulus_range, pr_correct, color='red', label='Psychometric Function')
    plt.axhline(y=target_probability, color='blue', linestyle='--', label=f'Convergence Target (p): {target_probability:.2f}')
    plt.axvline(x=target_intensity, color='green', linestyle='--', label=f'Convergence Amplitude: {target_intensity:.2f}')
    plt.xlabel('Stimulus Intensity')
    plt.ylabel('p(correct)')
    plt.title(f'{Criterion[0]}-Down-{Criterion[1]}-Up Staircase Convergence Target for {NumAFC}AFC Task')
    plt.legend()
    plt.show()

PlotPsychometricFunctionTarget(stimulus_range, pr_correct, target_probability, target_intensity)


def PlotStaircaseProcedure(PsychometricCurveMu = 50,
                                 PsychometricCurveSigma = 10,
                                 StimulusIntensityStart = 0, 
                                 StimulusIntensityStop = 100, 
                                 NumAFC = 2, 
                                 InitialStepSize = 5, 
                                 StepFactor = 0.5, 
                                 MaxNumTrials = 100, 
                                 MaxNumReversions = 10,
                                 Criterion = (3,1),
                                 NumInitialReversionsSkipped = 0):
    # initialize variables
    stimulus_range = np.linspace(StimulusIntensityStart, StimulusIntensityStop, 100) 
    stimulus_intensity = random.choice(stimulus_range)
    step_size = InitialStepSize
    reversion_counter = 0
    reached_criterion = False
    current_direction, new_direction = 0, 0
    cumulative_score = [0,0] #[correct, incorrect]
    intensity_values = []
    reversions = [] # stores the trial number of a reversal
    reversion_intensities = []
    chance = 1/NumAFC
    

    for trial in range(MaxNumTrials): 
        if np.random.uniform(0, 1) < norm.cdf(stimulus_intensity, loc = PsychometricCurveMu, scale = PsychometricCurveSigma) or np.random.uniform(0, 1) > chance:
            response = 1 # correct, so stimulus intensity decreases 
            color = 'green'
        else: 
            response = 0
            color = 'red'
            
        intensity_values.append(stimulus_intensity)
        
        print(f"Trial = {trial + 1}: Stimulus Intensity = {stimulus_intensity}, Response = {response}")
        
        plt.step(trial, stimulus_intensity)
        plt.scatter(trial, stimulus_intensity, color = color)
        plt.xlabel('Trial #')
        plt.ylabel('Stimulus Intensity')
        
        # Evaluate response
        if response == 1: 
            cumulative_score[0] += 1
        else:
            cumulative_score[1] +=1 
        
        # Evaluate criterion
        if cumulative_score[0] >= Criterion[0]: # if three correct responses
            new_direction = -1 # direction = down 
            stimulus_intensity = max(0, stimulus_intensity - step_size) 
            cumulative_score = [0,0]
            reached_criterion = True
        elif cumulative_score[1] == Criterion[1]: # if one wrong 
            new_direction = +1
            stimulus_intensity = min(100, stimulus_intensity + step_size)
            cumulative_score = [0,0]
            reached_criterion = True
        
        # compare new direction vs old direction 
        if reached_criterion:
            if current_direction == 0: 
                current_direction = new_direction
            elif current_direction != new_direction:
                reversion_counter += 1
                reversions.append(trial)
                reversion_intensities.append(intensity_values[trial])
                plt.scatter(trial, intensity_values[trial], facecolors = "none", edgecolors = "black")
                current_direction = new_direction
                step_size = step_size*StepFactor
            
        if reversion_counter == MaxNumReversions:
            break 
        
    if NumInitialReversionsSkipped == 0:
            staircase_threshold = sum(reversion_intensities)/len(reversion_intensities)
    else:
            staircase_threshold =  (sum(reversion_intensities) - sum(reversion_intensities[:NumInitialReversionsSkipped-1]))/(len(reversion_intensities) - NumInitialReversionsSkipped)
        
    plt.axhline(y = staircase_threshold, color='blue', linestyle='--')
    so = print(f'Estimated Threshold: {staircase_threshold:.2f}')
    
    return plt.show(), so


PlotStaircaseProcedure()

def SimulateTransformedStaircase(NumSimulations = 1000, 
                                 PsychometricCurveMu = 50,
                                 PsychometricCurveSigma = 10,
                                 StimulusIntensityStart = 0, # start of stimulus intensity range 
                                 StimulusIntensityStop = 100, # end of stimulus intensity range 
                                 MaxNumTrials = 1000, 
                                 MaxReversions = 15,  
                                 NumAFC = 2, 
                                 Criterion = (3,1), 
                                 InitialStepSize = 10, 
                                 StepFactor = 0.8,
                                 NumInitialReversionsSkipped = 0):
    
    # initialize variables for every simulation 
    
    # store stimulus intensity values by trial number for each simulation 
    Trial_Amplitude_History = np.full((MaxNumTrials, NumSimulations), 0)
    
    # store stimulus intensity values for each reversion number for each simulation 
    Reversion_Amplitude_History = np.full((MaxReversions, NumSimulations), 0)
    
    for simulation in range(NumSimulations):
        
        # initialize variables for every trial 
        StimulusIntensity = random.randrange(StimulusIntensityStart, StimulusIntensityStop)
        trial_counter = 0 
        reversion_counter = 0 
        reached_criterion = False 
        current_direction, new_direction = 0, 0
        cumulative_score = [0, 0]
        step_size = InitialStepSize
        chance = 1/NumAFC
        
        for trial in range(MaxNumTrials): 
            
            Trial_Amplitude_History[trial, simulation] = StimulusIntensity
            trial_counter += 1
            
            if np.random.uniform(0, 1) < norm.cdf(StimulusIntensity, loc = PsychometricCurveMu, scale = PsychometricCurveSigma) or np.random.uniform(0, 1) > (1 - chance):
                response = 1 # correct, so stimulus intensity decreases 
            else: 
                response = 0 # incorrect, so stimulus intensity increases
               
            if response == 1: 
                cumulative_score[0] += 1
            else:
                cumulative_score[1] +=1 
            
            if cumulative_score[0] >= Criterion[0]: # if 3 correct 
                new_direction = -1
                StimulusIntensity = max(0, StimulusIntensity - step_size) 
                cumulative_score = [0,0]
                reached_criterion = True
            elif cumulative_score[1] == Criterion[1]: # if one wrong 
                new_direction = +1
                StimulusIntensity = min(100, StimulusIntensity + step_size)
                cumulative_score = [0,0]
                reached_criterion = True
             
            if reached_criterion == True:
                 if current_direction == 0: 
                    current_direction = new_direction
                 elif current_direction != new_direction:
                    Reversion_Amplitude_History[reversion_counter, simulation] = StimulusIntensity
                    reversion_counter += 1
                    # when there's a reversion, decrease the step size 
                    step_size = step_size*StepFactor
                    current_direction = new_direction
                    
            if reversion_counter == MaxReversions :
                break
                
        
    return Trial_Amplitude_History, Reversion_Amplitude_History

Trial_Amplitude_History, Reversion_Amplitude_History = SimulateTransformedStaircase()

# number of trials vs number of reversions plot  ?

# optimizing number of reversions + number of reversions to skip + plots 
    ## calculate the estimated threshold for each number of reversals or for each number of initial reversals skipped - what stimulus intensity would the staircase converge at if it had been stopped at x reversions or if x reversions were skipped? 
 
def CalculateReversionThresholds(Reversion_Amplitude_History):
    NumReversions, NumSimulations = Reversion_Amplitude_History.shape
    
    # estimated threshold for each number of reversals counted 
    reversions_counted_thresholds = np.full((NumSimulations, NumReversions), 0)
    for i in range(NumSimulations): 
        for r in range(NumReversions): 
            reversions_counted_thresholds[i, r] = np.mean(Reversion_Amplitude_History[:r+1, i])
    
    # estimated threshold for each number of initial reversals skipped
    reversions_skipped_thresholds = np.full((NumSimulations, NumReversions), 0)
    for i in range(NumSimulations): 
        for r in range(NumReversions): 
            if NumReversions - (r+1) > 0:
                reversions_skipped_thresholds[i, r] = (sum(Reversion_Amplitude_History[:, i]) - sum(Reversion_Amplitude_History[:r+1, i]))/(NumReversions - (r+1))
            else:
                reversions_skipped_thresholds[i, r] = np.sum(Reversion_Amplitude_History[r+1:, i])
    
    return reversions_counted_thresholds, reversions_skipped_thresholds, NumReversions

reversions_counted_thresholds, reversions_skipped_thresholds, NumReversions = CalculateReversionThresholds(Reversion_Amplitude_History)

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

# number of initial reversions skipped plot 
plt.errorbar(list(range(NumReversions-1)), rs_mean_threshold[:-1], yerr=rs_threshold_sd[:-1], fmt='o-', color='b', capsize=5, label='Mean Threshold Estimate ± SD')
plt.axhline(y=target_intensity, color='green', linestyle='--', label=f'Staircase Convergence Threshold: {target_intensity:.2f}')
plt.xlabel('Number of Initial Reversions Skipped')
plt.ylabel('Mean Threshold Estimate ± SD')
plt.xlim(0, NumReversions-2) 
plt.legend()
plt.show()
    
    
  
  
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

