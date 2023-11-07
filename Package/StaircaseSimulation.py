#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:55:09 2023

@author: mirayabaid
"""
# Detection Threshold 2AFC Task: 3-Down-1-Up Transformed Staircase Simulation 

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
                                        
# find the stimulus amplitude at which the target staircase convergence target is reached 
# basically the amplitude at which p(correctly detecting) = 0.5 (point of subjective equality)

def GetStaircaseConvergenceIntensity(stimulus_range, pr_correct, target_probability): # values 
    
    # find value in pr_correct that is closest to the target, then index by that value to find a stimulus intensity value 
    # first find the absolute difference between pr_correct values and the target 
    # whichever value from the resulting array is the smallest, find the index of that value, then use to find the target intensity 
    
    target_index = np.abs(pr_correct - target_probability).argmin()
    target_intensity = stimulus_range[target_index]
    return target_intensity 
    
def PlotPsychometricFunctionTarget(stimulus_range, pr_correct, target_probability, target_intensity, Criterion, NumAFC):
    plt.plot(stimulus_range, pr_correct, color='red', label='Psychometric Function')
    plt.axhline(y=target_probability, color='blue', linestyle='--', label=f'Convergence Target (p): {target_probability:.2f}')
    plt.axvline(x=target_intensity, color='green', linestyle='--', label=f'Convergence Intensity: {target_intensity:.2f}')
    plt.xlabel('Stimulus Intensity')
    plt.ylabel('p(correct)')
    plt.title(f'{Criterion[0]}-Down-{Criterion[1]}-Up Staircase Convergence Target for {NumAFC}AFC Task')
    plt.legend()
    plt.show()

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
                                 StepFactor = 0.725,
                                 NumInitialReversionsSkipped = 0,
                                 ExamplePlot = True):
    
    # initialize variables for every simulation 
    
    # store stimulus intensity values by trial number for each simulation 
    Trial_Amplitude_History = np.full((MaxNumTrials, NumSimulations), 0)
    
    # store stimulus intensity values for each reversion number for each simulation 
    Reversion_Amplitude_History = np.full((MaxReversions, NumSimulations), 0)
    
    # store whether or not trial was detected for plotting the staircase later 
    Detection_History = np.full((MaxNumTrials, NumSimulations), 0)
    
    # store number of trials per simulation 
    Num_Trials_History = np.full((NumSimulations, 1), 0)
    
    # calculate thresholds per simulation + error 
    Threshold_History = np.full((NumSimulations, 1), 0)
    
    # store reversion trial numbers 
    Reversion_Trials_History = np.full((MaxReversions, NumSimulations), 0)
    
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
            
            Detection_History[trial, simulation] = response 
               
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
                    Reversion_Amplitude_History[reversion_counter, simulation] = Trial_Amplitude_History[trial, simulation] 
                    reversion_counter += 1
                    # when there's a reversion, decrease the step size 
                    step_size = step_size*StepFactor
                    current_direction = new_direction
                    Reversion_Trials_History[reversion_counter, simulation] = trial
                    
            if reversion_counter == MaxReversions :
                break
        
        if NumInitialReversionsSkipped == 0:
            staircase_threshold = np.mean(Reversion_Amplitude_History[:, simulation])
        else: staircase_threshold = np.mean(Reversion_Amplitude_History[:-NumInitialReversionsSkipped, simulation])
        
        Threshold_History[simulation, 0] = staircase_threshold
        Num_Trials_History[simulation, 0] = trial_counter
        
    return Trial_Amplitude_History, Reversion_Amplitude_History, Threshold_History, Detection_History, Num_Trials_History, Reversion_Trials_History

 # plot the staircase for any simulation number specified

Trial_Amplitude_History, Reversion_Amplitude_History, Threshold_History, Detection_History, Num_Trials_History, Reversion_Trials_History = SimulateTransformedStaircase(NumSimulations = 1000, 
                                 PsychometricCurveMu = 50,
                                 PsychometricCurveSigma = 10,
                                 StimulusIntensityStart = 0, # start of stimulus intensity range 
                                 StimulusIntensityStop = 100, # end of stimulus intensity range 
                                 MaxNumTrials = 1000, 
                                 MaxReversions = 10,  
                                 NumAFC = 2, 
                                 Criterion = (3,1), 
                                 InitialStepSize = 5, 
                                 StepFactor = 0.725,
                                 NumInitialReversionsSkipped = 0,
                                 ExamplePlot = True)

def PlotExampleStaircase(Trial_Amplitude_History,
                         Reversion_Amplitude_History,
                         Threshold_History,
                         Detection_History, 
                         Num_Trials_History,
                         SimulationNumber = 1):
   
    NumTrials = Num_Trials_History[SimulationNumber-1]
 
    for trial in range(int(NumTrials-1)):
        plt.step(trial, Trial_Amplitude_History[trial, SimulationNumber-1])
        if Detection_History[trial, SimulationNumber-1] == 0:
            plt.scatter(trial, Trial_Amplitude_History[trial, SimulationNumber-1], color = 'red')
        elif Detection_History[trial, SimulationNumber-1] == 1:
            plt.scatter(trial, Trial_Amplitude_History[trial, SimulationNumber-1], color = 'green')
        plt.axhline(y = Threshold_History[SimulationNumber-1], color = 'blue', linestyle = '--')
    return plt.show()
    
    
PlotExampleStaircase(Trial_Amplitude_History,
                         Reversion_Amplitude_History,
                         Threshold_History,
                         Detection_History, 
                         Num_Trials_History,
                         SimulationNumber = 5)
    
    
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


  
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

