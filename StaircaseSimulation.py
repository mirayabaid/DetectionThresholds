#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:55:09 2023

@author: mirayabaid
"""

# Transformed Staircase Simulation Revised 2 

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import random


def PlotPsychometricFunction(stim_start, stim_end, mu, sigma):
    stim_x = np.linspace(stim_start, stim_end, 100)
    pr_correct = norm.cdf(stim_x, loc = mu, scale = sigma) 
    plt.plot(stim_x, pr_correct, color = 'red')
    plt.xlabel('stimulus intensity')
    plt.ylabel('p(correct)')
    return plt.show()

def SimulateTransformedStaircase(NumSimulations = 1000, 
                                 PsychometricCurveMu = 50,
                                 PsychometricCurveSigma = 5,
                                 StimulusIntensityStart = 0, # start of stimulus intensity range 
                                 StimulusIntensityStop = 100, # end of stimulus intensity range 
                                 MaxNumTrials = 1000, 
                                 MaxReversions = 8,  
                                 NumAFC = 2, 
                                 Criterion = (3,1), 
                                 InitialStepSize = 10, 
                                 StepFactor = 0.5,
                                 NumReversionsSkipped = 0):
    
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
            
            if cumulative_score[0] >= Criterion[0]:
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

# number of trials plot 

# optimizing number of reversions + number of reversions to skip + plots 

## calculate the estimated threshold for each number of reversals or for each number of initial reversals skipped
 
def CalculateThresholds(Reversion_Amplitude_History):
    NumReversions, NumSimulations = Reversion_Amplitude_History.shape
    
    # estimated threshold for each number of reversals counted 
    reversals_counted_thresholds = np.full((NumSimulations, NumReversions), 0)
    for i in range(NumSimulations): 
        for r in range(NumReversions): 
            reversals_counted_thresholds[i, r] = np.mean(Reversion_Amplitude_History[:r+1, i])
    
    # estimated threshold for each number of reversals skipped
    reversals_skipped_thresholds = np.full((NumSimulations, NumReversions), 0)
    for i in range(NumSimulations): 
        for r in range(NumReversions): 
            if NumReversions - (r+1) > 0:
                reversals_skipped_thresholds[i, r] = (sum(Reversion_Amplitude_History[:, i]) - sum(Reversion_Amplitude_History[:r+1, i]))/(NumReversions - (r+1))
            else:
                reversals_skipped_thresholds[i, r] = np.sum(Reversion_Amplitude_History[r+1:, i])
    
    return reversals_counted_thresholds, reversals_skipped_thresholds





    
    
  
  
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

