# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:02:51 2023

@author: Somlab
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import random

def PlotStaircaseProcedure(PsychometricCurveMu = 50,
                                 PsychometricCurveSigma = 10,
                                 StimulusIntensityStart = 0, 
                                 StimulusIntensityStop = 100, 
                                 NumAFC = 2, 
                                 InitialStepSize = 5, 
                                 StepFactor = 0.5, 
                                 MaxNumTrials = 100, 
                                 MaxNumReversions = 5,
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
    statement = print(f'Estimated Threshold: {staircase_threshold:.2f}')
    
    return plt.show(), statement

PlotStaircaseProcedure()