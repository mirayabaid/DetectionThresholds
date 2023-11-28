#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:55:09 2023

@author: mirayabaid
"""

''' Detection Threshold 2AFC Task: 3-Down-1-Up Transformed Staircase Simulation '''

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import random
from scipy.optimize import minimize


def GetPsychometricFunction(PsychometricCurveMu = 50,
                                 PsychometricCurveSigma = 10,
                                 StimulusIntensityStart = 0, 
                                 StimulusIntensityStop = 100):
    stimulus_range = np.linspace(StimulusIntensityStart, StimulusIntensityStop+1, 100)
    pr_correct = norm.cdf(stimulus_range, loc = PsychometricCurveMu, scale = PsychometricCurveSigma) 
    return stimulus_range, pr_correct 

def GroundTruthPlot(stimulus_range, pr_correct):
    q1 = np.percentile(stimulus_range, 25, interpolation='midpoint') # only works if the stimulus range is spaced evenly 
    q2 = np.percentile(stimulus_range, 50, interpolation='midpoint')
    q3 = np.percentile(stimulus_range, 75, interpolation='midpoint')

    plt.plot(stimulus_range, pr_correct, color='black', label='Psychometric Function')
    plt.axhline(y=0.5, color='blue', linestyle='--', label='PSE (p): 0.5')
    plt.axvline(x=q2, color='red', linestyle='--', label=f'PSE/threshold: {q2:.1f}')
    plt.axvline(x=q1, color='pink', linestyle='--', label=f'q1: {q1:.1f}')
    plt.axvline(x=q3, color='pink', linestyle='--', label=f'q2: {q3:.1f}')
    plt.xlabel('Stimulus Intensity')
    plt.ylabel('p(correct)')
    plt.title('Ground Truth Plot')
    plt.legend()
    plt.show()
    
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
    
def StaircaseConvergencePlot(stimulus_range, pr_correct, target_probability, target_intensity, Criterion, NumAFC, PsychometricCurveMu):
    plt.plot(stimulus_range, pr_correct, color='black', label='Psychometric Function')
    plt.axhline(y=target_probability, color='blue', linestyle='--', label=f'Convergence Target (p): {target_probability:.2f}')
    plt.axvline(x=target_intensity, color='green', linestyle='--', label=f'Convergence Intensity: {target_intensity:.2f}')
    plt.axvline(x=PsychometricCurveMu, color='red', linestyle='--', label=f'Ground Truth: {PsychometricCurveMu:.2f}')
    plt.xlabel('Stimulus Intensity')
    plt.ylabel('p(correct)')
    plt.title(f'{Criterion[0]}-Down-{Criterion[1]}-Up Staircase Convergence Plot for {NumAFC}AFC Task')
    plt.legend()
    plt.show()

# for MLE 

def logistic_function(x, a, b):
    return 1 / (1 + np.exp(-(x - a) / b))

def negative_log_likelihood(params, intensities, responses):
    a, b = params
    probabilities = logistic_function(intensities, a, b)
    likelihoods = responses*np.log(probabilities) + (1 - responses)*np.log(1 - probabilities)
    return -np.sum(likelihoods)


def SimulateTransformedStaircase(NumSimulations = 1000,
                                 PsychometricCurveMu = 50,
                                 PsychometricCurveSigma = 15,
                                 StimulusIntensityStart = 0, # start of stimulus intensity range 
                                 StimulusIntensityStop = 100, # end of stimulus intensity range 
                                 MaxNumTrials = 1000, 
                                 MaxReversions = 8,  
                                 NumAFC = 2, 
                                 Criterion = [3,1], 
                                 InitialStepSize = 10, 
                                 StepFactor = 0.725,
                                 NumInitialReversionsSkipped = 0, 
                                 use_MLE = False): 
    
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
        StimulusIntensity = random.randrange(StimulusIntensityStart, StimulusIntensityStop+1)
        trial_counter = 0 
        reversion_counter = 0 
        reached_criterion = False 
        current_direction, new_direction = 0, 0
        cumulative_score = [0, 0] # correct, incorrect 
        step_size = InitialStepSize
        
        for trial in range(MaxNumTrials): 
            
            Trial_Amplitude_History[trial, simulation] = StimulusIntensity
            trial_counter += 1
        
            # new method 
            detection_draw = np.random.default_rng().normal(PsychometricCurveMu, (1/norm.ppf(.75))*PsychometricCurveSigma)
            afc_draw = np.random.default_rng().uniform()
            
            if detection_draw < StimulusIntensity or (NumAFC > 1 and afc_draw < 1/NumAFC):
                response = 1
            else:
                response = 0
            
            # store the response for this trial 
            Detection_History[trial, simulation] = response 
             
            if response == 1: 
                cumulative_score[0] += 1 # count correct responses 
            elif response == 0:
                cumulative_score[1] += 1 # count incorrect responses 
            
            # check if criterion is reached  
            if cumulative_score[0] >= Criterion[0]: # if there are 3 correct responses, stimulus intensity for the next trial decreases  
                new_direction = -1 # direction = decreasing 
                reached_criterion = True
            elif cumulative_score[1] >= Criterion[1]: # if one wrong, stimulus intensity for the next trial increases
                new_direction = +1
                reached_criterion = True
            
            # check if 3-correct or 1-incorrect is matched 
            if reached_criterion == True:
                 if current_direction == 0: 
                    current_direction = new_direction
                 elif new_direction != current_direction and current_direction != 0: 
                     step_size = step_size*StepFactor 
                     if step_size < 2:
                         step_size = 2
                     current_direction = new_direction
                     # store trial information 
                     Reversion_Amplitude_History[reversion_counter, simulation] = Trial_Amplitude_History[trial, simulation] 
                     Reversion_Trials_History[reversion_counter, simulation] = trial
                     reversion_counter += 1
                # change next stimulus amplitude  
                 if new_direction == +1:
                    StimulusIntensity = min(StimulusIntensityStop, StimulusIntensity + step_size)
                 elif new_direction == -1: 
                     StimulusIntensity = max(StimulusIntensityStart, StimulusIntensity - step_size)
                     
                 StimulusIntensity = np.round(StimulusIntensity / 2) * 2
                     
                 reached_criterion = False
                 cumulative_score = [0,0]
                    
            if reversion_counter == MaxReversions:
                break
        
        # estimating the threshold 
        if use_MLE:
            # Use MLE for threshold estimation
            intensities = Trial_Amplitude_History[:, simulation]
            responses = Detection_History[:, simulation]
            initial_params = [np.mean(intensities), np.std(intensities)]
            result = minimize(negative_log_likelihood, initial_params, args=(intensities, responses))
            threshold_estimate = result.x[0]
            Threshold_History[simulation, 0] = threshold_estimate
        else:
            # Use mean of reversions for threshold estimation
            if NumInitialReversionsSkipped == 0:
                staircase_threshold = np.mean(Reversion_Amplitude_History[:, simulation])
            else: 
                staircase_threshold = np.mean(Reversion_Amplitude_History[NumInitialReversionsSkipped:, simulation])
            Threshold_History[simulation, 0] = staircase_threshold
            
        Num_Trials_History[simulation, 0] = trial_counter
        
    return Trial_Amplitude_History, Reversion_Amplitude_History, Threshold_History, Detection_History, Num_Trials_History, Reversion_Trials_History
            
            
# plot the staircase for any simulation number specified
def PlotExampleStaircase(Trial_Amplitude_History,
                         Reversion_Amplitude_History,
                         Threshold_History,
                         Detection_History, 
                         Num_Trials_History,
                         Reversion_Trials_History,
                         SimulationNumber = 1):
   
    NumTrials = Num_Trials_History[SimulationNumber-1]
    
    for trial in range(int(NumTrials)):
        plt.step(trial, Trial_Amplitude_History[trial, SimulationNumber-1])
        if Detection_History[trial, SimulationNumber-1] == 0:
            plt.scatter(trial, Trial_Amplitude_History[trial, SimulationNumber-1], color = 'red')
        elif Detection_History[trial, SimulationNumber-1] == 1:
            plt.scatter(trial, Trial_Amplitude_History[trial, SimulationNumber-1], color = 'green')
        for trial in Reversion_Trials_History[:, SimulationNumber-1]:
            plt.scatter(trial, Trial_Amplitude_History[trial, SimulationNumber-1], facecolors = "none", edgecolors = "black")    
    
    plt.axhline(y = Threshold_History[SimulationNumber-1], color = 'blue', linestyle = '--', label = f'Estimated Threshold = {Threshold_History[SimulationNumber-1]}')
    plt.legend()
    return plt.show()

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


# Calculating the threshold using MLE 
  
# MLE would involve fitting a logistic function to the sequence of stimulus intensities and responses (correct/incorrect detections), 
# and then estimating the threshold as the stimulus intensity at which the probability of a correct response is 50%.
    
    
# define a logistic function
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

