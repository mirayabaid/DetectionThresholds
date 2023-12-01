# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:49:04 2023

@author: Somlab
"""

# staircase simulation function without calculating threshold 
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import random
from scipy.optimize import minimize


def SimulateTransformedStaircase(NumSimulations = 1000,
                                 PsychometricCurveMu = 50,
                                 PsychometricCurveSigma = 10,
                                 StimulusIntensityStart = 0, # start of stimulus intensity range 
                                 StimulusIntensityStop = 100, # end of stimulus intensity range 
                                 MaxNumTrials = 1000, 
                                 MaxReversions = 8,  
                                 NumAFC = 2, 
                                 Criterion = [3,1], 
                                 InitialStepSize = 10, 
                                 StepFactor = 0.725):
                                 
    # initialize variables for every simulation 
    
    # store stimulus intensity values by trial number for each simulation 
    Trial_Amplitude_History = np.full((MaxNumTrials, NumSimulations), 0)
    
    # store stimulus intensity values for each reversion number for each simulation 
    Reversion_Amplitude_History = np.full((MaxReversions, NumSimulations), 0)
    
    # store whether or not trial was detected for plotting the staircase later 
    Detection_History = np.full((MaxNumTrials, NumSimulations), 0)
    
    # store number of trials per simulation 
    Num_Trials_History = np.full((NumSimulations, 1), 0)
    
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
            
        Num_Trials_History[simulation, 0] = trial_counter
        
    return Trial_Amplitude_History, Reversion_Amplitude_History, Detection_History, Num_Trials_History, Reversion_Trials_History

Trial_Amplitude_History, Reversion_Amplitude_History, Detection_History, Num_Trials_History, Reversion_Trials_History = SimulateTransformedStaircase()

# for MLE 

def logistic_function(x, a, b):
    return 1 / (1 + np.exp(-(x - a) / b))

# x = stimulus amplitudes, either all amplitudes or just reversion amplitudes 
# a = mu 
# b = sigma 

def negative_log_likelihood(params, intensities, responses):
    a, b = params
    probabilities = logistic_function(intensities, a, b)
    likelihoods = responses*np.log(probabilities) + (1 - responses)*np.log(1 - probabilities)
    return -np.sum(likelihoods)


def CalculateTransformedStaircaseThreshold(Reversion_Amplitude_History, Detection_History, Reversion_Trials_History,
                                NumInitialReversionsSkipped = 0, use_MLE = False, MLE_Type = 'Reversion Amplitudes'):
   
    a, b = Reversion_Amplitude_History.shape 
    # a = number of reversions, b = number of simulations 
    
    # create array to store thresholds 
    Threshold_History = np.zeros(b)
    
    for simulation in range(b):
        if use_MLE == True:
            if MLE_Type == 'All Amplitudes':
                intensities = Trial_Amplitude_History[:, simulation]
                responses = Detection_History[:, simulation]
                initial_params = [np.mean(intensities), np.std(intensities)]
                result = minimize(negative_log_likelihood, initial_params, args=(intensities, responses))
                threshold_estimate = result.x[0]
                Threshold_History[simulation] = threshold_estimate
            elif MLE_Type == 'Reversion Amplitudes':
                # Use MLE on reversion amplitudes only 
                intensities = Reversion_Amplitude_History[:, simulation]
                # index responses for reversion trials 
                reversion_trials = Reversion_Trials_History[:, simulation]
                responses = Detection_History[reversion_trials, simulation]
                initial_params = [np.mean(intensities), np.std(intensities)]
                result = minimize(negative_log_likelihood, initial_params, args=(intensities, responses))
                threshold_estimate = result.x[0]
                Threshold_History[simulation] = threshold_estimate
        else:
        # Use mean of reversions for threshold estimation
            if NumInitialReversionsSkipped == 0:
                staircase_threshold = np.mean(Reversion_Amplitude_History[:, simulation])
                staircase_threshold = staircase_threshold.astype(int)
                Threshold_History[simulation] = staircase_threshold
            else: 
                staircase_threshold = np.mean(Reversion_Amplitude_History[NumInitialReversionsSkipped:, simulation])
                staircase_threshold = staircase_threshold.astype(int)
                Threshold_History[simulation] = staircase_threshold

    return Threshold_History
       
# # Compute staircase threshold 

Threshold_History = CalculateTransformedStaircaseThreshold(Reversion_Amplitude_History, Detection_History, Reversion_Trials_History,
                                NumInitialReversionsSkipped = 0, use_MLE = True, MLE_Type = 'Reversion Amplitudes')


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

    
    
    
    
    
    