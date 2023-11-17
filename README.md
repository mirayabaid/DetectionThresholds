# DetectionThresholds

This code is designed to simulate variations of the up-down staircase procedure used to detect sensory thresholds for psychophysical experiments. 
Many variations of the staircase procedure are used for detecting sensory thresholds, and here, we aim to find the optimal parameters for this method to maximize its efficiency & accuracy. 

Transformed staircase parameters: 
Reversion criteria (3-down-1-up)
Stop criteria (Number of trials/number of reversions)
Convergence rate 

Staircase efficiency, or the time it takes to detect a sensory threshold, is operationalized as the length of the procedure, i.e., the number of trials. 
The factors affecting the staircase's efficiency are a) the number of reversions after which the procedure is halted, b)
Staircase accuracy relates to the level of precision at which the staircase can reliably and accurately detect a sensory threshold (how close is the staircase's estimate to the ground truth?).
Staircase accuracy can be operationalized as the variabillity (standard deviation) in the simulated staircase's estimated thresholds. The standard deviation (error) is impacted by a) the number of simulations, which we keep constant, b) the number of reversions, c) the number of intial reversions that are skipped in calculating the threshold, d) the initial step size, e) the factor by which the step size is reduced after each reversion, f) using the maximum likelihood method, etc.  

