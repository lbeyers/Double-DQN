# MSc Project
For the fully centralised agent: the file to run is smac_doublerun.py. 
The file gymwrapper.py wraps the SMAC environment to work the way gymnasium environments do, so that it's compatible with the run file.
From there, you'll see that the superagent comes from agentclasses.py, the intermittent evaluations (where epsilon is 0) use evals.py and the replay buffer comes from the replaybuffers.py.
