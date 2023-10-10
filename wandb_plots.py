import wandb
import pandas as pd
import seaborn as sns

api = wandb.Api()

entity = 'msc-marl'
project = 'SMAC'
sweep_ids = {'superagent': 'ejlbhiqx',
               'decentralised': 'h1d2oekp',
               'sharedweight':'z3rxoz5u',
               'vdn':'e6567810'}

sweep = api.sweep(entity+'/'+project+'/'+sweep_ids['vdn'])

print(sweep.runs[0].config)

