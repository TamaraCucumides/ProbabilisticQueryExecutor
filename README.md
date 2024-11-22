# Probabilistic Query Executor

This repository contains the source code for PQE: Probabilistic Query Executor. 

# Step 1: Materialize the graph completion
In the folder data one must include one matrix per relation type, that contains the graph completion. Any link predictor can be used. 

To materialize the graph completion using `NBFNet` you can run this [script](https://github.com/TamaraCucumides/NBFNet/blob/master/script/save_predictions.py)


File number of each relation must be
{relation_number}.pt

# Step 2: Evaluate queries
* A sample of queries from the BetaE benchmark is included in the queries folder

Code can be run from `run.py` file. 
