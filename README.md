# Probabilistic Query Executor

This repository contains the source code for PQE: Probabilistic Query Executor a method for Complex Query answering inspired by *probabilistic databases*. 

# Step 1: Materialize the graph completion
In the folder data one must include one matrix per relation type, that contains the graph completion. Any link predictor can be used. 

To materialize the graph completion using `NBFNet` you can run this [script](https://github.com/TamaraCucumides/NBFNet/blob/master/script/save_predictions.py) (after training NBFNet, code for dataset FB15k-237)


File number of each relation must be
{relation_number}.pt

# Step 2: Evaluate queries
* A sample of queries from the BetaE benchmark is included in the queries folder. Dataset is FB15k-237. 

Code can be run from `run.py` file. 
