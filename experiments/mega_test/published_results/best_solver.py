#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Select the most efficient forward solver satisfying benchmark tolerance
import pandas as pd 
file_path = 'mpc_qpbenchmark'
result = pd.read_csv(file_path + '.csv')
result = result[result["settings"]=="mid_accuracy"]
tol = 1e-6
best_solvers = []

for problem, group in result.groupby("problem"):
    filtered_group = group[(group['primal_residual'] < tol)  & (group['dual_residual'] < tol) & (group['duality_gap'] < tol)]
    

    if not filtered_group.empty:
        # Find the row with the minimum 'runtime'
        best_solver = filtered_group.loc[filtered_group['runtime'].idxmin()]
        
        # Append the best solver's details
        best_solvers.append(best_solver)
    else:
        tol1 = 1e-4
        filtered_group = group[(group['primal_residual'] < tol1)  & (group['dual_residual'] < tol1) & (group['duality_gap'] < tol1)]
        if not filtered_group.empty:
            # Find the row with the minimum 'runtime'
            best_solver = filtered_group.loc[filtered_group['runtime'].idxmin()]
            
            # Append the best solver's details
            best_solvers.append(best_solver)
        else:
            if problem=="BOYD1":
                filtered_group = group[(group["solver"]=="gurobi") & (group["settings"]=="mid_accuracy")]
                best_solver = filtered_group.iloc[0]
                best_solvers.append(best_solver)
            if problem=="ICULS1":
                filtered_group = group[(group["solver"]=="quadprog") & (group["settings"]=="mid_accuracy")]
                best_solver = filtered_group.iloc[0]
                best_solvers.append(best_solver)
            else:  
                print(problem + " no best solver")

        
    

# Convert the list of best solvers to a DataFrame
best_solvers_df = pd.DataFrame(best_solvers)
best_solvers_df.to_csv("best_solver_" + file_path +'.csv',index=False)
# Display the result
print(best_solvers_df)
