#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
import argparse
sys.path.append('../')
import os
import torch 
import math
import time 
import pandas as pd
import numpy as np
import scipy.sparse as spa
import matplotlib.pyplot as plt
import exp_utils
import signal

def solve():
    x,mu,nu,forward_time = exp_utils.forward_solve(model_name,model,prob_type,if_eq, Q, q, G, h, A, b)
    print("Forward: ",forward_time)
    #------------------------Backward------------------------
    start_time = time.time()
    x.backward(torch.ones(x.shape))
    backward_time = time.time() - start_time
    return x,mu,nu,forward_time, backward_time
def handle_timeout(signum, frame):
    raise Exception("Time out") 


parser = argparse.ArgumentParser()
parser.add_argument('data_name', metavar='data_name', type=str)
parser.add_argument('model', nargs='+',metavar='model', type=str)
parser.add_argument('--nolarge',action='store_true',default=False, help="if not run problem with dim>10k")
parser.add_argument('--onlylarge',action='store_true',default=False, help="if just run problem with dim>10k")

args = parser.parse_args()
model_names = args.model
data_name = args.data_name
nolarge = args.nolarge
onlylarge = args.onlylarge


if nolarge and onlylarge:
    raise Exception("no large and onlylarge can not be True in both")


data_path = './data/' + data_name + '/'
if nolarge:
    file_path = './results/' + data_name + '_nolarge_' + '_'.join(model_names) + '.csv'

elif onlylarge:
    file_path = './results/' + data_name + '_onlylarge_' +  '_'.join(model_names) + '.csv'
else:
    file_path = './results/' + data_name + '_' + '_'.join(model_names) + '.csv'
data_names = ["randomqp_dense",
              "randomqp_sparse",
              "random_projection",
              "chain",
              "mm",
              "mpc"]



if data_name in data_names:
    file_names = os.listdir(data_path)
    if ".DS_Store" in file_names:
        file_names.remove(".DS_Store")
    
    # Just used for testing, will delete later
    # idx = np.random.randint(low=0,high=len(file_names),size=20)
    # file_names = file_names[idx]
    if data_name in ['randomqp_dense','randomqp_sparse','random_projection','chain']: 
        best_solvers = None
        file_names = sorted(file_names, key=lambda x: (int(x.split('_')[0][1:]), int(x.split('_')[-1].split('.')[0])))
    elif data_name in ['mm','mpc']:
        best_solvers = pd.read_csv('published_results/best_solver_' + data_name + '_qpbenchmark.csv')
        file_names = sorted(file_names, key=lambda x: x[0])
else:
    raise Exception("Wrong data_name, which should be one of " + str(data_names))



#%%
# Just used for testing, will change later
n_sim = 5

# ---------------------tolerances------------------------------
eps = 1e-6
eps_active = 1e-5
success_thr = 1
timelim = 800
# ---------------------Load models-----------------------------


nprob = len(file_names)

results = pd.DataFrame(columns=['Problem','dataset','dim','nIneq','nEq','solver','dQP solver','status','Forward time','Backward time','Total time','Eq residual','Ineq residual','Dual residual','Duality gap','Obj value'])
for sim_id in range(n_sim):
    print("--------------sim: " + str(sim_id) +" --------------")
    for model_id,model_name in enumerate(model_names):
        print("--------------model: " + model_name +" --------------")
        for i,file_name in enumerate(file_names):
            results.to_csv(file_path, header=True, index=False)
            
            

            prob_name = file_name[:-4]
            try:
                prob = np.load(data_path+prob_name+'.npz',allow_pickle=True)
            except:
                print(file_name + " not loaded")
                continue
            prob_type = 'dense'
            
            
            if data_name=="mpc":
                Q,q,G,h,A,b = prob['P'][()],prob['q'][()],prob['G'][()],prob['h'][()],prob['A'][()],prob['b'][()]
            else:
                Q,q,G,h,A,b = prob['Q'][()],prob['q'][()],prob['G'][()],prob['h'][()],prob['A'][()],prob['b'][()]
            
            
            if nolarge:
                if Q.shape[0]>1e4:
                    print("nolarge: Skip large problem: " + prob_name)
                    continue 
            elif onlylarge:
                if Q.shape[0]<=1e4:
                    print("onlylarge: Skip small problem: " + prob_name)
                    continue 
            
            
                
            if G is None:
                print(prob_name + ' do not have ineq cond, skipped')
                continue
            if spa.issparse(Q):
                prob_type = 'sparse'
            
            
            if_eq = True
            if model_name[:3]=='dQP' and prob_type=='sparse':
                dim = Q.shape[0]
                nIneq = G.shape[0]
                q = torch.tensor(q,requires_grad=True)
                h = torch.tensor(h,requires_grad=True)
                Q,G = Q.tocoo(),G.tocoo()
                __to_coo = lambda M: torch.sparse_coo_tensor(torch.stack([torch.tensor(M.row),torch.tensor(M.col)]),torch.tensor(M.data),M.shape,dtype=torch.float64,requires_grad=True)
                Q,G = __to_coo(Q),__to_coo(G)
                if A is not None:
                    nEq = A.shape[0]
                    A = A.tocoo()
                    A = __to_coo(A)
                    b = torch.tensor(b,requires_grad=True)
                    
                else:
                    if_eq = False
                    nEq = 0
                    
            else:
                if prob_type=="sparse":
                    Q,G = Q.todense(),G.todense()
                    if A is not None:
                        A = A.todense()
                
                Q = torch.tensor(Q,requires_grad=True).unsqueeze(0)
                q = torch.tensor(q,requires_grad=True).unsqueeze(0)
                G = torch.tensor(G,requires_grad=True).unsqueeze(0)
                h = torch.tensor(h,requires_grad=True).unsqueeze(0)
                nIneq = G.shape[1]
                dim = Q.shape[1]
                if A is not None:
                    A = torch.tensor(A,requires_grad=True).unsqueeze(0)
                    b = torch.tensor(b,requires_grad=True).unsqueeze(0)
                    nEq = A.shape[1]
                else:
                    if_eq = False
                    A,b = exp_utils.__Ab_conversion(model_name,Q,A,b)
                    nEq = 0
                
                
            
           
    
            
            
            
            
            
            print("-----------------------model: " + model_name + ' sim:' + str(sim_id) + " ---------------------")
            print("prob: " + file_name)
            print(prob_type + ' |Dim:' + str(dim) + ' |nIneq:' + str(nIneq) + ' |nEq:' + str(nEq))
   
            if model_name=='dQP_best' and (best_solvers is not None):
                filtered = best_solvers[best_solvers["problem"]==prob_name]
                best_solver = filtered["solver"].tolist()[0]
                print("Best solver: ",best_solver)
                model_name1 = "dQP_" + best_solver
                settings = filtered["settings"].tolist()[0]
                eps0 = eps
               
                model = exp_utils.load_model(model_name1,dim,nIneq,nEq,eps,eps_active,prob_type=prob_type)           
                eps = eps0
            else:
                model = exp_utils.load_model(model_name,dim,nIneq,nEq,eps,eps_active,prob_type=prob_type)           
            
            
            # Time limit
            signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(timelim)
            try:
                # mu: dual variables of eq cond
                # nu: ineq cond
                x,mu,nu,forward_time,backward_time = solve()

                status = 'Success'
                assert not ((x is None) or torch.isnan(x).any())

            except Exception as e:
                status = "Fail"
                print(model_name + " failed")
                print(repr(e))
                try:
                    if e.args[0]=="Time out":
                        status = "Time out"
                except:
                    status = "Fail"
                if model_name=='dQP_best' and (best_solvers is not None):
                    result = {'Problem': prob_name,'dataset': data_name, 'dim':dim,
                             'nIneq': nIneq, 'nEq':nEq, 
                             'solver': model_name, 
                             'dQP solver': best_solver,
                             'status': status,
                             }
                else:
                    result = {'Problem': prob_name,'dataset': data_name, 'dim':dim,
                             'nIneq': nIneq, 'nEq':nEq, 
                             'solver': model_name, 
                             'status': status,
                             }
                results.loc[nprob * (model_id-1) + i] = result
                if os.path.isfile(file_path):
                    results.to_csv(file_path, mode='a', header=False,index=False)
                else:
                    results.to_csv(file_path, header=True, index=False)
                continue

            finally:
                signal.alarm(0)
                
            
            total_time = forward_time + backward_time
            print("Backward: " + str(backward_time) + " | Total: "+str(total_time))

            
            x_copy = x.clone().detach()
            if len(x_copy.shape)==2:
                x_copy.unsqueeze_(-1)
            
            
            eq_res = exp_utils.eq_residual(x_copy,A,b,if_eq)
            ineq_res = exp_utils.ineq_residual(x_copy,G,h)
            obj_val = exp_utils.objective_value(x_copy, Q, q)
            
            if (ineq_res > success_thr):
                status = "Inaccurate: ineq res"
                print(status)

            if (eq_res is not None) and (eq_res > success_thr):
                status = "Inaccurate: eq res"
                print(status)
            
            if model_name=='OptNet':
                x.unsqueeze_(-1)
                if if_eq:
                    mu.unsqueeze_(-1) 
                nu.unsqueeze_(-1)
            if nu is not None:
                if (model_name=='SCQPTH') and (A is not None):
                        
                    G = torch.cat( (torch.cat((G,A),dim=-2),-A),dim=-2)
                    h = torch.cat( (torch.cat((h,b),dim=-1),-b),dim=-1)
                    
                    dual_res = exp_utils.dual_residual(x, nu, mu, Q, q, G, h, None, None,False)
                    dual_gap = exp_utils.duality_gap(x, nu, mu, Q, q, G, h, None, None,False)
                else:
                    dual_res = exp_utils.dual_residual(x, nu, mu, Q, q, G, h, A, b,if_eq)
                    dual_gap = exp_utils.duality_gap(x, nu, mu, Q, q, G, h, A, b,if_eq)
                if dual_res>success_thr :
                    status = "Inaccurate: dual res"
                    print(status)
                elif (dual_gap>success_thr):
                    status = "Inaccurate: dual gap"
                    print(status)

            else:
                dual_res = None
                dual_gap = None
            
                
            if model_name=='dQP_best' and (best_solvers is not None):
                result = {'Problem': prob_name,'dataset': data_name, 'dim':dim,
                         'nIneq': nIneq, 'nEq':nEq, 
                         'solver': model_name,
                         'dQP solver': best_solver,
                         'status': status,
                         'Forward time':forward_time, 'Backward time':backward_time , 'Total time':total_time ,
                         'Eq residual':eq_res , 'Ineq residual':ineq_res , 
                         'Dual residual':dual_res , 'Duality gap':dual_gap , 
                         'Obj value': obj_val}
            else:
                result = {'Problem': prob_name,'dataset': data_name, 'dim':dim,
                         'nIneq': nIneq, 'nEq':nEq, 
                         'solver': model_name, 
                         'status': status,
                         'Forward time':forward_time, 'Backward time':backward_time , 'Total time':total_time ,
                         'Eq residual':eq_res , 'Ineq residual':ineq_res , 
                         'Dual residual':dual_res , 'Duality gap':dual_gap , 
                         'Obj value': obj_val}
            if sim_id==0:
                results.loc[nprob * (model_id-1) + i] = result 
            else:
                for key in result:
                    if key in ['Forward time', 'Backward time', 'Total time', 
                               'Eq residual', 'Ineq residual', 'Dual residual', 'Duality gap', 'Obj value']: 
                        try:
                            results.loc[nprob * (model_id-1) + i, key] =  (results.loc[nprob * (model_id-1) + i, key] * sim_id + result[key]) / (sim_id+1)
                        except:
                            continue
            



results.to_csv(file_path, header=True, index=False)

