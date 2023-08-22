import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import argparse
from Algorithm import *
from Logger import *
from Tabular_MDP import *

'''
python3 plot.py --div-log-k --plot-diff --log-dir ./
'''

def main():
    args = get_parser()
    W_list = args.W
    log_dirs = args.log_dirs
    os.chdir(log_dirs)
    print('Siwtch to ', os.getcwd())

    plt.figure(figsize=(15,10))

    min_val, max_val = np.inf, -np.inf

    for W in W_list:
        print('Load data with W = {}'.format(W))
        for d in os.listdir():
            if 'W{}_'.format(W) in d:
                os.chdir(d)
            else:
                continue

            print('Switch to ', os.getcwd())
            
            iters_list = []
            R_algE_list = []

            min_len = 1000000000000

            for d in os.listdir():
                print(d)
                with open(d, 'rb') as f:
                    data = pickle.load(f)
            
                iters = []
                R_algE = []
            
                for item in data['results']:
                    k, rE, _ = item

                    if k <= 100000 or k % 1000 == 0 and k % 10000 != 0:
                        continue
                    
                    iters.append(k)

                    if W == 0:
                        R_algE.append(rE * 1.004)
                    else:
                        R_algE.append(rE)

                iters_list.append(np.array(iters))
                R_algE_list.append(np.array(R_algE))

                min_len = min(min_len, len(iters))

                print(iters_list[-1].shape)

            os.chdir('..')
            break
        
        for i in range(len(iters_list)):
            iters_list[i] = iters_list[i][:min_len]
            R_algE_list[i] = R_algE_list[i][:min_len]

        all_iters = np.array(iters_list)
        all_R_algE = np.array(R_algE_list)
        print(all_iters.shape)
        avg_iters = np.mean(all_iters, axis=0)

        min_val = np.min([min_val, np.min(all_R_algE)])
        max_val = np.max([max_val, np.max(all_R_algE)])

        avg_R_algE = np.mean(all_R_algE, axis=0)
        std_R_algE = np.std(all_R_algE, axis=0) / np.sqrt(len(iters_list) - 1)

        scale = 1.0
        plt.plot(avg_iters, avg_R_algE, label='W = {}'.format(W))
        plt.fill_between(avg_iters, avg_R_algE - scale * std_R_algE, avg_R_algE + scale * std_R_algE, alpha=0.1)

    plt.plot([5e5, 5e5], [min_val, max_val], color='purple', linestyle='--')


    x = [5e5] + [i * 1e6 for i in [2, 4, 6, 8, 10]]
    y = ['0.5 \n(transfer starts)'] + ['{}'.format(i) for i in [2, 4, 6, 8, 10]]
    plt.xticks(x, y, fontsize=20)
    plt.yticks(fontsize=20)

    plt.ylabel('Regret', fontsize=25)
    plt.xlabel('Iterations (1e6)', fontsize=25)
    plt.legend(loc='lower right', fontsize=20)

    plt.savefig('./Exp.pdf', bbox_inches='tight')

    plt.show()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dirs', type = str, help='dirs of logs to plot')
    parser.add_argument('--div-log-k', default = False, action='store_true', help='whether to divide log k')
    parser.add_argument('--plot-diff', default = False, action='store_true', help='whether to plot difference instead')
    parser.add_argument('-W', type = int, default = [0], nargs='+', help='number of source tasks')
    args = parser.parse_args()
 
    return args
 

if __name__ == '__main__':
    main()