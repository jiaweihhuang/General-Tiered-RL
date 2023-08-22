import numpy as np
import pickle
from copy import deepcopy
 
class Logger_Class():
    def __init__(self, S, A, H, ME, MO_list, gap_min, algO_alpha, algP_alpha, log_path):
        self.log_path = log_path
        self.doc = {
            'S': S,
            'A': A,
            'H': H,
            'ME': ME,
            'MO_list': MO_list,
            'gap_min': gap_min,
            'algO_alpha': algO_alpha,
            'algP_alpha': algP_alpha,
            'results': []
        }
 
    def update_info(self, iter, R_algE, R_algO_list):
        self.doc['results'].append((iter, R_algE, deepcopy(R_algO_list)))
 
    def dump(self):
        with open(self.log_path, 'wb') as f:
            pickle.dump(self.doc, f)