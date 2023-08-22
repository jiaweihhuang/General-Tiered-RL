import numpy as np
import argparse
import os
from Logger import Logger_Class
import random
from Tabular_MDP import Tabular_MDP_Class, Unique_OptPi_Tabular_MDP_Given_MinGap
 

INT_MAX = np.iinfo(np.int64).max

class RandomPerturbation(Tabular_MDP_Class):
    def __init__(self, MDP, permut_prob=1.0, min_gap=0.0, scale=0.0):
        super().__init__(MDP.S, MDP.A, MDP.H, MDP.P.copy(), MDP.R.copy())

        self.permut_prob = permut_prob
        self.scale = scale
        self.min_gap = min_gap

        self.random_action_permutation()
        self.compute_optimal_policy()
        self.compute_occupancy_for_opt_pi()

    def random_action_permutation(self):
        # randomly perturb the transition matrix
        for h in range(self.H):
            for sh in range(self.S):
                if np.random.rand() < self.permut_prob:
                    permut_sequence = np.random.permutation(self.A)
                    self.P[h][sh] = self.P[h][sh][permut_sequence]
                    self.R[h][sh] = self.R[h][sh][permut_sequence]

        self.R = np.array(self.R)
        self.P = np.array(self.P)


    def random_reward_revision(self, eps):
        # revise the reward to be R + noise * eps
        for h in range(self.H):
            for sh in range(self.S):
                assert self.R[h][sh].shape == (self.A, )
                self.R[h][sh] = self.R[h][sh] + np.random.rand(self.A) * eps * self.scale

class EstimatedModel():
    def __init__(self, S, A, H, R, alpha, tier, M=1.0, W=None, lam=0.1, true_opt=None, delay_transfer=-1):
        self.S = S
        self.A = A
        self.H = H
        self.R = R
        self.alpha = alpha
        self.M = M
        self.tier = tier
        self.lam = lam
        self.W = W if W else 1

        self.counter = 0
        self.delay_transfer = delay_transfer

        self.N_sas = []
        self.N_sa = []
        self.hatP = []
        self.max_N_sa = []

        self.true_opt = true_opt

        self.total_trust_num = 0
        self.total_trust_fail_num = 0
        self.total_trust_list = []

        self.mis_trust_matrix = np.zeros([self.H, self.S])
        self.acc_trust_matrix = np.zeros([self.H, self.S])

        self.counter = 0


        for h in range(self.H):
            self.N_sas.append(np.zeros([self.S, self.A, self.S]))
            self.N_sa.append(np.zeros([self.S, self.A]))
            self.max_N_sa.append(np.zeros([self.S]))
            self.hatP.append(np.ones([self.S, self.A, self.S]) / self.S)

        if self.tier == 'high':
            # initialize trusted task by -1, i.e. no trust task
            self.trust = np.ones(shape=[H,S], dtype=np.int32) * -1
            self.trust_action = np.ones(shape=[H,S], dtype=np.int32) * -1
            assert self.W is not None

            self.ME_UCB = np.ones(shape=[H, S]) * -1.0
            self.MO_LCB = np.ones(shape=[H, S]) * -1.0

            self.num_feasible = np.zeros(shape=[H,S], dtype=np.int32)

    def update(self, tau):
        self.counter += 1
        for h in range(self.H):
            sh, ah, rh, sh_plus_1 = tau[h]
            self.N_sas[h][sh][ah][sh_plus_1] += 1.0
            self.N_sa[h][sh][ah] += 1.0
 
            # update hat P
            self.hatP[h][sh][ah] = self.N_sas[h][sh][ah][:] / self.N_sa[h][sh][ah]
            self.max_N_sa[h][sh] = np.max(self.N_sa[h][sh])


    def compute_exploiting_action(self, h, sh, VO_LCB_array, piOLCB_array, QE_UCB, NO_feasible):
        QE_UCB_piO = np.take(QE_UCB[sh], piOLCB_array)
        feasible = np.logical_and(
            VO_LCB_array <= QE_UCB_piO,
            NO_feasible,
        )

        feasible_w = np.argwhere(feasible == True).squeeze().tolist()
        if type(feasible_w) is int:
            feasible_w = [feasible_w]

        self.num_feasible[h][sh] = len(feasible_w)

        if len(feasible_w) == 0:
            self.trust[h, sh] = -1
            self.trust_action[h, sh] = -1

            self.MO_LCB[h, sh] = -1
            return False, -1

        if self.trust[h, sh] == -1:
            wk = feasible_w[0]
        else:
            if int(self.trust[h, sh]) in feasible_w:
                wk = self.trust[h, sh]
            else:
                last_action = self.trust_action[h, sh]
                feasible_w_with_same_action = np.argwhere(piOLCB_array[feasible] == last_action).squeeze().tolist()
                if type(feasible_w_with_same_action) is int:
                    feasible_w_with_same_action = [feasible_w_with_same_action]
                if len(feasible_w_with_same_action) == 0:
                    wk = feasible_w[0]
                else:
                    wk = feasible_w_with_same_action[0]

        self.trust[h, sh] = wk
        self.trust_action[h, sh] = piOLCB_array[wk]

        self.ME_UCB[h, sh] = QE_UCB_piO[wk]
        self.MO_LCB[h, sh] = VO_LCB_array[wk]

        return True, self.trust_action[h, sh]


    def exploit_MO(self, h, k, org_pi, QE_UCB, info_from_MO):
        for sh in range(self.S):
            VO_LCB_array = np.array(
                [info_from_MO['VO_LCB_list'][w][h][sh] for w in range(self.W)]
            ).reshape([-1])
            piOLCB_array = np.array(
                [info_from_MO['piO_LCB_list'][w][h][sh] for w in range(self.W)]
            ).reshape([-1])
            max_NO_sa_array = np.array(
                [info_from_MO['max_NO_sa_list'][w][h][sh] for w in range(self.W)]
            ).reshape([-1])
            NO_feasible = max_NO_sa_array > self.lam / 3.0 * k

            ### Return value:
            # exploit_flag: if this state will exploit information
            # exploit_act: which action is recommended
            exploit_flag, exploit_act = self.compute_exploiting_action(h, sh, VO_LCB_array, piOLCB_array, QE_UCB, NO_feasible)

            if exploit_flag:
                if org_pi[sh] != exploit_act:
                    self.is_transfer_diff[h][sh] = 1.0
                    org_pi[sh] = exploit_act

                self.total_trust_num += 1
                if self.true_opt[h][sh] != exploit_act:
                    self.total_trust_list.append(0)
                    self.mis_trust_matrix[h][sh] += 1
                else:
                    self.total_trust_list.append(1)
                    self.acc_trust_matrix[h][sh] += 1

        return org_pi


    def compute_policy(self, k, info_from_MO={}):
        Vh_plus_1_LCB = np.zeros([self.S, 1])
        self.Vh_LCB_list = []
        self.pi_LCB_list = []

        self.Vh_UCB_list = []
        self.Qh_UCB_list = []

        # union bound over all source tasks
        if self.tier == 'high':
            self.delta = 1.0 / k
        else:
            self.delta = 1.0 / k / self.W       # for the fixed confidence level
 
        self.policy = []
        self.is_transfer_diff = np.zeros([self.H, self.S])
 
        Vh_plus_1 = np.zeros([self.S, 1])
        Vh_plus_1_sub = np.zeros([self.S, 1])
 
        for h in reversed(range(self.H)):
            bonus = self.compute_bonus(Vh_plus_1, Vh_plus_1_sub, h)
 
            Q_h = self.R[h] + np.matmul(self.hatP[h].reshape([-1, self.S]), Vh_plus_1).reshape([self.S, self.A]) + self.alpha * bonus
            # h starts with 0, so we use H - h
            Q_h = np.clip(Q_h, a_min=0.0, a_max=self.H-h)
 
            greedy_pi_h = np.argmax(Q_h, axis=1)

            # if ME, leverage MO_list
            if self.tier == 'high' and self.counter >= self.delay_transfer:
                pi_h = self.exploit_MO(h, k, greedy_pi_h, Q_h, info_from_MO)
            else:
                pi_h = greedy_pi_h

            # compute LCB
            Qh_LCB = self.R[h] + np.matmul(self.hatP[h].reshape([-1, self.S]), Vh_plus_1_LCB).reshape([self.S, self.A]) - self.alpha * bonus
            # h starts with 0, so we use H - h
            Qh_LCB = np.clip(Qh_LCB, a_min=0.0, a_max=self.H-h)
            Vh_LCB = np.max(Qh_LCB, axis=1).reshape([-1, 1])
            pi_LCB = np.argmax(Qh_LCB, axis=1)
            Vh_plus_1_LCB = Vh_LCB

            self.Vh_LCB_list.insert(0, Vh_LCB)
            self.pi_LCB_list.insert(0, pi_LCB)
            
            V_h_sub = Vh_LCB
            V_h = np.max(Q_h, axis=1).reshape([-1, 1])
            if self.tier == 'high' and self.counter >= self.delay_transfer:
                Q_h_pi_h = Q_h[np.arange(self.S), pi_h].reshape([-1, 1])
                V_h = Q_h_pi_h + (V_h - V_h_sub) / self.H / np.e * self.is_transfer_diff[h][:].reshape([-1, 1])

            Vh_plus_1_sub = V_h_sub
            Vh_plus_1 = V_h
 
            self.policy.insert(0, pi_h)
            
            self.Vh_UCB_list.insert(0, V_h) 
            self.Qh_UCB_list.insert(0, Q_h)

        return self.policy
 
    # to avoid divide by 0, we divide by max(1e-9, n) or max(1e-9, n-1)
    def compute_bonus(self, Vh_plus_1, Vh_plus_1_sub, h):
        L = np.sqrt(2 * np.log(10 * self.M ** 2 * np.clip(self.N_sa[h], a_min=1.0, a_max=INT_MAX) / self.delta))
 
        N_sa_m1_clip = np.clip(self.N_sa[h] - 1.0, a_min=1.0, a_max=INT_MAX)
        N_sa_clip = np.clip(self.N_sa[h], a_min=1.0, a_max=INT_MAX)
 
        P = self.hatP[h].reshape([-1, self.S])
        assert len(Vh_plus_1.shape) == 2
        Var_Vh_plus_1 = np.clip(np.matmul(P, Vh_plus_1 ** 2) - np.matmul(P, Vh_plus_1) ** 2, a_min=1e-4, a_max=INT_MAX)
        Var_Vh_plus_1 = Var_Vh_plus_1.reshape([self.S, self.A])
 
        Diff_Square = np.matmul(P, np.square(Vh_plus_1 - Vh_plus_1_sub)).reshape([self.S, self.A])
 
        bprob = np.sqrt(2. * Var_Vh_plus_1 * L / N_sa_clip) + 8.0 * self.H * L / N_sa_m1_clip / 3.0 + np.sqrt(2. * L * Diff_Square / N_sa_clip)
        bprob = np.clip(bprob, a_min=0.0, a_max=self.H - h)

        bstr = np.sqrt(Diff_Square) * np.sqrt(self.S * L / N_sa_clip) + 8.0 / 3.0 * self.S * self.H * L / N_sa_clip
        bstr = np.clip(bstr, a_min=0.0, a_max=self.H - h)

        return np.clip(bprob + bstr, a_min=0.0, a_max=self.H - h)

class Algorithm():
    def __init__(self, S, A, H, ME, MO_list, alpha, M=1.0, lam=0.1, delay_transfer=-1):
        self.S = S
        self.A = A
        self.H = H

        self.ME = ME
        self.MO_list = MO_list
        self.W = len(MO_list)

        self.ME_EM = EstimatedModel(S, A, H, ME.R, alpha=alpha, tier='high', 
                                    M=M, W=self.W, lam=lam, 
                                    true_opt = self.ME.opt_pi, delay_transfer = delay_transfer)
        self.MO_EM_list = []
        for MO in MO_list:
            self.MO_EM_list.append(
                EstimatedModel(S, A, H, MO.R, alpha=alpha, tier='low', M=M, W=self.W, true_opt = MO.opt_pi)
            )


    def learn_for_one_step(self, k):
        VO_LCB_list = []
        piO_LCB_list = []
        max_NO_sa_list = []
        regretO_k_list = []
        # learning of MO_list
        piO_w_list = []
        for w in range(self.W):
            MO_EM_w = self.MO_EM_list[w]
            MO_w = self.MO_list[w]

            piO_w = MO_EM_w.compute_policy(k)
            tauO_w = MO_w.sample_trajectory(piO_w)
            MO_EM_w.update(tauO_w)

            VO_LCB_w, piO_LCB_w = MO_EM_w.Vh_LCB_list, MO_EM_w.pi_LCB_list
            max_NO_sa_w = MO_EM_w.max_N_sa
            VO_LCB_list.append(VO_LCB_w)
            piO_LCB_list.append(piO_LCB_w)
            max_NO_sa_list.append(max_NO_sa_w)

            regretO_k_list.append(MO_w.compute_policy_sub_opt_gap(piO_w))

            piO_w_list.append(piO_w)

        # learning of ME
        info_from_MO = {
            'VO_LCB_list': VO_LCB_list, 
            'piO_LCB_list': piO_LCB_list, 
            'max_NO_sa_list': max_NO_sa_list,
        }
        piE = self.ME_EM.compute_policy(k, info_from_MO)
        tauE = self.ME.sample_trajectory(piE)
        self.ME_EM.update(tauE)

        regretE_k = self.ME.compute_policy_sub_opt_gap(piE)

        return regretO_k_list, regretE_k
 
# compute transferable states (asymptotically)
def compute_transferable_states(ME, MO, eps=0.0, lam=0.0):
    S, A, H = ME.S, ME.A, ME.H
    transfer_states = np.zeros(shape=[H, S], dtype=bool)
    for h in range(H):
        for sh in range(S):
            if ME.opt_V[h][sh] + eps >= MO.opt_V[h][sh] and ME.opt_pi[h][sh] == MO.opt_pi[h][sh] and MO.d_opt_pi[h][sh] > lam / 3.0:
                transfer_states[h][sh] = True
    return transfer_states

def main():
    args = get_parser()
 
    random.seed(args.model_seed)
    np.random.seed(args.model_seed)
   
    alpha = args.alpha

    S, A, H = args.S, args.A, args.H
    W = args.W
    assert W <= 100
 
    ME = Unique_OptPi_Tabular_MDP_Given_MinGap(S=S, A=A, H=H, min_gap=args.min_gap)
    MO_list = []
    trans_s_w_all = []
    trans_s_w_ensemble = np.zeros(shape=[H, S], dtype=bool)
    
    for w in range(W):
        MO_w = RandomPerturbation(ME, permut_prob=1.0, min_gap=ME.min_gap, scale=0.0)
        MO_list.append(MO_w)

    MO_list = MO_list[:W]
    for MO_w in MO_list:
        trans_s_w = compute_transferable_states(ME, MO_w, eps=0.0, lam=args.lam)
        trans_s_w_all.append(trans_s_w)
        trans_s_w_ensemble = np.logical_or(trans_s_w_ensemble, trans_s_w)

    for seed in args.seed:
        random.seed(seed)
        np.random.seed(seed)
 
        Alg = Algorithm(
                    S=S,
                    A=A,
                    H=H,
                    ME=ME, 
                    MO_list=MO_list,
                    alpha=alpha,
                    lam=args.lam,
                    delay_transfer=args.delay_transfer,
                )
 
        R_algE = 0.0
        R_algO_list = [0.0 for w in range(W)]
        log_path = 'log/S{}_A{}_H{}_W{}_lam{}_minGap{}_model_seed{}_alpha{}'.format(args.S, args.A, args.H, args.W, 
                args.lam, ME.min_gap, args.model_seed, args.alpha)

        if args.delay_transfer > 0:
            log_path += '_dt{}'.format(args.delay_transfer)

        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file = '{}/seed{}'.format(log_path, seed)
 
        logger = Logger_Class(args.S, args.A, args.H,
                    ME, MO_list, ME.min_gap,
                    algO_alpha=args.alpha,
                    algP_alpha=-args.alpha,
                    log_path=log_file)
 
        for k in range(2, args.K):
            regretO_k_list, regretE_k = Alg.learn_for_one_step(k)
            R_algE += regretE_k
            for w in range(W):
                R_algO_list[w] += regretO_k_list[w]

            if k % 10000 == 0:
                print('Iter = ', k)
                print('R_algE: {}; R_algE / log k: {}'.format(R_algE, R_algE / np.log(k)))
                print('R_algO: {}; R_algO / log k: {}'.format(R_algO_list, [R_ / np.log(k) for R_ in R_algO_list]))
                print('Min Gap ', ME.min_gap)
                print('trans_s_w_ensemble \n', trans_s_w_ensemble)
                
                if Alg.ME_EM.total_trust_num > 0:
                    print('Number of Trust', Alg.ME_EM.total_trust_num, 'Accurate Rate', np.mean(Alg.ME_EM.total_trust_list), 'Accurate Rate (Recent 100)', np.mean(Alg.ME_EM.total_trust_list[-100:]))
                    print(Alg.ME_EM.mis_trust_matrix)

                print('Save to ', log_path)
                print('\n\n')
 
                logger.update_info(k, R_algE, R_algO_list)
                logger.dump()
 
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-K', type = int, default = 10000000, help='training iteration')
    parser.add_argument('--alpha', type = float, default = 0.5, help='coefficient of bonus term')
    parser.add_argument('--lam', type = float, default = 0.3, help='lambda')
    parser.add_argument('--min-gap', type = float, default = 0.1, help='min gap')
    parser.add_argument('-S', type = int, default = 3, help='number of states')
    parser.add_argument('-A', type = int, default = 3, help='number of actions')
    parser.add_argument('-H', type = int, default = 5, help='H')
    parser.add_argument('-W', type = int, default = 3, help='number of source tasks')
    parser.add_argument('--seed', type = int, default = [0], nargs='+', help='seed')
    parser.add_argument('--model-seed', type = int, default = 1000, help='seed')
    parser.add_argument('--delay-transfer', type = int, default = 0, help='seed')

 
    args = parser.parse_args()
 
    return args
 
if __name__ == '__main__':
    main()