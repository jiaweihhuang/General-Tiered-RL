import numpy as np

class Tabular_MDP_Class():
    def __init__(self, S, A, H, P, R) -> None:
        self.S = S
        self.A = A
        self.H = H
        self.P = P
        self.R = R

    def step(self, state, act, h):
        Ph_sa = self.P[h][state][act]
        sh_plus_1 = np.random.choice(self.S, p=Ph_sa)
        return sh_plus_1

    def sample_trajectory(self, policy):
        # uniform initial distribution
        sh = np.random.randint(self.S)  
 
        tau = []  
 
        for h in range(self.H):
            ah = policy[h][sh]
            sh_plus_1 = self.step(sh, ah, h)
            rh = self.R[h][sh][ah]
 
            tau.append((sh, ah, rh, sh_plus_1))
 
            sh = sh_plus_1
 
        return tau
 
    def policy_evaluation(self, policy):
        Vh_plus_1 = np.zeros([self.S, 1])
 
        for h in reversed(range(self.H)):
            Rh = self.R[h]
            Ph = self.P[h]
 
            pi_h = policy[h]
 
            Vh = Rh[np.arange(self.S), pi_h].reshape([-1, 1]) + np.matmul(Ph[np.arange(self.S), pi_h, :], Vh_plus_1)
 
            Vh_plus_1 = Vh
 
        V = np.mean(Vh_plus_1)
        return V

    def compute_policy_sub_opt_gap(self, policy):
        V_pi = self.policy_evaluation(policy)
        assert self.opt_value - V_pi >= 0.0, self.opt_value - V_pi
        return self.opt_value - V_pi

    def compute_optimal_policy(self):
        self.opt_pi = []
        self.opt_Q = []
        self.opt_V = []
        # gap(s,a)
        self.gap = []
 
        opt_Vh_plus_1 = np.zeros([self.S, 1])
 
        min_gap = self.H
 
        for h in reversed(range(self.H)):
            Rh = self.R[h]
            Ph = self.P[h]
 
            # check dimension
            opt_Q_h = Rh + np.matmul(Ph.reshape([-1, self.S]), opt_Vh_plus_1).reshape([self.S, self.A])
 
            opt_pi_h = np.argmax(opt_Q_h, axis=1)
            # opt_V_h = np.max(opt_Q_h, axis=1)
            opt_V_h = opt_Q_h[np.arange(self.S), opt_pi_h]
            assert len(opt_pi_h) == self.S
 
            self.opt_pi.insert(0, opt_pi_h)
            self.opt_Q.insert(0, opt_Q_h)
            self.opt_V.insert(0, opt_V_h)

            opt_Vh_plus_1 = opt_V_h

            # Compute Gap at Step h
            gap_h = opt_V_h.reshape([self.S, 1]) - opt_Q_h
 
            for g in gap_h.reshape([-1]):
                if g > 0:
                    min_gap = min(g, min_gap)
 
            self.gap.insert(0, gap_h)
       
        # uniform initial distribution
        self.opt_value = np.mean(self.opt_V[0])
 
        self.min_gap = min_gap
        print('Min Gap is ', self.min_gap)
 
    def compute_occupancy_for_opt_pi(self):
        self.d_opt_pi = []
        for h in range(self.H):
            if h == 0:
                d_opt_pi_h = np.ones([self.S, 1]) / self.S
            else:
                P_h_pi = self.P[h][np.arange(self.S), self.opt_pi[h]]
                d_opt_pi_h = P_h_pi.T @ last_d_opt_pi_h
            
            assert np.sum(d_opt_pi_h) == 1
            self.d_opt_pi.append(d_opt_pi_h)
            last_d_opt_pi_h = d_opt_pi_h



class Unique_OptPi_Tabular_MDP_Given_MinGap(Tabular_MDP_Class):
    def __init__(self, S=5, A=5, H=5, min_gap=0.1):
        super().__init__(S, A, H, [], [])

        self.reward_scale = 1.0 - min_gap
        self.reward_sub = min_gap
        self.assigned_min_gap = min_gap
 
        for h in range(self.H):
            # Ph[s,a,s'] = P(s'|s,a)
            Ph = np.random.randint(10, size=[S, A, S]) * 1.0
            Rh = np.random.rand(S, A) * self.reward_scale + self.reward_sub
            
            for s in range(S):
                for a in range(A):
                    Ph[s][a] = Ph[s][a] / np.sum(Ph[s][a])         # normalization
 
            self.R.append(Rh)
            self.P.append(Ph)
 
        self.R = np.array(self.R)
        self.P = np.array(self.P)
        self.compute_optimal_policy()
        self.compute_occupancy_for_opt_pi()

    def compute_optimal_policy(self):
        self.opt_pi = []
        self.opt_Q = []
        self.opt_V = []
        # gap(s,a)
        self.gap = []
 
        opt_Vh_plus_1 = np.zeros([self.S, 1])
 
        min_gap = self.H
 
        for h in reversed(range(self.H)):
            Rh = self.R[h]
            Ph = self.P[h]
 
            # check dimension
            opt_Q_h = Rh + np.matmul(Ph.reshape([-1, self.S]), opt_Vh_plus_1).reshape([self.S, self.A])
 
            opt_pi_h = np.argmax(opt_Q_h, axis=1)
            opt_V_h = opt_Q_h[np.arange(self.S), opt_pi_h]
            assert len(opt_pi_h) == self.S
 
            # for states with non-unique optimal action, we decrease the reward function for additional optimal actions by self.reward_sub
            for sh in range(self.S):
                for ah in range(self.A):
                    gap = np.abs(opt_V_h[sh] - opt_Q_h[sh][ah])
                    if gap < self.assigned_min_gap and opt_pi_h[sh] != ah:
                        # print('At h={}, sh={}, additional optimal action {}!={}'.format(h, sh, opt_pi_h[sh], ah))
                        # print(opt_Q_h[sh][ah], opt_V_h[sh])
                        self.R[h][sh][ah] -= (self.assigned_min_gap - gap)
                        opt_Q_h[sh][ah] -= (self.assigned_min_gap - gap)
 
            assert np.min(self.R) >= 0.0, 'negative reward'

            self.opt_pi.insert(0, opt_pi_h)
            self.opt_Q.insert(0, opt_Q_h)
            self.opt_V.insert(0, opt_V_h)

            opt_Vh_plus_1 = opt_V_h

            # Compute Gap at Step h
            gap_h = opt_V_h.reshape([self.S, 1]) - opt_Q_h
 
            for g in gap_h.reshape([-1]):
                if g > 0:
                    min_gap = min(g, min_gap)
 
            self.gap.insert(0, gap_h)
       
        # uniform initial distribution
        self.opt_value = np.mean(self.opt_V[0])
 
        self.opt_pi = np.array(self.opt_pi)
        self.opt_Q = np.array(self.opt_Q)
        self.opt_V = np.array(self.opt_V)

        self.min_gap = min_gap
        assert (self.min_gap - self.assigned_min_gap) > -1e-6, '{} < {}'.format(self.min_gap, self.assigned_min_gap)
        print('Min Gap is ', self.min_gap)
