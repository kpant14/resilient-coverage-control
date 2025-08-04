import numpy as np
import casadi
from bearing_rigidity_utils import create_adjacency_matrix, get_neighbors_bearing, get_neighbors_state
from anim_utils import animate

def dm_to_array(dm):
        return np.array(dm.full())

class BearingMaintenanceMPC:
    def __init__(self, init_state, n_neighbors, dt ,N, v_lim, omega_lim,  
                 Q, R, cbf_const, 
                 obstacles= None,  obs_diam = 0.5, robot_diam = 0.5, alpha=10, mu=10):
        
        self.dt = dt # Period
        self.N = N  # Horizon Length 
        self.n_neighbors = n_neighbors
        self.Q_x = Q[0]
        self.Q_y = Q[1]
        self.Q_theta = Q[2]
        self.R_v = R[0]
        self.R_omega = R[1]
        self.n_states = 0
        self.n_controls = 0

        self.v_lim = v_lim
        self.omega_lim = omega_lim

        # Initialized in mpc_setup
        self.solver = None
        self.f = None
        self.states = init_state

        self.robot_diam = robot_diam
        self.obs_diam = obs_diam
        self.cbf_const = cbf_const # Bool flag to enable obstacle avoidance
        self.alpha= alpha # Parameter for scalar class-K function, must be positive
        self.obstacles = obstacles

        self.mu = mu

        # Setup with initialization params
        self.setup()

    ## Utilies used in MPC optimization
    # CBF Implementation
    def h_obs(self, state, obstacle, r):
            ox, oy = obstacle
            return ((ox - state[0])**2 + (oy - state[1])**2 - r**2)


    def shift_timestep(self, h, time, state, control):
        delta_state = self.f(state, control[:, 0])
        next_state = casadi.DM.full(state + h * delta_state)
        next_time = time + h
        next_control = casadi.horzcat(control[:, 1:],
                                    casadi.reshape(control[:, -1], -1, 1))
        self.states = np.array(next_state)[:,0]
        return next_time, next_state, next_control

    def update_param(self, x0, ref, k, N, nb_states, target_bearing):
        p = casadi.vertcat(x0)
        # Reference trajectory as parameter
        for l in range(N):
            if k+l < ref.shape[0]:
                ref_state = ref[k+l, :]
            else:
                ref_state = ref[-1, :]
            xt = casadi.DM([ref_state[0], ref_state[1], ref_state[2]])
            p = casadi.vertcat(p, xt)
        
        # Neigbouring robots states as parameter
        for i in range(self.n_neighbors):
            nb_state = casadi.DM([nb_states[i,0], nb_states[i,1], nb_states[i,2]])
            p = casadi.vertcat(p, nb_state)
        
        # Target bearing as parameter
        for i in range(target_bearing.shape[0]):
            p = casadi.vertcat(p, target_bearing[i])
        return p
    
    def setup(self):
        x = casadi.SX.sym('x')
        y = casadi.SX.sym('y')
        theta = casadi.SX.sym('theta')
        states = casadi.vertcat(x, y, theta)
        self.n_states = states.numel()

        v = casadi.SX.sym('v')
        omega = casadi.SX.sym('omega')
        controls = casadi.vertcat(v, omega)
        self.n_controls = controls.numel()

        X = casadi.SX.sym('X', self.n_states, self.N + 1)
        U = casadi.SX.sym('U', self.n_controls, self.N)

        # Reference trajectory + Neigboring robots's state + Target Bearing
        P = casadi.SX.sym('P', (self.N + 1) * self.n_states + self.n_neighbors*self.n_states + self.n_neighbors*2)
        
        Q = casadi.diagcat(self.Q_x, self.Q_y, self.Q_theta)
        R = casadi.diagcat(self.R_v, self.R_omega)

        rhs = casadi.vertcat(v * casadi.cos(theta), v * casadi.sin(theta), omega)
        self.f = casadi.Function('f', [states, controls], [rhs])

        cost = 0
        bearing_cost = 0 
        #print(P)
        g = X[:, 0] - P[:self.n_states]
        for k in range(self.N):
            state = X[:, k]
            control = U[:, k]
            ref = P[(k+1)*self.n_states:(k+2)*self.n_states]
            target_bearing =  P[(self.N+1)*self.n_states + self.n_neighbors*self.n_states:] 

            for j in range(self.n_neighbors):
                nb_state = P[(self.N+1)*self.n_states + j*self.n_states: (self.N+1)*self.n_states + (j+1)*self.n_states] 
                nb_pos = nb_state[:2] 
                bearing = (nb_pos - X[:2, k])/np.sqrt((nb_pos[0] - X[0, k])**2 + (nb_pos[1] - X[1, k])**2)
                #print((target_bearing[2*j:2*j+2] - bearing).T @ (target_bearing[2*j:2*j+2] - bearing))
                bearing_cost = bearing_cost + (target_bearing[2*j:2*j+2] - bearing).T @ (target_bearing[2*j:2*j+2] - bearing)
                #print(nb_pos, target_bearing[j:j+2])
            track_cost = (state - ref).T @ Q @ (state - ref) 
            ctrl_cost = control.T @ R @ control 
            cost = cost + track_cost + ctrl_cost + self.mu*bearing_cost/self.n_neighbors
            next_state = X[:, k + 1]
            k_1 = self.f(state, control)
            k_2 = self.f(state + self.dt/2 * k_1, control)
            k_3 = self.f(state + self.dt/2 * k_2, control)
            k_4 = self.f(state + self.dt * k_3, control)
            predicted_state = state + self.dt/6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
            g = casadi.vertcat(g, next_state - predicted_state)
        
        # Add obstacle avoidance constraints
        if self.cbf_const:
            for k in range(self.N):
                state = X[:, k]
                next_state = X[:, k+1]
                for obs in self.obstacles:    
                    h = self.h_obs(state, obs, (self.robot_diam / 2 + self.obs_diam / 2))
                    h_next = self.h_obs(next_state, obs, (self.robot_diam / 2 + self.obs_diam / 2))
                    g = casadi.vertcat(g,-(h_next-h + self.alpha*h))

        opt_variables = casadi.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))

        nlp_prob = {
            'f': cost,
            'x': opt_variables,
            'g': g,
            'p': P
        }

        opts = {
            'ipopt': {
                'sb': 'yes',
                'max_iter': 2000,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': 0,
        }
        self.solver = casadi.nlpsol('solver', 'ipopt', nlp_prob, opts)


    def solve(self, X0, u0, ref, idx, nb_states, target_bearing):   
        lbx = casadi.DM.zeros((self.n_states * (self.N + 1) + self.n_controls * self.N, 1))
        ubx = casadi.DM.zeros((self.n_states * (self.N + 1) + self.n_controls * self.N, 1))

        lbx[0:self.n_states * (self.N + 1):self.n_states] = -casadi.inf
        lbx[1:self.n_states * (self.N + 1):self.n_states] = -casadi.inf
        lbx[2:self.n_states * (self.N + 1):self.n_states] = -casadi.inf

        ubx[0:self.n_states * (self.N + 1):self.n_states] = casadi.inf
        ubx[1:self.n_states * (self.N + 1):self.n_states] = casadi.inf
        ubx[2:self.n_states * (self.N + 1):self.n_states] = casadi.inf

        lbx[self.n_states * (self.N + 1):self.n_states * (self.N + 1) + self.n_controls * self.N:self.n_controls] = self.v_lim[0]
        ubx[self.n_states * (self.N + 1):self.n_states * (self.N + 1) + self.n_controls * self.N:self.n_controls] = self.v_lim[1]
        lbx[self.n_states * (self.N + 1) + 1:self.n_states * (self.N + 1) + self.n_controls *self.N:self.n_controls] = self.omega_lim[0]
        ubx[self.n_states * (self.N + 1) + 1:self.n_states * (self.N + 1) + self.n_controls * self.N:self.n_controls] = self.omega_lim[1]
        
        if self.cbf_const:
            lbg = casadi.DM.zeros((self.n_states * (self.N + 1) + len(self.obstacles)*(self.N), 1))
            ubg = casadi.DM.zeros((self.n_states * (self.N + 1) + len(self.obstacles)*(self.N), 1))

            lbg[self.n_states * (self.N + 1):] = -casadi.inf
            ubg[self.n_states * (self.N + 1):] = 0
        else:
            lbg = casadi.DM.zeros((self.n_states * (self.N+1)))
            ubg = -casadi.DM.zeros((self.n_states * (self.N+1)))
           
         
        args = {
            'lbg': lbg,
            'ubg': ubg,
            'lbx': lbx,
            'ubx': ubx
        }
        args['p'] = self.update_param(X0[:,0], ref, idx, self.N, nb_states, target_bearing)
        #print(args['p'])
        args['x0'] = casadi.vertcat(casadi.reshape(X0, self.n_states * (self.N + 1), 1),
                                        casadi.reshape(u0, self.n_controls * self.N, 1))

        sol = self.solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'],
                        lbg=args['lbg'], ubg=args['ubg'], p=args['p'])

        u = casadi.reshape(sol['x'][self.n_states * (self.N + 1):], self.n_controls, self.N)
        X = casadi.reshape(sol['x'][:self.n_states * (self.N + 1)], self.n_states, self.N + 1)
        
        return u, X 
        

def main(args=None):

    # Consider all homogenous agents with identical parameters
    Q_x = 10
    Q_y = 10
    Q_theta = 1
    R_v = 0.005
    R_omega = 0.005

    dt = 0.1
    N = 20

    v_lim = [-1, 1]
    omega_lim = [-casadi.pi/2, casadi.pi/2]
    Q = [Q_x, Q_y, Q_theta]
    R = [R_v, R_omega]

    n_agents = 4
    n_neighbors = np.zeros(n_agents, dtype=int)

    # Set Neighbors
    edges = [[0,1], [0,3], [0,2],
             [1,3], [1,2], [2,3] ] # these edges are undirected
    adj_matrix = create_adjacency_matrix(edges)

    agents_init_state = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.5, 0.0] ])
    agents_goal_state = np.array([[2.0, 2.0, 0.0], [2.0, 4.0, 0.0], [4.0, 4.0, 0.0], [4.0, 2.0, 0.0] ])

    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if (adj_matrix[i][j]==1):
                n_neighbors[i]+=1         

    t0_list = [0 for i in range(n_agents)]
    agents = [BearingMaintenanceMPC(agents_init_state[i], n_neighbors[i], dt, N, v_lim, omega_lim, Q, R, obstacles= None, cbf_const=False) for i in range(n_agents)]
    ref_state_list = []

    state_0_list = [casadi.DM([agents_init_state[i][0], agents_init_state[i][1], agents_init_state[i][2]]) for i in range(n_agents)]
    u0_list = [casadi.DM.zeros((agents[i].n_controls, N)) for i in range(n_agents)]
    X0_list = [casadi.repmat(state_0_list[i], 1, N + 1) for i in range(n_agents)]

    u_list = [casadi.DM.zeros((agents[i].n_controls, N)) for i in range(n_agents)]
    X_pred_list = [casadi.repmat(state_0_list[i], 1, N + 1) for i in range(n_agents)]
    agents_state_list = [dm_to_array(X0_list[i]) for i in range(n_agents)]
    agents_control_list = [dm_to_array(u0_list[i][:, 0]) for i in range(n_agents)]
    ref_state_list = [np.array([[agents_goal_state[j][0]], [agents_goal_state[j][1]], [np.pi/4]]).T for j in range(n_agents)]
    max_iter = 100

    for t in range(max_iter):
        print(t)
        # Construct a list of neighbor states for all robots
        agent_states = np.array([agents[i].states for i in range(n_agents)])
        for j in range(n_agents):
            neighbor_states = get_neighbors_state(j, agent_states, adj_matrix)
            curr_bearing = get_neighbors_bearing(agent_states[j], neighbor_states)

            neighbor_target_states = get_neighbors_state(j, agents_goal_state, adj_matrix)
            target_bearing = get_neighbors_bearing(agents_goal_state[j], neighbor_target_states)

            #print(neighbor_states, target_bearing)
            u_list[j], X_pred_list[j] = agents[j].solve(X0_list[j], u0_list[j], ref_state_list[j], t, neighbor_states, target_bearing)
        
        for j in range(n_agents):
            agents_state_list[j] = np.dstack((agents_state_list[j], dm_to_array(X_pred_list[j])))
            agents_control_list[j] = np.dstack((agents_control_list[j], dm_to_array(u_list[j][:, 0])))
            t0_list[j], X0_list[j], u0_list[j] = agents[j].shift_timestep(dt, t0_list[j], X_pred_list[j], u_list[j])

    anim_params = {
        'ref_state_list': ref_state_list,
        'agents_init_state':agents_init_state,
        'agents_state_list':agents_state_list,
        'agents_control_list':agents_control_list,
        'obstacles': None,
        'num_frames':max_iter,
        'max_iter':max_iter,
        'pred_horizon':N,
        'adj_matrix':adj_matrix,
        'vor':None,
        'vcentroids': None,
        'save': False,
        
    }
    sim = animate(anim_params)
   
if __name__ == '__main__':
    main()