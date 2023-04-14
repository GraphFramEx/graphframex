import time
import numpy as np
import threading
from tqdm import tqdm

import torch
from torch_geometric.data import Batch

EPS = 1e-15


class MDPSampler:
    '''Sample from pre-defined MDP'''
    def __init__(self, n, mbsize, mdp, sampling_model, sampling_model_prob=1):
        super(MDPSampler, self).__init__()
        self.mdp = mdp
        self.device = mdp.device
        self.graph = mdp.graph
        self.n = n
        self.mbsize = mbsize
        self._init_sampler(self.n, self.mbsize)

        self.sampling_model = sampling_model
        self.sampling_model_prob = sampling_model_prob
        
    def _init_sampler(self, n_threads, n_samples):
        self.stop_event = threading.Event()
        self.ready_events = [threading.Event() for _ in range(n_threads)]
        self.resume_events = [threading.Event() for _ in range(n_threads)]
        self.results = [None] * n_threads
        self.random_state = np.random.RandomState(int(time.time()))
        def prepare_events(idx):
            while not self.stop_event.is_set():
                try:
                    self.results[idx] = self.sample2batch(self.sample_multiple(n_samples))
                except Exception as e:
                    print(f"Exception while sampling: \n{e}")
                    self.sampler_threads[idx].failed = True
                    self.sampler_threads[idx].exception = e
                    self.ready_events[idx].set()
                    break
                self.ready_events[idx].set()
                self.resume_events[idx].clear()
                self.resume_events[idx].wait()

        self.sampler_threads = [threading.Thread(target=prepare_events, args=(i,)) for i in range(n_threads)]
        for i in self.sampler_threads: 
            setattr(i, 'failed', False)
        [i.start() for i in self.sampler_threads]
        round_robin_idx = [0]
        def generator():
            while True:
                idx = round_robin_idx[0]
                round_robin_idx[0] = (round_robin_idx[0] + 1) % n_threads
                if self.ready_events[idx].is_set():
                    r = self.results[idx]
                    self.ready_events[idx].clear()
                    self.resume_events[idx].set()
                    return r
                elif round_robin_idx[0] == 0:
                    time.sleep(0.001)
        self.generator = generator
    
    def __call__(self):
        return self.generator()

    def sample_multiple(self, mbsize, connected=True, PP=False):
        valid_states = []
        while len(valid_states) < mbsize:
            t1 = time.time()
            state = np.random.binomial(size=self.graph.num_edges, n=1, p=0.5)
            if PP:
                print(f"takes{time.time()-t1}")
            state = torch.from_numpy(state).bool()
            if connected and self.mdp.is_connected(state):
                valid_states.append(state)
            if not connected:
                valid_states.append(state)
        samples = [self.mdp.parents(state) for state in valid_states]
        return zip(*samples)   #return list of parents, list of actions, list of reward.... list_len=mbsize

    def sample2batch(self, samples):
        _parent, action, reward, _state, done = samples
        # The batch index of each parent
        parent_batch = torch.tensor(sum([[i]*len(p) for i, p in enumerate(_parent)], []),
                                    device=self.device).long()
        # Convert all parents and states to repr. Note that this
        # concatenates all the parent lists, which is why we need p_batch
        parent = Batch.from_data_list(sum([self.mdp.states2data_list(p) for p in _parent], [])).to(self.device)
        state = self.mdp.states2batch(_state).to(self.device)
        # Concatenate all the actions (one per parent per sample)
        # action = torch.tensor(sum(action, []), device=self.device).long()
        # rewards and dones
        reward = torch.tensor(reward, device=self.device)
        done = torch.tensor(done, device=self.device)
        return (parent, _parent), parent_batch, action, reward, (state, _state), done

    def stop_samplers_and_join(self):
        self.stop_event.set()
        if hasattr(self, 'sampler_threads'):
          while any([i.is_alive() for i in self.sampler_threads]):
            [i.set() for i in self.resume_events]
            [i.join(0.05) for i in self.sampler_threads]


def create_samplers(n, mbsize, mdps, sampling_model, sampling_model_prob):
    samplers = []
    for mdp in tqdm(mdps):
        samplers.append(
            MDPSampler(n, mbsize, mdp, sampling_model, sampling_model_prob))
    return samplers