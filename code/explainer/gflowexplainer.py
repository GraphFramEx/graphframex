import os
import os.path as osp
import argparse
import pdb
import time
import numpy as np
import gc

import torch
# get utils folder
from explainer.explainer_utils.gflowexplainer.agent import create_agent
from explainer.explainer_utils.gflowexplainer.mdp import create_mdps
from explainer.explainer_utils.gflowexplainer.sampler import create_samplers

EPS = 1e-15


# gflow explainer related parameters
def gflow_parse_args(parser):
    # parser = argparse.ArgumentParser(description="Train GflowExplainers")
    parser.add_argument('--cf_flag', type=bool, default=False)
    parser.add_argument('--balanced_loss', type=bool, default=False)
    parser.add_argument('--verbose', type=int, default=10)
    parser.add_argument('--n_conv', type=int, default=3)
    parser.add_argument('--n_hidden', type=int, default=16)
    parser.add_argument('--n_out_stem', type=int, default=1)
    parser.add_argument('--n_out_graph', type=int, default=1)
    parser.add_argument('--n_thread', type=int, default=8)
    parser.add_argument('--mbsize', type=int, default=4)
    parser.add_argument('--sampling_model_prob', type=float, default=8)
    parser.add_argument('--log_reg_c', type=float, default=2.5e-5)
    parser.add_argument('--leaf_coef', type=float, default=1)
    parser.add_argument('--clip_loss', type=float, default=0)
    parser.add_argument('--clip_grad', type=float, default=0)
    parser.add_argument('--sample_prob', type=float, default=1)
    parser.add_argument('--learning_rate_gflow', type=float, default=1e-2)
    parser.add_argument('--weight_decay_gflow', type=float, default=0)
    parser.add_argument('--opt_beta', type=float, default=0.9)
    parser.add_argument('--opt_beta2', type=float, default=0.999)
    parser.add_argument('--opt_epsilon', type=float, default=1e-08)
    parser.add_argument('--num_steps', type=int, default=10000)
    parser.add_argument('--N', type=int, default=1000)
    return parser.parse_args()


class GFlowExplainer(torch.nn.Module):
    def __init__(self, model, device):
        super(GFlowExplainer, self).__init__()
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.random_state = np.random.RandomState(int(time.time()))
        # Is random state the seed?

    def train_explainer(self, args, train_dataset, **kwargs):
        self.cf_flag = args.cf_flag
        agent = create_agent(args, device=self.device)
        mdps = create_mdps(train_dataset, self.model, device=self.device)
        samplers = create_samplers(args.n_thread, args.mbsize, mdps, agent, args.sampling_model_prob)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        agent = self.train_agent(args, agent, self.model, samplers)
        return agent


    def train_agent(self, args, agent, gnn_model, samplers):
        '''Train the RL agent'''

        for param in gnn_model.parameters():
            param.requires_grad = False
        torch.autograd.set_detect_anomaly(True)

        # 1. initialization of variables, optimizer, and sampler
        debug_no_threads = False
        mbsize = args.mbsize
        log_reg_c = args.log_reg_c
        leaf_coef = args.leaf_coef
        balanced_loss = args.balanced_loss
        clip_loss = torch.tensor([args.clip_loss], device=self.device).to(torch.float)

        optimizer = torch.optim.Adam(agent.parameters(),
                                     args.learning_rate_gflow,
                                     weight_decay=args.weight_decay_gflow,
                                     betas=(args.opt_beta, args.opt_beta2),
                                     eps=args.opt_epsilon
                                     )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=args.num_steps // 3)
        for i in range(len(samplers)):
            # 2. Sample trajectories on MDP
            sampler = samplers[i]
            if not debug_no_threads:
                samples = sampler()
                for thread in sampler.sampler_threads:
                    if thread.failed:
                        sampler.stop_samplers_and_join()
                        pdb.post_mortem(thread.exception.__traceback__)
                        return
                (p_graph, p_state), pb, action, reward, (s_graph, s_state), done = samples
            else:
                (p_graph, p_state), pb, action, reward, (s_graph, s_state), done = sampler.sample2batch(
                    sampler.sample_multiple(mbsize, PP=False))
            done = done.float()
            # Since we sampled 'mbsize' trajectories, we're going to get
            # roughly mbsize * H (H is variable)  transitions
            ntransitions = reward.shape[0]
            # 3. Forward the trajectories in agent to compute the flows
            edge_out_s, graph_out_s = agent(s_graph, s_state, sampler.graph, gnn_model, actions=None)
            edge_out_p, graph_out_p = agent(p_graph, p_state, sampler.graph, gnn_model, actions=action)
            qsa_p = agent.index_output_by_action(p_graph, edge_out_p, graph_out_p, action)

            exp_inflow = (torch.zeros((ntransitions,), device=self.device)
                          .index_add_(0, pb, torch.exp(qsa_p)))  # pb is the parents' batch index
            inflow = torch.log(exp_inflow + log_reg_c)

            edge_out_s = torch.cat([edge_out.view(1, -1).to(self.device) for edge_out in edge_out_s], dim=0)
            exp_outflow = agent.sum_output(s_graph, torch.exp(edge_out_s), 0)
            outflow_plus_r = torch.log(log_reg_c + reward + exp_outflow * (1 - done))

            # 4. Compute flow loss and backward
            losses = (inflow - outflow_plus_r).pow(2)

            if clip_loss > 0:
                ld = losses.detach()
                losses = losses / ld * torch.minimum(ld, clip_loss)

            term_loss = (losses * done).sum() / (done.sum() + 1e-20)
            flow_loss = (losses * (1 - done)).sum() / ((1 - done).sum() + 1e-20)
            if balanced_loss:
                loss = term_loss * leaf_coef + flow_loss
            else:
                loss = losses.mean()

            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_value_(agent.parameters(),
                                                args.clip_grad)
            optimizer.step()
            scheduler.step()
            agent.training_steps = i + 1

            # free up unnecessary memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            else:
                gc.collect()

        sampler.stop_samplers_and_join()
        return agent





