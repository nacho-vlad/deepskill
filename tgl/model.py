import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from memorys import MailBox
from modules import GeneralModel
from sampler import *
from utils import *

class TemporalGraphModel:

    def __init__(self, data: str, config: str, stored_model: str):        
        node_feats, edge_feats = load_feat(data)
        g, df = load_graph(data)
        sample_param, memory_param, gnn_param, train_param = parse_config(config)

        gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
        gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
        combine_first = False
        if 'combine_neighs' in train_param and train_param['combine_neighs']:
            combine_first = True
        model = GeneralModel(
                    gnn_dim_node, gnn_dim_edge, 
                    sample_param, memory_param, 
                    gnn_param, train_param, 
                    combined=combine_first
                ).cuda()
        mailbox = MailBox(memory_param, g['indptr'].shape[0] - 1, gnn_dim_edge) if memory_param['type'] != 'none' else None
        
        sampler = None
        if not ('no_sample' in sample_param and sample_param['no_sample']):
            sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                                    sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                                    sample_param['strategy']=='recent', sample_param['prop_time'],
                                    sample_param['history'], float(sample_param['duration']))
        neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1)

        model.load_state_dict(torch.load(stored_model))

        if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
            if node_feats is not None:
                node_feats = node_feats.cuda()
            if edge_feats is not None:
                edge_feats = edge_feats.cuda()
            if mailbox is not None:
                mailbox.move_to_gpu()
        
        self.node_feats = node_feats
        self.edge_feats = edge_feats
        self.g = g
        self.df = df
        self.sample_param = sample_param
        self.memory_param = memory_param
        self.gnn_param = gnn_param
        self.train_param = train_param
        self.model = model
        self.mailbox = mailbox
        self.sampler = sampler
        self.neg_link_sampler = neg_link_sampler
        self.processed_edge_id = 0
        self.combine_first = combine_first
        
    
    def forward_model_to(self, time):
        if self.processed_edge_id >= len(self.df):
            return
        while self.df.time[self.processed_edge_id] < time:
            rows = self.df[self.processed_edge_id:min(self.processed_edge_id + self.train_param['batch_size'], len(self.df))]
            self.model.eval()
            root_nodes = np.concatenate([rows.src.values, rows.dst.values, self.neg_link_sampler.sample(len(rows))]).astype(np.int32)
            ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
            if self.sampler is not None:
                if 'no_neg' in self.sample_param and self.sample_param['no_neg']:
                    pos_root_end = root_nodes.shape[0] * 2 // 3
                    self.sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    self.sampler.sample(root_nodes, ts)
                ret = self.sampler.get_ret()
            if self.gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, self.sample_param['history'])
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts)
            mfgs = prepare_input(mfgs, self.node_feats, self.edge_feats, combine_first=self.combine_first)
            if self.mailbox is not None:
                self.mailbox.prep_input_mails(mfgs[0])
            with torch.no_grad():
                _, _ = self.model(mfgs)
                if self.mailbox is not None:
                    eid = rows['Unnamed: 0'].values
                    mem_edge_feats = self.edge_feats[eid] if self.edge_feats is not None else None
                    block = None
                    if self.memory_param['deliver_to'] == 'neighbors':
                        block = to_dgl_blocks(ret, self.sample_param['history'], reverse=True)[0][0]
                    self.mailbox.update_mailbox(self.model.memory_updater.last_updated_nid, self.model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
                    self.mailbox.update_memory(self.model.memory_updater.last_updated_nid, self.model.memory_updater.last_updated_memory, root_nodes, self.model.memory_updater.last_updated_ts)
            self.processed_edge_id += self.train_param['batch_size']
            if self.processed_edge_id >= len(self.df):
                return

    def get_node_emb(self, root_nodes, ts):
        self.forward_model_to(ts[-1])
        if self.sampler is not None:
            self.sampler.sample(root_nodes, ts)
            ret = self.sampler.get_ret()
        if self.gnn_param['arch'] != 'identity':
            mfgs = to_dgl_blocks(ret, self.sample_param['history'])
        else:
            mfgs = node_to_dgl_blocks(root_nodes, ts)
        mfgs = prepare_input(mfgs, self.node_feats, self.edge_feats, combine_first=self.combine_first)
        if self.mailbox is not None:
            self.mailbox.prep_input_mails(mfgs[0])
        with torch.no_grad():
            ret = self.model.get_emb(mfgs)
        return ret.detach().cpu()

    def graph(self):
        return self.df, self.edge_feats
    
    def node_count(self):
        return max(self.df.max()[2], self.df.max()[3])
    
    def timestamps(self):
        return self.df['time']

# tgm = TemporalGraphModel('LICHESS', 'config/TGN.yml', 'models/LICHESS_2013-02.pkl')
# node_count = tgm.node_count()
# all_nodes = np.arange(0, node_count, dtype=int)
# ts = np.repeat(tgm.timestamps().max(), len(all_nodes))
# print(tgm.node_count())
# print(tgm.timestamps().max())
# print(tgm.get_node_emb(all_nodes, ts))
