import argparse
import os

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model_name', type=str, default='', help='name of stored model')
args=parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import time
import random
import dgl
import numpy as np
from tgl.modules import *
from tgl.sampler import *
from tgl.utils import *
from sklearn.metrics import roc_auc_score, recall_score, precision_score, classification_report, accuracy_score
import wandb

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# set_seed(0)

node_feats, edge_feats = load_feat(args.data, 0, 0)
g, df = load_graph(args.data)
sample_param, memory_param, gnn_param, train_param = parse_config(args.config)
train_edge_end = df[df['ext_roll'].gt(0)].index[0]
val_edge_end = df[df['ext_roll'].gt(1)].index[0]

class_weights = torch.ones(3) - edge_feats[:train_edge_end, 0:3].mean(dim = 0)

gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
combine_first = False
if 'combine_neighs' in train_param and train_param['combine_neighs']:
    combine_first = True
model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, combined=combine_first, game_feats = 2).cuda()
mailbox = MailBox(memory_param, g['indptr'].shape[0] - 1, gnn_dim_edge) if memory_param['type'] != 'none' else None
criterion = torch.nn.CrossEntropyLoss(weight = class_weights.cuda())
optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])
if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
    if node_feats is not None:
        node_feats = node_feats.cuda()
    if edge_feats is not None:
        edge_feats = edge_feats.cuda()
    if mailbox is not None:
        mailbox.move_to_gpu()

sampler = None
if not ('no_sample' in sample_param and sample_param['no_sample']):
    sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                              sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                              sample_param['strategy']=='recent', sample_param['prop_time'],
                              sample_param['history'], float(sample_param['duration']))

wandb.init(
    project="deepskill",
    
    config={
        "learning_rate": train_param["lr"],
        "architecture": args.config,
        "dataset": args.data,
        "epochs": train_param["epoch"],
        "batch_size": train_param["batch_size"],
        "reorder": train_param["reorder"] if "reorder" in train_param else 0,
        "dim_emb": memory_param["dim_out"]
    }
)

def eval(mode='val'):
    neg_samples = 1
    model.eval()
    aps = list()
    aucs_mrrs = list()
    if mode == 'val':
        eval_df = df[train_edge_end:val_edge_end]
    elif mode == 'test':
        eval_df = df[val_edge_end:]
    elif mode == 'train':
        eval_df = df[:train_edge_end]
    y_pred = torch.empty(len(eval_df), 3)
    y_true = torch.empty(len(eval_df))
    with torch.no_grad():
        total_loss = 0
        for _, rows in eval_df.groupby(eval_df.index // train_param['batch_size']):
            root_nodes = np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)
            ts = np.tile(rows.time.values, 2).astype(np.float32)
            if sampler is not None:
                sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'])
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts)
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
            
            edge_f = torch.index_select(edge_feats, 0, torch.tensor(rows.index.values).cuda())
            
            start_idx = rows.index.values[0] - eval_df.index.values[0]
            end_idx = rows.index.values[-1] + 1 - eval_df.index.values[0]
            
            pred = model(mfgs, neg_samples = 0, edge_feats = edge_f[:, 3:5])
            total_loss += criterion(pred, edge_f[:, 0:3])
            
            y_pred[start_idx:end_idx, :] = pred.softmax(dim = 1).cpu()
            y_true[start_idx:end_idx] = edge_f[:, 0:3].argmax(dim = 1).cpu()
            if mailbox is not None:
                eid = rows['Unnamed: 0'].values
                mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                block = None
                if memory_param['deliver_to'] == 'neighbors':
                    block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block, neg_samples = 0)
                mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, neg_samples= 0)
        if mode == 'val':
            val_losses.append(float(total_loss))
    if mode == 'test':
        return classification_report(y_true, y_pred.argmax(dim = 1), target_names = ['White', 'Black', 'Draw'], zero_division = 0, digits = 4)
    acc = accuracy_score(y_true, y_pred.argmax(dim = 1))
    ap = precision_score(y_true, y_pred.argmax(dim = 1), average = 'macro', zero_division = 0, labels = [0, 1])
    auc = roc_auc_score(y_true, y_pred, multi_class = 'ovr')
    return ap, auc, acc

if not os.path.isdir('tgl/models'):
    os.mkdir('tgl/models')
if args.model_name == '':
    path_saver = 'tgl/models/{}.pkl'.format(time.time())
else:
    path_saver = 'tgl/models/{}.pkl'.format(args.model_name)
    if os.path.isfile(path_saver):
        print(f"Loading model {path_saver}")
        model.load_state_dict(torch.load(path_saver))
        path_saver = 'tgl/models/{}.pkl'.format(time.time())

best_ap = 0
best_e = 0
val_losses = list()
group_indexes = list()
group_indexes.append(np.array(df[:train_edge_end].index // train_param['batch_size']))
if 'reorder' in train_param:
    # random chunk shceduling
    reorder = train_param['reorder']
    group_idx = list()
    for i in range(reorder):
        group_idx += list(range(0 - i, reorder - i))
    group_idx = np.repeat(np.array(group_idx), train_param['batch_size'] // reorder)
    group_idx = np.tile(group_idx, train_edge_end // train_param['batch_size'] + 1)[:train_edge_end]
    group_indexes.append(group_indexes[0] + group_idx)
    base_idx = group_indexes[0]
    for i in range(1, train_param['reorder']):
        additional_idx = np.zeros(train_param['batch_size'] // train_param['reorder'] * i) - 1
        group_indexes.append(np.concatenate([additional_idx, base_idx])[:base_idx.shape[0]])
for e in range(train_param['epoch']):
    print('Epoch {:d}:'.format(e))
    time_sample = 0
    time_prep = 0
    time_tot = 0
    total_loss = 0
    # training
    model.train()
    if sampler is not None:
        sampler.reset()
    if mailbox is not None:
        mailbox.reset()
        model.memory_updater.last_updated_nid = None
    for _, rows in df[:train_edge_end].groupby(group_indexes[random.randint(0, len(group_indexes) - 1)]):
        t_tot_s = time.time()
        root_nodes = np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)
        ts = np.concatenate([rows.time.values, rows.time.values]).astype(np.float32)
        if sampler is not None:
            sampler.sample(root_nodes, ts)
            ret = sampler.get_ret()
            time_sample += ret[0].sample_time()
        t_prep_s = time.time()
        if gnn_param['arch'] != 'identity':
            mfgs = to_dgl_blocks(ret, sample_param['history'])
        else:
            mfgs = node_to_dgl_blocks(root_nodes, ts)
        mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
        if mailbox is not None:
            mailbox.prep_input_mails(mfgs[0])
        time_prep += time.time() - t_prep_s
        optimizer.zero_grad()
        
        edge_f = torch.index_select(edge_feats, 0, torch.tensor(rows.index.values).cuda())
        
        pred = model(mfgs, neg_samples = 0, edge_feats = edge_f[:, 3:5])
        loss = criterion(pred, edge_f[:, 0:3])
        total_loss += float(loss) * train_param['batch_size'] 
        loss.backward()
        optimizer.step()
        t_prep_s = time.time()
        if mailbox is not None:
            eid = rows['Unnamed: 0'].values
            mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
            block = None
            if memory_param['deliver_to'] == 'neighbors':
                block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
            mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block, neg_samples = 0)
            mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, neg_samples = 0)
        time_prep += time.time() - t_prep_s
        time_tot += time.time() - t_tot_s
    ap, auc, acc = eval('val')
    if e > 2 and ap > best_ap:
        best_e = e
        best_ap = ap
        torch.save(model.state_dict(), path_saver)
    wandb.log({"ap": ap, "loss": total_loss, "acc": acc})
    print('\ttrain loss:{:.4f}  val ap:{:4f}  val auc:{:4f}'.format(total_loss, ap, auc))
    print('\ttotal time:{:.2f}s sample time:{:.2f}s prep time:{:.2f}s'.format(time_tot, time_sample, time_prep))

print('Loading model at epoch {}...'.format(best_e))
model.load_state_dict(torch.load(path_saver))
model.eval()
if sampler is not None:
    sampler.reset()
if mailbox is not None:
    mailbox.reset()
    model.memory_updater.last_updated_nid = None
    eval('train')
    eval('val')
print(eval('test'))
