# General
import numpy as np
import random
import argparse
import os
import copy

# Pytorch
import torch
import torch.nn as nn
from torch_geometric.utils.convert import to_networkx, to_scipy_sparse_matrix
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

# Center loss
from center_loss import CenterLoss

# W&B
import wandb

# Own code
from generate_input import read_data, get_metapaths, get_centerloss_labels
import model as mdl
import utils
import minibatch_utils as mb_utils
from parse_args import get_args, get_hparams

# Seed
seed = 3
print("SEED:", seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
np.random.seed(seed)
random.seed(seed)
# torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Setup
args = get_args()
hparams_raw = get_hparams(args)

save_log = args.save_prefix + "_gnn_train.log"
save_graph = args.save_prefix + "_graph.pkl"
save_model = args.save_prefix + "_model_save.pth"
save_plots = args.save_prefix + "_train_embed_plots.pdf"
save_ppi_embed = args.save_prefix + "_ppi_embed.pth"
save_cci_bto_embed = args.save_prefix + "_cci_bto_embed.pth"
save_labels_dict = args.save_prefix + "_labels_dict.txt"

log_f = open(save_log, "w")
log_f.write("Number of epochs: %s \n" % args.epochs)
log_f.write("Save model directory: %s \n" % save_model)
log_f.write("Save embeddings directory: %s, %s \n" % (save_ppi_embed, save_cci_bto_embed))
log_f.write("Save graph: %s \n" % save_graph)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
if device.type == 'cuda': print(torch.cuda.get_device_name(0))
best_val_acc = -1
best_model = None
eps = 10e-4

wandb.init(config = hparams_raw, project = "aware", entity = "sc-drug")

hparams = wandb.config

# Read data
ppi_data, cci_bto_data, edge_attr_dict, celltype_map, tissue_neighbors, ppi_layers, cci_bto = read_data(args.G_f, args.ppi_f, args.cci_bto_f, hparams['feat_mat'], args.run, global_G = hparams['global_G'])
ppi_metapaths, cci_bto_metapaths = get_metapaths()
center_loss_labels, train_mask, val_mask, test_mask = get_centerloss_labels(args, celltype_map, ppi_layers)

def train(epoch, model, optimizer, center_loss):

    global args, ppi_data, cci_bto_data, best_model, best_val_acc, hparams

    # Generate PPI batches
    ppi_train_loader_dict, _, ppi_metapaths_train, ppi_x_ori = mb_utils.generate_batch(ppi_data, ppi_metapaths, edge_attr_dict, "train", args.batch_size, device, ppi=True, loader_type=args.loader)
    ppi_val_loader_dict, _, ppi_metapaths_val, _ = mb_utils.generate_batch(ppi_data, ppi_metapaths, edge_attr_dict, "val", args.batch_size, device, ppi=True, loader_type=args.loader)
    
    # Generate CCI-BTO batches
    _, cci_bto_data_train, cci_bto_metapaths_train, cci_bto_x_ori = mb_utils.generate_batch({0: cci_bto_data}, cci_bto_metapaths, edge_attr_dict, "train", args.batch_size, device, ppi=False, loader_type=args.loader)
    _, cci_bto_data_val, cci_bto_metapaths_val, _ = mb_utils.generate_batch({0: cci_bto_data}, cci_bto_metapaths, edge_attr_dict, "val", args.batch_size, device, ppi=False, loader_type=args.loader)

    cci_bto_x_ori = cci_bto_x_ori[0]
    cci_bto_data_train = cci_bto_data_train[0]
    cci_bto_data_val = cci_bto_data_val[0]
    cci_bto_metapaths_train = cci_bto_metapaths_train[0]
    cci_bto_metapaths_val = cci_bto_metapaths_val[0]
    for i, val in enumerate(cci_bto_metapaths_train):
        cci_bto_metapaths_train[i] = val.to(device)
    for key, val in ppi_metapaths_train.items():
        ppi_metapaths_train[key] = [val[0].to(device)]
    
    model.train()
    
    # Run batch training
    _, _, cci_bto_pred, ppi_preds_all, ppi_data_train_y, loss = mb_utils.iterate_train_batch(ppi_train_loader_dict, ppi_x_ori, ppi_metapaths, cci_bto_x_ori, cci_bto_metapaths_train, cci_bto_data_train, tissue_neighbors, model, hparams, device, wandb, center_loss, optimizer, train_mask)
    # ppi_x_ori, cci_bto_x_ori, cci_bto_pred, ppi_preds_all, ppi_data_train_y, loss = utils.iterate_train_batch(ppi_train_loader_dict, ppi_x_ori, ppi_metapaths, cci_bto_x_ori, cci_bto_metapaths_train, cci_bto_data_train, tissue_neighbors, model, hparams, device, wandb, center_loss, optimizer, train_mask)

    # Training metrics
    roc_score, ap_score, train_acc, train_f1 = utils.calc_metrics(cci_bto_pred, cci_bto_data_train, ppi_preds_all, ppi_data_train_y)
    print("Training Metrics:", "ROC", roc_score, "AP", ap_score, "ACC", train_acc, "F1", train_f1)
    wandb.log({"train_roc": roc_score, "train_ap": ap_score, "train_acc": train_acc, "train_f1": train_f1})

    utils.metrics_per_rel(cci_bto_pred, cci_bto_data_train, ppi_preds_all, ppi_data_train_y, edge_attr_dict, celltype_map, log_f, wandb, "train")

    # Validation set predictions
    ppi_x, _, cci_bto_pred, ppi_preds_all, ppi_data_val_y = mb_utils.iterate_predict_batch(ppi_val_loader_dict, ppi_x_ori, ppi_metapaths_train, cci_bto_x_ori, cci_bto_metapaths_train, cci_bto_data_val, tissue_neighbors, model, hparams, device)  # Using train metapaths.
    
    # Validation metrics
    roc_score, ap_score, val_acc, val_f1 = utils.calc_metrics(cci_bto_pred, cci_bto_data_val, ppi_preds_all, ppi_data_val_y)
    print("Validation Metrics:", "ROC", roc_score, "AP", ap_score, "ACC", val_acc, "F1", val_f1)
    utils.metrics_per_rel(cci_bto_pred, cci_bto_data_val, ppi_preds_all, ppi_data_val_y, edge_attr_dict, celltype_map, log_f, wandb, "val")

    calinski_harabasz, davies_bouldin = utils.calc_cluster_metrics(ppi_x)
    
    # Save metrics
    res = "\t".join(["Epoch: %04d" % (epoch + 1), 
                     "train_loss = {:.5f}".format(loss), 
                     "val_roc = {:.5f}".format(roc_score), 
                     "val_ap = {:.5f}".format(ap_score), 
                     "val_f1 = {:.5f}".format(val_acc), 
                     "val_acc = {:.5f}".format(val_f1)])
    print(res)
    log_f.write(res + "\n")
    wandb.log({"total_loss": loss, "total_val_roc": roc_score, "total_val_ap": ap_score, "total_val_acc": val_acc, "total_val_f1": val_f1, "total_val_calinski_harabasz_score": calinski_harabasz, "total_val_davies_bouldin_score": davies_bouldin})

    # Save best model and parameters
    if best_val_acc <= np.mean(val_acc) + eps:
        best_val_acc = np.mean(val_acc)
        with open(save_model, 'wb') as f:
            torch.save(model.state_dict(), f)
        best_model = copy.deepcopy(model)
    
    for i, val in enumerate(cci_bto_metapaths_train):
        cci_bto_metapaths_train[i] = val.detach().cpu()
    for key, val in ppi_metapaths_train.items():
        ppi_metapaths_train[key] = [val[0].detach().cpu()]
    
    return ppi_metapaths_train, cci_bto_metapaths_train, ppi_metapaths_val, cci_bto_metapaths_val


@torch.no_grad()
def test(model, ppi_metapaths_test, cci_bto_metapaths_test):

    if not args.sweep:
        model.load_state_dict(torch.load(save_model))  # !!UNCOMMENT this line after sweep!!
    model.to(device)
    model.eval()
    
    # Generate PPI batches
    ppi_test_loader_dict, _, _, ppi_x = mb_utils.generate_batch(ppi_data, ppi_metapaths, edge_attr_dict, "test", args.batch_size, device, ppi=True, loader_type=args.loader)
    
    if hparams['metagraph']: # Generate CCI-BTO batches
        _, cci_bto_data_test, _, cci_bto_x = mb_utils.generate_batch({0: cci_bto_data}, cci_bto_metapaths, edge_attr_dict, "test", args.batch_size, device, ppi=False, loader_type=args.loader)
        cci_bto_data_test = cci_bto_data_test[0]
        cci_bto_x = cci_bto_x[0]
    else: 
        cci_bto_data_test = {"total_edge_index":[]}
        cci_bto_x = []

    _, _, cci_bto_pred, ppi_preds_all, ppi_data_test_y = mb_utils.iterate_predict_batch(ppi_test_loader_dict, ppi_x, ppi_metapaths_test, cci_bto_x, cci_bto_metapaths_test, cci_bto_data_test, tissue_neighbors, model, hparams, device)

    roc_score, ap_score, test_acc, test_f1 = utils.calc_metrics(cci_bto_pred, cci_bto_data_test, ppi_preds_all, ppi_data_test_y)
    
    print('Test ROC score: {:.5f}'.format(roc_score))
    print('Test AP score: {:.5f}'.format(ap_score))
    print('Test Accuracy: {:.5f}'.format(test_acc))
    print('Test F1 score: {:.5f}'.format(test_f1))
    log_f.write('Test ROC score: {:.5f}\n'.format(roc_score))
    log_f.write('Test AP score: {:.5f}\n'.format(ap_score))
    log_f.write('Test Accuracy: {:.5f}\n'.format(test_acc))
    log_f.write('Test F1 score: {:.5f}\n'.format(test_f1))

    wandb.log({"test_roc": roc_score, "test_ap": ap_score, "test_acc": test_acc, "test_f1": test_f1})
    utils.metrics_per_rel(cci_bto_pred, cci_bto_data_test, ppi_preds_all, ppi_data_test_y, edge_attr_dict, celltype_map, log_f, wandb, "test")


def main():

    global args, ppi_data, cci_bto_data, best_model, hparams, device
    
    # Set up
    model = mdl.TrainNet(cci_bto_data.x.shape[1], hparams['output'], len(ppi_metapaths), len(cci_bto_metapaths), ppi_data, hparams['n_heads'], hparams['norm'], hparams['dropout'], hparams['metagraph'], hparams['metagraph_relw']).to(device)
    if args.run != "": 
        model.load_state_dict(torch.load(save_model))
        save_model = "%s_model_save.pth" % args.run
        print(save_model)
    params = list(model.parameters())
    center_loss = None
    if hparams['use_center_loss']: 
        center_loss = CenterLoss(num_classes=len(set(center_loss_labels)), feat_dim=hparams['output'] * hparams['n_heads'], use_gpu=torch.cuda.is_available())
        params += list(center_loss.parameters())
    optimizer = torch.optim.Adam(params, lr = hparams['lr'], weight_decay = hparams['wd'])
    wandb.watch(model)
    print(model)

    # Train model
    for epoch in range(args.epochs):
        ppi_metapaths_train, cci_bto_metapaths_train, ppi_metapaths_val, cci_bto_metapaths_val = train(epoch, model, optimizer, center_loss)

    print("Optimization finished!")

    # Generate test metapaths
    ppi_metapaths_test = {}
    cci_bto_metapaths_test = []
    for key in ppi_metapaths_train.keys():
        ppi_metapaths_test[key] = [torch.cat(ppi_metapaths_val[key] + ppi_metapaths_train[key], dim=1)]
    for cci_bto_mt_t, cci_bto_mt_v in zip(cci_bto_metapaths_train, cci_bto_metapaths_val):
        cci_bto_metapaths_test.append(torch.cat([cci_bto_mt_t, cci_bto_mt_v], dim=1))
    
    device = torch.device("cpu")
            
    # Test (w/train metapaths trained node embeddings and test links)
    test(best_model, ppi_metapaths_test, cci_bto_metapaths_test)
    
    del ppi_metapaths_train, ppi_metapaths_val, ppi_metapaths_test, cci_bto_metapaths_train, cci_bto_metapaths_val, cci_bto_metapaths_test
    torch.cuda.empty_cache()
    
    # Generate and save best embeddings (w/complete metapaths trained node embeddings)
    _, ppi_data_all, ppi_metapaths_adjs, ppi_x = mb_utils.generate_batch(ppi_data, ppi_metapaths, edge_attr_dict, "all", args.batch_size, device, ppi = False, loader_type=args.loader)
    _, cci_bto_data_all, cci_bto_metapaths_adjs, cci_bto_x = mb_utils.generate_batch({0: cci_bto_data}, cci_bto_metapaths, edge_attr_dict, "all", args.batch_size, device, ppi = False, loader_type=args.loader)
    
    best_ppi_x, best_cci_bto_x = utils.get_embeddings(best_model, ppi_x, cci_bto_x[0], ppi_metapaths_adjs, cci_bto_metapaths_adjs[0], ppi_data_all, cci_bto_data_all[0]["total_edge_index"], tissue_neighbors)

    for celltype, x in best_ppi_x.items():
        best_ppi_x[celltype] = x.to(device)
    best_cci_bto_x = best_cci_bto_x.to(device)
    torch.save(best_ppi_x, save_ppi_embed)
    torch.save(best_cci_bto_x, save_cci_bto_embed)

    # Generate plots
    if hparams['metagraph']:
        labels_dict = utils.plot_emb(best_ppi_x, best_cci_bto_x, celltype_map, ppi_layers, cci_bto, wandb, center_loss_labels)
    else: 
        labels_dict = utils.plot_emb(best_ppi_x, [], celltype_map, ppi_layers, cci_bto, wandb, center_loss_labels)
    
    labels_fout = open(save_labels_dict, "w")
    labels_fout.write(str(labels_dict))
    labels_fout.close()


if __name__ == "__main__":
    main()
