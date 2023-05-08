import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Learning node embeddings.")

    # Input
    parser.add_argument("--G_f", type=str, default="/n/data1/hms/dbmi/zitnik/lab/datasets/2020-12-PPI/processed/ppi_edgelist.txt", help="Directory to global PPI")
    parser.add_argument("--ppi_f", type=str, default="/n/data1/hms/dbmi/zitnik/lab/datasets/2022-09-TabulaSapiens/processed/all/ppi_TabulaSapiens_iteration_maxpval=1.0.csv", help="Directory to PPI layers")
    parser.add_argument("--metagraph_f", type=str, default="/n/data1/hms/dbmi/zitnik/lab/datasets/2022-09-TabulaSapiens/processed/all/mg_edgelist.txt", help="Directory to metagraph")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs to train")
    parser.add_argument("--run", type=str, default="", help="Model hyperparameters")
    
    # Parameters
    parser.add_argument("--sweep", type=bool, default=True, help="Perform sweep")
    parser.add_argument("--loader", type=str, default="graphsaint", choices=["neighbor", "graphsaint"], help="Loader for minibatching.")

    # Hyperparameters
    parser.add_argument("--feat_mat", type=int, default=2048, help="Random Gaussian vectors of shape (1 x 2048)")
    parser.add_argument("--output", type=int, default=8, help="Output size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--lmbda", type=float, default=0.01, help="Lambda (for loss function)")
    parser.add_argument("--lr_cent", type=float, default=0.01, help="Learning rate for center loss")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--norm", type=str, default=None, help="Type of normalization layer to use in up-pooling")
    
    # Save
    parser.add_argument('--save_prefix', type=str, default='TS', help='Prefix of all saved files')
    
    args = parser.parse_args()
    return args


def get_hparams(args):
        
    hparams = {
                'norm': args.norm,
                'feat_mat': args.feat_mat, 
                'output': args.output,
                'lr': args.lr,
                'wd': args.wd,
                'dropout': args.dropout,
                'gradclip': 1.0,
                'n_heads': args.n_heads,
                'lambda': args.lmbda,
                'lr_cent': args.lr_cent,
                'loss_type': "BCE"         
              }
              
    print("Hyperparameters:", hparams)

    return hparams
