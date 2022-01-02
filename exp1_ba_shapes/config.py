import argparse
import json
import sys, os
sys.path.append(os.getcwd())

def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', help='Configuration JSON file', default='exp1_ba_shapes/config_file.json')

    # saving data
    parser.add_argument('--data_save_dir', help='File list by write RTL command', default='data')
    parser.add_argument('--data_name', default='ba_shapes')

    # build ba-shape graphs
    parser.add_argument('--n_basis', help='number of nodes in graph', default=2000)
    parser.add_argument('--n_shapes', help='number of houses', default=200)

    # gnn achitecture parameters
    parser.add_argument('--num_layers', help='number of GCN layers', default=3)
    parser.add_argument('--hidden_dim', help='number of neurons in hidden layers', default=16)

    # training parameters
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=200)

    # saving model
    parser.add_argument('--model_save_dir', help='saving directory for gnn model', default='model')

    # explainer params
    parser.add_argument('--true_label', help='do you take target as true label or predicted label', default=True)
    parser.add_argument('--explainer_name', help='explainer', default='random')


    # Parse argument defined so far (in order to process JSON load
    # All arguments not yet defined and passed in command -line are stored in left_argv
    # and processed AFTER JSON load file
    args, unknown = parser.parse_known_args()


    if args.config_file is not None:
        if '.json' in args.config_file:
            # The escaping of "\t" in the config file is necesarry as
            # otherwise Python will try to treat is as the string escape
            # sequence for ASCII Horizontal Tab when it encounters it
            # during json.load
            config = json.load(open(args.config_file))
            print(config)
            parser.set_defaults(**config)

            [
                parser.add_argument(arg)
                for arg in [arg for arg in unknown if arg.startswith('--')]
                if arg.split('--')[-1] in config
            ]

        args = parser.parse_args()
        return args
