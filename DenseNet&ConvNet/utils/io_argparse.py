import argparse

# DO NOT MODIFY THIS FILE
# DO NOT MODIFY THIS FILE

def get_args(args : list) -> dict:
    """Read in arguments for parsing by main_* module. Helper for trainUnetModel.py
	Arguments:
		args (list): sys.argv list; list of commands passed in from command line
	Returns:
		options (dict) : dictionary object with specified arguments
	"""
    parser = argparse.ArgumentParser(description = "HW3 command line argument parser")
    # dataset location arguments
    parser.add_argument('--dataDir',
                        dest = 'data_dir',
                        help = 'directory that contains your data files in npy format',
                        type = str,
                        required=False,
                        default="datasets")
    
    # Logging arguments
    parser.add_argument('--logDir',
                        dest = 'log_dir',
                        help = 'directory that contains your training logs (only accessed during train mode)',
                        type = str,
                        required=False,
                        default="log_files")
    parser.add_argument('--modelSaveDir',
                        dest = 'model_save_dir',
                        help = 'path to save your model files',
                        type = str,
                        required=False)
    parser.add_argument('--predictionsFile',
                        dest = 'predictions_file',
                        help = 'only used during predict mode, points to where to save your output model predictions',
                        type = str, default= None,
                        required=False)
    
    # Training/inference arguments
    parser.add_argument('--mode', 
                         dest = 'mode', 
                        help = 'run in either train or predict mode', 
                        type = str, 
                        choices = ["train", "predict"],
                        required = True)
    parser.add_argument('--LR',
                        dest = 'lr',
                        help = "learning rate to use with your optimizer during training",
                        required = False, default = None,
                        type = float)
    parser.add_argument('--bs',
                        dest = 'bs',
                        help = "number of examples to include in your batch",
                        required = False, default = None,
                        type = int)
    parser.add_argument('--epochs',
                        dest = 'epochs',
                        help = "number of epochs to train upon",
                        required = False, default = None,
                        type = int)
    parser.add_argument('--weights',
                        dest = 'weights',
                        help = 'directory that contains your trained model weights in pt format',
                        type = str, default=None,
                        required=False)
    options = vars(parser.parse_args())
    return options
