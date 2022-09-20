""" Main file. This is the starting point for your code execution.

You shouldn't need to change anything in this code.
"""

import os
import argparse as ap
import pickle
import numpy as np

import model_HAC


def get_args():
    p = ap.ArgumentParser(description="This is the main test harness for your models.")

    # Meta arguments
    p.add_argument("--test-data", type=str, help="Test data file")
    p.add_argument("--predictions-file", type=str,
                    help="Where to dump predictions")

    # Model Hyperparameters
    p.add_argument("--linkage", type=str, choices=["single", "complete", "average"],
                        help="Which linkage criterion to use", default="single")
    p.add_argument("--number-of-clusters", type=int,
                        help="The number of clusters to find", default=2)
    return p.parse_args()


def check_args(args):
    mandatory_args = {'test_data', 'predictions_file', 'linkage', 'number_of_clusters'}
    if not mandatory_args.issubset(set(dir(args))):
        raise Exception("You're missing essential arguments!"
                         "We need these to run your code.")
    if args.predictions_file is None:
        raise Exception("--predictions-file should be specified during testing")
    if args.test_data is None:
        raise Exception("--test-data should be specified during testing")
    elif not os.path.exists(args.test_data):
        raise Exception("data file specified by --test-data does not exist.")

def test(args):
    """ 
    Make predictions over the input test dataset, and store the predictions.
    """
    #load dataset and model
    n_clusters = args.number_of_clusters
    linkage = args.linkage
    X = np.loadtxt(args.test_data, delimiter=",")

    # predict labels for dataset
    preds = model_HAC.AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit_predict(X)
    
    # output model predictions
    np.savetxt(args.predictions_file, preds, fmt='%d')


if __name__ == "__main__":
    args = get_args()
    check_args(args)

    test(args)
