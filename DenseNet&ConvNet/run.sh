#!/usr/bin/env bash

# For training 

python main_densenet.py --mode "train" \
                        --dataDir "datasets" \
                        --logDir "log_files" \
                        --modelSaveDir "model_files" \
                        --LR 0.0001 \
                        --bs 100 \
                        --epochs 2000


python main_convnet.py --mode "train" \
                        --dataDir "datasets" \
                        --logDir "log_files" \
                        --modelSaveDir "model_files" \
                        --LR 0.01 \
                        --bs 100 \
                        --epochs 2000


python main_bestmodel.py --mode "train" \
                        --dataDir "datasets" \
                        --logDir "log_files" \
                        --modelSaveDir "model_files" \
                        --LR 0.01 \
                        --bs 100 \
                        --epochs 2000


# For inference

python main_densenet.py --mode "predict" \
                        --dataDir "datasets" \
                        --weights "model_files/densenet.pt" \
                        --predictionsFile "densenet_predictions.csv"


python main_convnet.py --mode "predict" \
                        --dataDir "datasets" \
                        --weights "model_files/convnet.pt" \
                        --predictionsFile "convnet_predictions.csv"


python main_bestmodel.py --mode "predict" \
                        --dataDir "datasets" \
                        --weights "model_files/bestmodel.pt" \
                        --predictionsFile "best_model_predictions.csv"