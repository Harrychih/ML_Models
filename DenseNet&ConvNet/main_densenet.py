import sys
import csv
import os
import numpy as np
import datetime
import torch
import torch.nn.functional as F
from utils.io_argparse import get_args
from utils.accuracies import (dev_acc_and_loss, accuracy, approx_train_acc_and_loss)


class TwoLayerDenseNet(torch.nn.Module):
    def __init__(self, input_shape, hidden_layer_width, n_classes):
        """Instantiate two nn.LInear modules and assign them as member variables

        Args:
            input_shape (int): shape of input going into neural net
            hidden_layer_width (int): number of nodes in the single hidden layer within the model
            n_classes (int): number of output classes
        """
        super().__init__()
        ### Implement the network architecture
        self.input_shape = input_shape
        self.hidden_layer_width = hidden_layer_width
        self.n_classes = n_classes
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(self.input_shape,self.hidden_layer_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_width, self.n_classes)
            # torch.nn.Dropout(0.1)
            # torch.nn.Softmax()
        )
        
        # raise NotImplementedError


    def forward(self, x):
        """Forward function accepts tensor of input data, returns tensor of output data.
        Modules defined in constructor are used, along with arbitrary operators on tensors
        """
        
        ### Implement the forward function
        logits = self.linear_relu_stack(x)
        return logits
        
        # raise NotImplementedError

if __name__ == "__main__":
    arguments = get_args(sys.argv)
    MODE = arguments.get('mode')
    DATA_DIR = arguments.get('data_dir')
    
    
    if MODE == "train":
        
        LOG_DIR = arguments.get('log_dir')
        MODEL_SAVE_DIR = arguments.get('model_save_dir')
        LEARNING_RATE = arguments.get('lr')
        BATCH_SIZE = arguments.get('bs')
        EPOCHS = arguments.get('epochs')
        DATE_PREFIX = datetime.datetime.now().strftime('%Y%m%d%H%M')
        if LEARNING_RATE is None: raise TypeError("Learning rate has to be provided for train mode")
        if BATCH_SIZE is None: raise TypeError("batch size has to be provided for train mode")
        if EPOCHS is None: raise TypeError("number of epochs has to be provided for train mode")
        # Training data
        TRAIN_IMAGES = np.load(os.path.join(DATA_DIR, "fruit_images.npy"))
        TRAIN_LABELS = np.load(os.path.join(DATA_DIR, "fruit_labels.npy"))
        # validation data
        DEV_IMAGES = np.load(os.path.join(DATA_DIR, "fruit_dev_images.npy"))
        DEV_LABELS = np.load(os.path.join(DATA_DIR, "fruit_dev_labels.npy"))
        
        ### TODO get the following parameters and name them accordingly: 
        # [N_IMAGES] Number of images in the training corpus
        N_IMAGES = TRAIN_IMAGES.shape[0]
        # [HEIGHT] Height and [WIDTH] width dimensions of each image
        HEIGHT = TRAIN_IMAGES.shape[1]
        WIDTH = TRAIN_IMAGES.shape[2]
        # [N_CLASSES] number of output classes
        N_CLASSES = len(np.unique(TRAIN_LABELS))
        # [N_DEV_IMGS] number of images in the validation corpus (DEV_IMAGES)
        N_DEV_IMGS = DEV_IMAGES.shape[0]
        # [FLATTEN_DIM] the dimension of one image if you were to turn it into a vector
        FLATTEN_DIM = HEIGHT * WIDTH
        # raise NotImplementedError
        
        ### Normalize each of the flattened images in BOTH the training and validation dataset to a mean of 0; variance of 1.
        ### Store flattened training images into variable [flat_train_imgs]
        train_imgs = TRAIN_IMAGES.reshape((N_IMAGES,FLATTEN_DIM))
        train_imgs_mean = train_imgs.mean(axis=1)[:, np.newaxis]
        train_imgs_std = train_imgs.std(axis=1)[:, np.newaxis]
        flat_train_imgs = (train_imgs - train_imgs_mean)/train_imgs_std
        ### Store flattened validation images into variable [flat_dev_imgs]
        dev_imgs = DEV_IMAGES.reshape((N_DEV_IMGS,FLATTEN_DIM))
        dev_imgs_mean = dev_imgs.mean(axis=1)[:, np.newaxis]
        dev_imgs_std = dev_imgs.std(axis=1)[:, np.newaxis]
        flat_dev_imgs = (dev_imgs - dev_imgs_mean)/dev_imgs_std
        # raise NotImplementedError
        
        # do not touch the following 4 lines (these write logging model performance to an output file 
        # stored in LOG_DIR with the prefix being the time the model was trained.)
        LOGFILE = open(os.path.join(LOG_DIR, f"densenet.log"),'w')
        log_fieldnames = ['step', 'train_loss', 'train_acc', 'dev_loss', 'dev_acc']
        logger = csv.DictWriter(LOGFILE, log_fieldnames)
        logger.writeheader()
        
        ### Set the hidden layer width, input shape, and number of classes for the model, as defined in the class init.
        hidden_layer_width = 100
        n_classes = N_CLASSES
        model = TwoLayerDenseNet(input_shape=FLATTEN_DIM,hidden_layer_width=hidden_layer_width,n_classes=n_classes)
        
        # raise NotImplementedError
        
        
        
        #  can change the choice of optimizer here if wish.
        optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=1e-4)
        
        for step in range(EPOCHS):
            i = np.random.choice(flat_train_imgs.shape[0], size=BATCH_SIZE, replace=False)
            x = torch.from_numpy(flat_train_imgs[i].astype(np.float32))
            y = torch.from_numpy(TRAIN_LABELS[i].astype(np.int))
            
            
            # Forward pass: Get logits for x
            logits = model(x)
            # Compute loss
            loss = F.cross_entropy(logits, y)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            if step % 100 == 0:
                train_acc, train_loss = approx_train_acc_and_loss(model, flat_train_imgs, TRAIN_LABELS)
                dev_acc, dev_loss = dev_acc_and_loss(model, flat_dev_imgs, DEV_LABELS)
                step_metrics = {
                    'step': step, 
                    'train_loss': loss.item(), 
                    'train_acc': train_acc,
                    'dev_loss': dev_loss,
                    'dev_acc': dev_acc
                }

                # print(f"On step {step}:\tTrain loss {train_loss}\t|\tDev acc is {dev_acc}")
                print(f"On step {step}:\tTrain loss {train_loss}\t|\t Train acc {train_acc}\t|\tDev acc is {dev_acc}")
                logger.writerow(step_metrics)
        LOGFILE.close()
        
        ### (OPTIONAL) can remove the date prefix if don't want to save every model trained
        ### i.e. "{DATE_PREFIX}_densenet.pt" > "densenet.pt"
        # model_savepath = os.path.join(MODEL_SAVE_DIR,f"{DATE_PREFIX}_densenet.pt")
        model_savepath = os.path.join(MODEL_SAVE_DIR,"densenet.pt")
        
        print("Training completed, saving model at {model_savepath}")
        torch.save(model, model_savepath)
        
        
    elif MODE == "predict":
        PREDICTIONS_FILE = arguments.get('predictions_file')
        WEIGHTS_FILE = arguments.get('weights')
        if WEIGHTS_FILE is None : raise TypeError("for inference, model weights must be specified")
        if PREDICTIONS_FILE is None : raise TypeError("for inference, a predictions file must be specified for output.")
        # Testing images
        TEST_IMAGES = np.load(os.path.join(DATA_DIR, "fruit_test_images.npy"))
        
        model = torch.load(WEIGHTS_FILE)
        
        predictions = []
        for test_case in TEST_IMAGES:
            ### TODO Normalize your test dataset  (identical to how you did it with training images)
            flat_test_case = test_case.reshape(-1)
            test_case = (flat_test_case - flat_test_case.mean())/flat_test_case.std()  
            
            # raise NotImplementedError
        
        
            x = torch.from_numpy(test_case.astype(np.float32))
            x = x.view(1,-1)
            logits = model(x)
            pred = torch.max(logits, 1)[1]
            predictions.append(pred.item())
        print(f"Storing predictions in {PREDICTIONS_FILE}")
        predictions = np.array(predictions)
        np.savetxt(PREDICTIONS_FILE, predictions, fmt="%d")
        
    else: raise Exception("Mode not recognized")

    # python3 main_densenet.py --mode "train" \   
    #                        --dataDir "datasets" \
    #                        --logDir "log_files" \               
    #                        --modelSaveDir "model_files" \              
    #                        --LR 0.001 \
    #                        --bs 200 \
    #                        --epochs 2000

    # python3 main_densenet.py --mode "predict" --dataDir "datasets" --weights "model_files/densenet.pt" --predictionsFile "densenet_predictions.csv"
