import sys
import csv
import os
import numpy as np
import datetime
import torch
import torch.nn.functional as F
from utils.io_argparse import get_args
from utils.accuracies import (dev_acc_and_loss, accuracy, approx_train_acc_and_loss)
import skimage


class BestModel(torch.nn.Module):
    ### Implement model's structure and input/filter/output dimensions
    def __init__(self, n_classes):
        super().__init__()
        
        ### TODO Implement your model here
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=(3, 3))
        self.batchNorm1 = torch.nn.BatchNorm2d(8)
        # self.drop1 = torch.nn.Dropout2d(p=0.1)
        self.relu1 = torch.nn.ReLU()
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=(5, 5))
        self.batchNorm2 = torch.nn.BatchNorm2d(16)
        # self.drop2 = torch.nn.Dropout2d(p=0.1)
        self.relu2 = torch.nn.ReLU()
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=2)
        # self.drop3 = torch.nn.Dropout2d(p=0.5)
        self.linear1 = torch.nn.Linear(256, 64)
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear3 = torch.nn.Linear(32, n_classes)

    def forward(self, x):
        
        ### Implement best model's forward pass module    
        x = self.conv1(x)  # torch.Size([100, 8, 26, 26])
        # x = self.drop1(x)
        x = self.batchNorm1(x)
        x = self.relu1(x)  # shape: same
        x = self.max_pool1(x)  # shape, 13*13
        x = self.conv2(x)  # torch.Size([100, 16, 5, 5])
        x = self.batchNorm2(x)
        # x = self.drop2(x)
        x = self.relu2(x)  # shape: same
        x = self.max_pool2(x)  # 100,16,2,2
        x = x.view(-1, self.num_flat_features(x))
        # x = self.drop3(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

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
        TRAIN_IMAGES = np.load(os.path.join(DATA_DIR, "fruit_images.npy"))
        TRAIN_LABELS = np.load(os.path.join(DATA_DIR, "fruit_labels.npy"))
        DEV_IMAGES = np.load(os.path.join(DATA_DIR, "fruit_dev_images.npy"))
        DEV_LABELS = np.load(os.path.join(DATA_DIR, "fruit_dev_labels.npy"))
        
        ### format the dataset to the appropriate shape/dimensions necessary to be input into your model.
        # rotate image
        # TRAIN_IMAGES = skimage.transform.rotate(TRAIN_IMAGES, 90)
        N_IMAGES = TRAIN_IMAGES.shape[0]
        HEIGHT = TRAIN_IMAGES.shape[1]
        WIDTH = TRAIN_IMAGES.shape[2]
        N_CLASSES = len(np.unique(TRAIN_LABELS))
        # blur images --> no big improvements
        # TRAIN_IMAGES = skimage.filters.gaussian(TRAIN_IMAGES, sigma=(0.5, 0.5), multichannel=True)
        # flip images --> no big improvements
        # TRAIN_IMAGES = np.flip(TRAIN_IMAGES, axis=2)

        ### Normalize the dataset if desired
        # 1. flatten dataset shape from 72000,28,28 to 72000,784
        train_image = TRAIN_IMAGES.reshape(-1, HEIGHT * WIDTH)
        # 2. calculate mean within each image in train, new-axis to add 1 dimensionn for later operation
        train_mean = train_image.mean(axis=1)[:, np.newaxis]
        # 3. calculate std within each image in train
        train_std = train_image.std(axis=1)[:, np.newaxis]
        # 4. normalize: (x-mean)/std and reshape back to 72000,1,28,28
        flat_train_imgs = ((train_image - train_mean) / train_std).reshape(-1, 1, HEIGHT, WIDTH)

        # Normalize dev set
        # 1. flatten dataset shape from xxxx,28,28 to xxxx,784
        dev_image = DEV_IMAGES.reshape(-1, HEIGHT * WIDTH)
        # 2. calculate mean within each image in dev, new-axis to add 1 dimension
        dev_mean = dev_image.mean(axis=1)[:, np.newaxis]
        # 3. std
        dev_std = dev_image.std(axis=1)[:, np.newaxis]
        # 4. normalize: (x-mean)/std
        flat_dev_imgs = ((dev_image - dev_mean) / dev_std).reshape(-1, 1, HEIGHT, WIDTH)

        # do not touch the following 4 lines (these write logging model performance to an output file 
        # stored in LOG_DIR with the prefix being the time the model was trained.)
        LOGFILE = open(os.path.join(LOG_DIR, f"bestmodel.log"),'w')
        log_fieldnames = ['step', 'train_loss', 'train_acc', 'dev_loss', 'dev_acc']
        logger = csv.DictWriter(LOGFILE, log_fieldnames)
        logger.writeheader()
        
        ### change depending on the model's instantiation

        # raise NotImplementedError
        model = BestModel(n_classes=N_CLASSES)
        
        
        ### change the choice of optimizer here if wish.
        optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
        
    
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
            
            # log model performance every 100 epochs
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

                print(f"On step {step}:\tTrain loss {train_loss}\t|\tDev acc is {dev_acc}")
                logger.writerow(step_metrics)
        LOGFILE.close()
        
        ### TODO (OPTIONAL) You can remove the date prefix if you don't want to save every model you train
        ### i.e. "{DATE_PREFIX}_bestmodel.pt" > "bestmodel.pt"
        model_savepath = os.path.join(MODEL_SAVE_DIR,f"{DATE_PREFIX}_bestmodel.pt")
        
        
        print("Training completed, saving model at {model_savepath}")
        torch.save(model, model_savepath)
        
        
    elif MODE == "predict":
        PREDICTIONS_FILE = arguments.get('predictions_file')
        WEIGHTS_FILE = arguments.get('weights')
        if WEIGHTS_FILE is None : raise TypeError("for inference, model weights must be specified")
        if PREDICTIONS_FILE is None : raise TypeError("for inference, a predictions file must be specified for output.")
        TEST_IMAGES = np.load(os.path.join(DATA_DIR, "fruit_test_images.npy"))
        
        model = torch.load(WEIGHTS_FILE)
        
        predictions = []
        for test_case in TEST_IMAGES:
            
            ### TODO implement any normalization schemes you need to apply to your test dataset before inference
            flat_test_case = test_case.reshape(-1)
            norm_test_case = (flat_test_case - flat_test_case.mean()) / flat_test_case.std()
            test_case = norm_test_case.reshape(1, 1, test_case.shape[0], test_case.shape[1])

            x = torch.from_numpy(test_case.astype(np.float32))
            # x = x.view(1,-1)
            logits = model(x)
            pred = torch.max(logits, 1)[1]
            predictions.append(pred.item())
        print(f"Storing predictions in {PREDICTIONS_FILE}")
        predictions = np.array(predictions)
        np.savetxt(PREDICTIONS_FILE, predictions, fmt="%d")

    else:
        raise Exception("Mode not recognized")

    #  python3 main_bestmodel.py --mode "train" \
    #                        --dataDir "datasets" \
    #                        --logDir "log_files" \
    #                        --modelSaveDir "model_files" \
    #                        --LR 0.001 \
    #                        --bs 400 \
    #                        --epochs 4000
    # python3 main_bestmodel.py --mode "predict" --dataDir "datasets" --weights "model_files/bestmodel.pt" --predictionsFile "bestmodel_predictions.csv"
