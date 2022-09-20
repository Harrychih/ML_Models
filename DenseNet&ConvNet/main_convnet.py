import sys
import csv
import os
import numpy as np
import datetime
import torch
import torch.nn.functional as F
from utils.io_argparse import get_args
from utils.accuracies import (dev_acc_and_loss, accuracy, approx_train_acc_and_loss)


class TooSimpleConvNN(torch.nn.Module):
    def __init__(self, input_height, input_width, n_classes):
        super().__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.n_classes = n_classes

        ### Implement Convnet architecture
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=(3, 3))
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2))
        self.relu2 = torch.nn.ReLU()
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=12)
        self.conv3 = torch.nn.Conv2d(16, n_classes, kernel_size=(1, 1))

    

    def forward(self, x):
        ### Implement feed forward function
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.avg_pool(x)
        x = self.conv3(x)
        return x.view(-1, self.n_classes)



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

        ### get the following parameters and name them accordingly: 
        # [N_IMAGES] Number of images in the training corpus
        N_IMAGES = TRAIN_IMAGES.shape[0]
        # [HEIGHT] Height and [WIDTH] width dimensions of each image
        HEIGHT = TRAIN_IMAGES.shape[1]
        WIDTH = TRAIN_IMAGES.shape[2]
        # [N_CLASSES] number of output classes
        N_CLASSES = len(np.unique(TRAIN_LABELS))


        ### Normalize each of the individual images to a mean of 0 and a variance of 1
        # 1. flatten dataset shape from 72000,28,28 to 72000,784
        train_image = TRAIN_IMAGES.reshape(-1, HEIGHT * WIDTH)
        # 2. calculate mean within each image in train, new-axis to add 1 dimensionn for later operation
        train_mean = train_image.mean(axis=1)[:, np.newaxis]
        # 3. calculate std within each image in train
        train_std = train_image.std(axis=1)[:, np.newaxis]
        # 4. normalize: (x-mean)/std and reshape back to 72000,1,28,28
        flat_train_imgs = ((train_image - train_mean) / train_std).reshape(-1, 1, HEIGHT, WIDTH)

        ### Store flattened validation images into variable [flat_dev_imgs]
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
        LOGFILE = open(os.path.join(LOG_DIR, f"convnet.log"), 'w')
        log_fieldnames = ['step', 'train_loss', 'train_acc', 'dev_loss', 'dev_acc']
        logger = csv.DictWriter(LOGFILE, log_fieldnames)
        logger.writeheader()

        # call on model
        model = TooSimpleConvNN(input_height=HEIGHT, input_width=WIDTH,
                                n_classes=N_CLASSES)

        ### (OPTIONAL) : can change the choice of optimizer here if wish.
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
        ### i.e. "{DATE_PREFIX}_convnet.pt" > "convnet.pt"
        model_savepath = os.path.join(MODEL_SAVE_DIR, f"{DATE_PREFIX}_convnet.pt")

        print("Training completed, saving model at {model_savepath}")
        torch.save(model, model_savepath)


    elif MODE == "predict":
        PREDICTIONS_FILE = arguments.get('predictions_file')
        WEIGHTS_FILE = arguments.get('weights')
        if WEIGHTS_FILE is None: raise TypeError("for inference, model weights must be specified")
        if PREDICTIONS_FILE is None: raise TypeError("for inference, a predictions file must be specified for output.")
        TEST_IMAGES = np.load(os.path.join(DATA_DIR, "fruit_test_images.npy"))

        model = torch.load(WEIGHTS_FILE)

        predictions = []
        for test_case in TEST_IMAGES:
            ### Normalize test dataset
            flat_test_case = test_case.reshape(-1)
            norm_test_case = (flat_test_case - flat_test_case.mean()) / flat_test_case.std()
            test_case = norm_test_case.reshape(-1, 1, test_case.shape[0], test_case.shape[1])

            x = torch.from_numpy(test_case.astype(np.float32))
            # x = x.view(1, -1)
            logits = model(x)
            pred = torch.max(logits, 1)[1]
            predictions.append(pred.item())
        print(f"Storing predictions in {PREDICTIONS_FILE}")
        predictions = np.array(predictions)
        np.savetxt(PREDICTIONS_FILE, predictions, fmt="%d")


    else:
        raise Exception("Mode not recognized")

    #  python3 main_convnet.py --mode "train" \
    #                        --dataDir "datasets" \
    #                        --logDir "log_files" \
    #                        --modelSaveDir "model_files" \
    #                        --LR 0.01 \
    #                        --bs 200 \
    #                        --epochs 4000
    # python3 main_convnet.py --mode "predict" --dataDir "datasets" --weights "model_files/convnet.pt" --predictionsFile "convnet_predictions.csv"
