########################################################################################################################
# Hyper-Parameters:

Learning Rate: 0.001, Batch Size: 400, Epochs 4000.

Highest Dev ACC achieved: 0.92

Highest Train ACC achieved: 0.94

Lowest Dev Loss achieved: 0.26

Lowest Train Loss achieved: 0.15


(PS: CNN best dev acc: 0.83, Densenet best dev acc: 0.87)
########################################################################################################################

# Network Architecture:

There are 5 layers in total, with the first 2 layers being convolutional layers and the rest three layers are the full-connected linear layers. Overall it is as follows:

Conv1 --> Conv2 --> Linear1 --> Linear2 --> Linear3 --> Output

### A. For Each CNN Layers (and after), We Implement The Following 4 steps: 


1. Constructed **1st convolutional layer**: torch.nn.Conv2d(1, 8, kernel_size=(3, 3)) with kernel size 3x3 no stride, takes in a 1x28x28 black & white image to output a shape of 8x26x26 feature maps. The **2nd Convolutional Layer**: torch.nn.Conv2d(8, 16, kernel_size=(5, 5)), takes in 8x13x13 to produce 16x9x9 feature maps. (Dimension reduction due to the later MaxPool procedures)


2. We implement a **Batch Normalization** after each convolutional layer. Because it enables us to apply normalization step before (or right after) the nonlinear function layers, which makes our training faster and more stable.


3. Use **ReLU** after Batch Normalization to convert negative values to 0. It is used because it does not activate all the neurons at the same time, and that way it can create some sparsity among our neurons somehow similar to drop-out regularization, and avoid over-fitting to some extent. 


4. Deploying a **Max-Pooling** (each with kernel size 2x2) rather than Avg-Pooling afterwards, it is used not only because we want to reduce spatial dimension, but also we want to have some degree of tolerance of image distortion which Average Pooling can not offer us well. By extracting the dominant feature value (max value) from the given region irrespective of the position of the feature value, we can tolerate rotational and position invariant features so that  translating the image by a small amount does not significantly affect much. Whereas Average pooling sometimes can’t extract extreme features because it takes all into count and results an average value which may not be important for classifying.


The detailed process of 2 convolutional layers: Conv1 --> BatchNorm1 --> ReLu1 --> MaxPool1 --> Conv2 --> BatchNorm2 --> ReLu2 --> MaxPool2

### B. For each Linear Layers: 

1. Flatten images and gradually reduce dimensions to generate the output. No any other implementation techniques used between each linear layer in order to keep the architecture simple.


2. The 1st Linear layer takes in 1x256 neurons and out put1x64 neurons.


3. The 2nd Linear layer takes in 1x64 neurons and out put1x32 neurons.


4. The 3rd Linear layer takes in 1x32 neurons and makes the final predictions.


The process of 3 Linear layers: linear1 --> linear2 --> linear3 --> output

##################################################################################################################
# Other Things We Tried: 
All experiments are done with Learning Rate: 0.001, Batch Size: 400, Epochs 4000.

1. (**Not** in our model) We tried to add DropOuts after Convolutional layers with dropout probability (p=0.1), but it does not seem to be helpful in improving the dev accuracy, and even goes down approximately 3~4% (around 0.88). So we did **not** include it in our best model.


2. (**Not** in our model) Same thing happen when we add DropOuts before the 1st Linear layer, with dropout probability 0.5 (p=0.5). 


3. (**Not** in our model) We tried to flip all training images horizontally and make predictions, and the model's highest dev accuracy stays the same comparing to that without flipping (around 0.91). 


4. (**Not** in our model) Also tried to blur all training images a bit (with blurring strength: sigma = 0.5), predictions accuracy stays the same comparing to that without blurring (also around 0.91). 


5. (**Not** in our model) Combing both blurring and flipping makes dev acc goes down approximately 2% to 0.90. 


6. (**In** our model) Batch Normalization works well in our model, and it improves approximately 2~3 % of dev acc, but honestly we don’t exactly know why it works that well.
