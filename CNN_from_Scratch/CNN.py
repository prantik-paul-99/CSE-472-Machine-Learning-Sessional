import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageFilter
import PIL
# ImageFilter
import os
import pickle

# import accuracy, log_loss, f1_score from sklearn
from sklearn.metrics import accuracy_score, f1_score, log_loss

# Building the CNN

import numpy as np
import time
import pandas as pd

def gradient_clipping(gradients, overflow_threshold=100.0, underflow_threshold=1e-15):

    # avoid any nan values
    # gradients = [np.nan_to_num(gradient) for gradient in gradients]

    overflow_indices = np.where(np.abs(gradients) > overflow_threshold)
    gradients[overflow_indices] = np.sign(gradients[overflow_indices]) * overflow_threshold

    zero_indices = np.where(np.abs(gradients) == 0)
    underflow_indices = np.where(np.abs(gradients) < underflow_threshold)
    gradients[underflow_indices] = np.sign(gradients[underflow_indices]) * underflow_threshold
    gradients[zero_indices] = 1e-3

    gradients = [np.nan_to_num(gradient) for gradient in gradients]

    return gradients

def dilate(input, padding = 0, stride = 0):
    # # dilate the output to the original size
    # dilated = np.zeros((output.shape[0], output.shape[1] * stride + padding, output.shape[2] * stride + padding))
    # for h in np.arange(0, output.shape[1]):
    #     for w in np.arange(0, output.shape[2]):
    #         dilated[:, h*stride:h*stride+output.shape[0], w*stride:w*stride+output.shape[0]] = output[:, h, w]
    # return dilated
    dilated = input
    dilated = np.insert(dilated, range(1, input.shape[2]), 0, axis=2)
    dilated = np.insert(dilated, range(1, input.shape[3]), 0, axis=3)
    return dilated


class Convolutional_layer:
    def __init__(self, no_output_chnl, filter_dim, stride, pad):
        self.no_output_chnl = no_output_chnl
        self.filter_dim = filter_dim
        self.stride = stride
        self.pad = pad
        self.filters = None
        self.bias = None

    def get_all_windows(self, input, output_dim, filter_dim, stride):
        # extract necessary information from the input and output_dim
        batch_strides, channel_strides, height_strides, width_strides = input.strides

        batch_size, input_channels, input_height, input_width = input.shape

        output_height, output_width = output_dim[2], output_dim[3]

        return np.lib.stride_tricks.as_strided(input,
        (batch_size, input_channels, output_height, output_width, filter_dim[0], filter_dim[1]),
        (batch_strides, channel_strides, height_strides * stride, width_strides * stride, height_strides, width_strides))

    def forward_propagation(self, input):

        self.input_shape = input.shape
        
        if self.filters is None:
            self.filters = np.random.randn(self.no_output_chnl, input.shape[1], self.filter_dim[0], self.filter_dim[1]) * np.sqrt(2 / (input.shape[1] * self.filter_dim[0] * self.filter_dim[1]))
            self.bias = np.random.randn(self.no_output_chnl)

        # calculate the output dimension with padding and stride
        self.output_dim = (input.shape[0],
                            self.no_output_chnl,
                            int((input.shape[2] - self.filter_dim[0] + 2*self.pad) / self.stride) + 1,
                            int((input.shape[3] - self.filter_dim[1] + 2*self.pad) / self.stride) + 1)

        # initialize the output
        output = np.zeros(self.output_dim)

        # apply padding to the input
        input = np.pad(input, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), 'constant')

        # get all the windows
        self.all_windows = self.get_all_windows(input, self.output_dim, self.filter_dim, self.stride)

        # use the windows to calculate the output
        output = np.einsum('bihwkl,oikl->bohw', self.all_windows, self.filters) + self.bias[None, :, None, None]

        return output


    def back_propagation(self, prev_grad, learning_rate):

        # initialize the gradient of the filters and bias
        grad_filters = np.zeros(self.filters.shape)
        grad_bias = np.zeros(self.bias.shape)

        # calculate the gradient of the filters
        grad_filters = np.einsum('bihwkl,bohw->oikl', self.all_windows, prev_grad)

        # calculate the gradient of the bias
        grad_bias = np.sum(prev_grad, axis=(0, 2, 3))

        # calculate the gradient of the input

        # dilate the prev_grad
        prev_grad_dilated = dilate(prev_grad, self.pad, self.stride)

        # pad the prev_grad
        if(self.pad != 0):
            pad_h =  self.pad
            pad_w = self.pad
        else:
            pad_h = self.filter_dim[0] - 1
            pad_w = self.filter_dim[1] - 1
        prev_grad_dilated_padded = np.pad(prev_grad_dilated, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), 'constant')

        # rotate the filters by 180 degrees
        rotated_filters = np.rot90(self.filters, 2, (2, 3))

        # get all the windows
        prev_grad_windows = self.get_all_windows(prev_grad_dilated_padded, self.input_shape, self.filter_dim, 1)

        grad_input = np.einsum('oikl,bohwkl->bihw', rotated_filters, prev_grad_windows)

        # clip the gradient to avoid exploding gradient
        grad_input = np.copy(gradient_clipping(gradients=grad_input))
        # grad_filters = np.copy(gradient_clipping(gradients=grad_filters))
        # grad_bias = np.copy(gradient_clipping(gradients=grad_bias))

        # update the filters and bias
        self.filters -= learning_rate * grad_filters
        self.bias -= learning_rate * grad_bias

        return grad_input, grad_filters, grad_bias

    def get_filters(self):
        return self.filters

    def get_bias(self):
        return self.bias

    def set_filters(self, filters):
        self.filters = filters

    def set_bias(self, bias):
        self.bias = bias

class ReLU_layer:
    def __init__(self):
        self.mask = None

    def forward_propagation(self, input):
        self.mask = (input <= 0)
        output = input.copy()
        output[self.mask] = 0

        return output

    def back_propagation(self, prev_grad):
        prev_grad[self.mask] = 0
        return prev_grad

class Max_pooling_layer:
    def __init__(self, filter_dim, stride, pad):
        self.filter_dim = filter_dim
        self.stride = stride
        self.pad = pad
        self.max_indices = None

    def get_all_windows(self, input, output_dim, filter_dim, stride):
        # extract necessary information from the input and output_dim
        batch_strides, channel_strides, height_strides, width_strides = input.strides

        batch_size, input_channels, input_height, input_width = input.shape

        output_height, output_width = output_dim[2], output_dim[3]

        return np.lib.stride_tricks.as_strided(input,
        (batch_size, input_channels, output_height, output_width, filter_dim[0], filter_dim[1]),
        (batch_strides, channel_strides, height_strides * stride, width_strides * stride, height_strides, width_strides))

    def forward_propagation(self, input):

        # input shape is (batch_size, height, width, no_channels)
        # output shape is (batch_size, (height - filter_dim + 2*pad)/stride + 1, (width - filter_dim + 2*pad)/stride + 1, no_channels)

        self.input_shape = input.shape
        # calculate the output dimension with padding and stride
        self.output_dim = (input.shape[0],
                        input.shape[1],
                        int((input.shape[2] - self.filter_dim[0] + 2*self.pad) / self.stride) + 1,
                        int((input.shape[3] - self.filter_dim[1] + 2*self.pad) / self.stride) + 1)
        self.max_indices = np.zeros(self.output_dim)

        # initialize the output        
        output = np.zeros(self.output_dim)

        # apply padding to the input
        input = np.pad(input, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), 'constant')
        
        # implement max pooling in a vectorized manner 
        self.all_windows = self.get_all_windows(input, self.output_dim, self.filter_dim, self.stride)
        output = np.max(self.all_windows, axis=(4, 5))
        # print("max pooling output")
        # print(output)

        self.max_indices = np.zeros(self.output_dim).astype(int)
        
        for batch in range(self.output_dim[0]):
            for chnl in range(self.output_dim[1]):
                for hght in range(self.output_dim[2]):
                    for wdth in range(self.output_dim[3]):
                        self.max_indices[batch, chnl, hght, wdth] = np.argmax(input[batch, chnl, hght * self.stride: hght * self.stride + self.filter_dim[0], wdth * self.stride: wdth * self.stride + self.filter_dim[1]])
        # print("max pooling indices")
        # print(self.max_indices)
        return output

    def back_propagation(self, prev_grad):
        # initialize the gradient of the input
        grad_input = np.zeros(self.input_shape)

        # implement max pooling in a vectorized manner
        for batch in range(self.output_dim[0]):
            for chnl in range(self.output_dim[1]):
                for hght in range(self.output_dim[2]):
                    for wdth in range(self.output_dim[3]):
                        # check if we can place the filter at the current position
                        if(hght * self.stride + self.filter_dim[0] <= self.input_shape[2] and wdth * self.stride + self.filter_dim[1] <= self.input_shape[3]):
                            # print(hght * self.stride + self.filter_dim[0])
                            # print(prev_grad.shape[2])
                            # print(wdth * self.stride + self.filter_dim[1])
                            # print(prev_grad.shape[3])
                            # print(grad_input[batch, chnl, hght * self.stride: hght * self.stride + self.filter_dim[0], wdth * self.stride: wdth * self.stride + self.filter_dim[1]].shape)
                            grad_input[batch, chnl, hght * self.stride: hght * self.stride + self.filter_dim[0], wdth * self.stride: wdth * self.stride + self.filter_dim[1]] = np.zeros((self.filter_dim[0], self.filter_dim[1]))
                            grad_input[batch, chnl, hght * self.stride: hght * self.stride + self.filter_dim[0], wdth * self.stride: wdth * self.stride + self.filter_dim[1]].flat[int(self.max_indices[batch, chnl, hght, wdth])] = prev_grad[batch, chnl, hght, wdth]

        return grad_input

class Flatten_layer:
    def __init__(self):
        pass

    def forward_propagation(self, input):
        self.input_shape = input.shape
        return input.reshape(input.shape[0], -1)

    def back_propagation(self, prev_grad):
        return prev_grad.reshape(self.input_shape)

class Fully_connected_layer:
    def __init__(self):
        self.filters = None
        self.bias = None

    def forward_propagation(self, input, output_size):
        if self.filters is None:
            # initialize the filters and bias
            self.filters = np.random.randn(input.shape[1], output_size) * np.sqrt(2.0 / input.shape[1])
            self.bias = np.random.randn(output_size)
        self.input = input
        return input.dot(self.filters) + self.bias

    def back_propagation(self, prev_grad, learning_rate):
        # grad shape is (batch_size, output_size, 1)
        # filters shape is (batch_size, 1, output_size)

        # multiply the grad with the filters to get the gradient of the input
        grad_input = prev_grad.dot(self.filters.T)

        # calculate the gradient of the filters and bias
        grad_wgts = self.input.T.dot(prev_grad)
        grad_bias = np.sum(prev_grad, axis=0)

        # update the filters and bias
        self.filters -= learning_rate * grad_wgts
        self.bias -= learning_rate * grad_bias

        # clip the gradients to avoid exploding gradients
        grad_input = np.copy(gradient_clipping(grad_input))

        return grad_input

    def get_filters(self):
        return self.filters

    def get_bias(self):
        return self.bias

    def set_filters(self, filters):
        self.filters = filters

    def set_bias(self, bias):
        self.bias = bias
        
class Softmax_layer:
    def __init__(self, no_classes):
        self.no_classes = no_classes

    def forward_propagation(self, input):
        self.input = input
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def back_propagation(self, prev_grad):
        # prev_grad is the gradient of the loss function
        # shape of the prev_grad is (batch_size, output_size)
        # shape of the input is (batch_size, output_size)
        # shape of the output is (batch_size, output_size)
        return np.copy(prev_grad)


# Importing the dataset of images using opencv
# Combine training-a, training-b, training-c datasets to form your training + validation set. split by 70 30 ratio
'''
# specify the folder path
folder_paths = ['Dataset/NumtaDB_with_aug/training-a','Dataset/NumtaDB_with_aug/training-b','Dataset/NumtaDB_with_aug/training-c']
# folder_paths = ['Dataset/NumtaDB_with_aug/training-b']
for folder_path in folder_paths:
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
np.save('image_paths.npy', image_paths)

images = []
# get a list of all image file paths in the folder
for folder_path in folder_paths:
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
    for i in range(len(image_paths)):
        image = Image.open(image_paths[i])
        image = image.convert('L') # convert to grayscale
        # invert the image
        image = PIL.ImageOps.invert(image)
        # dilate the image
        image = image.filter(ImageFilter.MaxFilter(3))

        image = image.resize((28,28)) # resize to 28 * 28

        image = np.array(image)

        # normalize the image
        image = image / 255.0

        images.append(image)
    print("done for folder: ", folder_path)
    print(len(images))

print(len(images))
print(images[0].shape)

# get the labels from csv file
folder_paths = ['Dataset/NumtaDB_with_aug/training-a.csv','Dataset/NumtaDB_with_aug/training-b.csv','Dataset/NumtaDB_with_aug/training-c.csv']
# folder_paths = ['Dataset/NumtaDB_with_aug/training-b.csv']
labels = []
for folder_path in folder_paths:
    df = pd.read_csv(folder_path)
    labels.extend(df['digit'])
print(len(labels))
# labels = np.array(labels)

# save the images and labels
np.save('images.npy', images)
np.save('labels.npy', labels)

# load the images and labels
images = np.load('images.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)
image_paths = np.load('image_paths.npy', allow_pickle=True)

print(len(images))
print(images[0].shape)

# Splitting the dataset into the Training set and Validation set
# randomize the dataset
random_indices = np.random.permutation(len(image_paths))

images = images[random_indices]
labels = labels[random_indices]

training_images = images[:int(len(images)*0.7)]
training_labels = labels[:int(len(images)*0.7)]

validation_images = images[int(len(images)*0.7):]
validation_labels = labels[int(len(images)*0.7):]

# generate a random index
idx = np.random.randint(0, len(training_images))
#training_images[idx] = cv2.resize(training_images[idx], (100,100))

# cv2.imshow('image', training_images[idx])
# cv2.waitKey(0)
# show image using pillow package
Image.fromarray(training_images[idx]).show()

print(training_labels[idx])

# # Feature Scaling
# training_images = training_images / 255.0
# validation_images = validation_images / 255.0

############# test dataset ################

folder_paths = ['Dataset/NumtaDB_with_aug/training-d']
for folder_path in folder_paths:
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
np.save('image_paths_test.npy', image_paths)

images = []
# get a list of all image file paths in the folder
for folder_path in folder_paths:
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
    for i in range(len(image_paths)):
        image = Image.open(image_paths[i])
        image = image.convert('L') # convert to grayscale
        # invert the image
        image = PIL.ImageOps.invert(image)
        # dilate the image
        image = image.filter(ImageFilter.MaxFilter(3))

        image = image.resize((28,28)) # resize to 28 * 28

        image = np.array(image)

        # normalize the image
        image = image / 255.0

        images.append(image)
    print("done for folder: ", folder_path)
    print(len(images))

print(len(images))
print(images[0].shape)

# get the labels from csv file
# folder_paths = ['Dataset/NumtaDB_with_aug/training-a.csv','Dataset/NumtaDB_with_aug/training-b.csv','Dataset/NumtaDB_with_aug/training-c.csv']
folder_paths = ['Dataset/NumtaDB_with_aug/training-d.csv']
labels = []
for folder_path in folder_paths:
    df = pd.read_csv(folder_path)
    labels.extend(df['digit'])
print(len(labels))
# labels = np.array(labels)


# save the images and labels
np.save('images_test.npy', images)
np.save('labels_test.npy', labels)
'''

def main():

    # create a log.txt file to store the training history
    if(os.path.exists('log.txt')):
        os.remove('log.txt')
    log_file = open('log.txt', 'w')

    # lr = [0.001, 0.003, 0.005, 0.008, 0.01, 0.03, 0.0005]
    lr = [0.003]

    for learning_rate in lr:
        print("loading data...")
        # print the line to log.txt file
        log_file.write("learning rate: " + str(learning_rate) + "\n")
        # load the training images and labels
        images = np.load('images.npy', allow_pickle=True)
        labels = np.load('labels.npy', allow_pickle=True)
        image_paths_train = np.load('image_paths.npy', allow_pickle=True)

        # # load the test images and labels
        # images_test = np.load('images_test.npy', allow_pickle=True)
        # labels_test = np.load('labels_test.npy', allow_pickle=True)
        # image_paths_test = np.load('image_paths_test.npy', allow_pickle=True)

        print("learning rate: ", learning_rate)
        num_samples = 40000

        # take 2000 images from training set and 1000 images from test set randomly
        random_indices = np.random.choice(len(images), num_samples, replace=False)
        images = images[random_indices]
        labels = labels[random_indices]

        # split the train set into train and validation set
        images_train = images[:int(len(images)*.95)]
        labels_train = labels[:int(len(labels)*.95)]

        images_val = images[int(len(images)*.95):]
        labels_val = labels[int(len(labels)*.95):]

        print(len(images_train))
        log_file.write("training set size: " + str(len(images_train)) + "\n")
        print(len(images_val))
        log_file.write("validation set size: " + str(len(images_val)) + "\n")

        # random_indices = np.random.choice(len(images_test), 100, replace=False)
        # images_test = images_test[random_indices]
        # labels_test = labels_test[random_indices]

        # do one hot encoding on the labels
        labels_train = np.eye(10)[labels_train]
        labels_val = np.eye(10)[labels_val]
        # labels_test = np.eye(10)[labels_test]

        # standard normalization
        images_train = (images_train - np.mean(images_train)) / np.std(images_train)
        images_val = (images_val - np.mean(images_val)) / np.std(images_val)
        # images_test = (images_test - np.mean(images_test)) / np.std(images_test)

        in_channels = 1
        out_channels = 12
        kernel_size = (3,3)
        stride = 1
        padding = 1
        batch_size = (16, in_channels, 28, 28)

        no_of_classes = 10
        # images_train = images_test.reshape(batch_size)
        print(images_train.shape)
        log_file.write("images_train shape: " + str(images_train.shape) + "\n")

        # define the model

        # conv = Convolutional_layer(no_output_chnl=out_channels, filter_dim=kernel_size, stride=stride, pad=padding)
        # relu = ReLU_layer()
        # max_o = Max_pooling_layer(filter_dim=(2, 2), stride=1, pad=1)
        # flat = Flatten_layer()
        # fc = Fully_connected_layer()
        # soft = Softmax_layer(no_of_classes)

        conv1 = Convolutional_layer(no_output_chnl=6, filter_dim=(5, 5), stride=1, pad=2)
        relu1 = ReLU_layer()
        max_o1 = Max_pooling_layer(filter_dim=(2, 2), stride=2, pad=0)
        conv2 = Convolutional_layer(no_output_chnl=16, filter_dim=(5, 5), stride=1, pad=0)
        relu2 = ReLU_layer()
        max_o2 = Max_pooling_layer(filter_dim=(2, 2), stride=2, pad=0)
        flat = Flatten_layer()
        fc1 = Fully_connected_layer()
        relu3 = ReLU_layer()
        fc2 = Fully_connected_layer()
        relu4 = ReLU_layer()
        fc3 = Fully_connected_layer()
        soft = Softmax_layer(no_of_classes)

        for epoch in range(20):

            training_output = np.zeros((len(images_train), no_of_classes))

            batch_size = (16, in_channels, 28, 28)
            
            for i in range(0, len(images_train), batch_size[0]):
                if(len(images_train) - i < batch_size[0]):
                    batch_size = (len(images_train) - i, in_channels, 28, 28)
                # forward pass
                # conv_out = conv.forward_propagation((images_train[i:i+batch_size[0]]).reshape(batch_size))
                # relu_out = relu.forward_propagation(conv_out)
                # max_out = max_o.forward_propagation(relu_out)
                # flat_out = flat.forward_propagation(max_out)
                # fc_out = fc.forward_propagation(flat_out, no_of_classes)
                # soft_out = soft.forward_propagation(fc_out)

                conv1_out = conv1.forward_propagation((images_train[i:i+batch_size[0]]).reshape(batch_size))
                relu1_out = relu1.forward_propagation(conv1_out)
                max_o1_out = max_o1.forward_propagation(relu1_out)
                conv2_out = conv2.forward_propagation(max_o1_out)
                relu2_out = relu2.forward_propagation(conv2_out)
                max_o2_out = max_o2.forward_propagation(relu2_out)
                flat_out = flat.forward_propagation(max_o2_out)
                fc1_out = fc1.forward_propagation(flat_out, 120)
                relu3_out = relu3.forward_propagation(fc1_out)
                fc2_out = fc2.forward_propagation(relu3_out, 84)
                relu4_out = relu4.forward_propagation(fc2_out)
                fc3_out = fc3.forward_propagation(relu4_out, 10)
                soft_out = soft.forward_propagation(fc3_out)

                loss = soft_out - labels_train[i:i+batch_size[0]]
                training_output[i:i+batch_size[0]] = soft_out

                # soft_grad = soft.back_propagation(loss)
                # fc_grad = fc.back_propagation(soft_grad, 0.001)
                # flat_grad = flat.back_propagation(fc_grad)
                # max_grad = max_o.back_propagation(flat_grad)
                # relu_grad = relu.back_propagation(max_grad)
                # conv_grad, conv_grad_filter, conv_grad_bias = conv.back_propagation(relu_grad, 0.001)
                soft_grad = soft.back_propagation(loss)
                fc3_grad = fc3.back_propagation(soft_grad, learning_rate)
                relu4_grad = relu4.back_propagation(fc3_grad)
                fc2_grad = fc2.back_propagation(relu4_grad, learning_rate)
                relu3_grad = relu3.back_propagation(fc2_grad)
                fc1_grad = fc1.back_propagation(relu3_grad, learning_rate)
                flat_grad = flat.back_propagation(fc1_grad)
                max_o2_grad = max_o2.back_propagation(flat_grad)
                relu2_grad = relu2.back_propagation(max_o2_grad)
                conv2_grad, conv2_grad_filter, conv2_grad_bias = conv2.back_propagation(relu2_grad, learning_rate)
                max_o1_grad = max_o1.back_propagation(conv2_grad)
                relu1_grad = relu1.back_propagation(max_o1_grad)
                conv1_grad, conv1_grad_filter, conv1_grad_bias = conv1.back_propagation(relu1_grad, learning_rate)


            # if(epoch%10 == 0):
            print("epoch: ", epoch)
            log_file.write("epoch: " + str(epoch) + "\n")

            # print training accuracy, training loss, validation accuracy, validation loss and macro f1 score by sklearn
            # print(training_output.shape, labels_train.shape)
            print("training accuracy: ", accuracy_score(np.argmax(labels_train, axis=1), np.argmax(training_output, axis=1)))
            print("training loss: ", log_loss(labels_train, training_output))

            log_file.write("training accuracy: " + str(accuracy_score(np.argmax(labels_train, axis=1), np.argmax(training_output, axis=1))) + "\n")
            log_file.write("training loss: " + str(log_loss(labels_train, training_output)) + "\n")
            
            # test on validation set
            batch_size_val = (1, in_channels, 28, 28)

            y_pred = np.zeros((len(images_val), no_of_classes))

            for i in range(0, len(images_val), batch_size_val[0]):
                # conv_out = conv.forward_propagation((images_val[i:i+batch_size_val[0]]).reshape(batch_size_val))
                # relu_out = relu.forward_propagation(conv_out)
                # max_out = max_o.forward_propagation(relu_out)
                # flat_out = flat.forward_propagation(max_out)
                # fc_out = fc.forward_propagation(flat_out, no_of_classes)
                # soft_out = soft.forward_propagation(fc_out)

                conv1_out = conv1.forward_propagation((images_val[i:i+batch_size_val[0]]).reshape(batch_size_val))
                relu1_out = relu1.forward_propagation(conv1_out)
                max_o1_out = max_o1.forward_propagation(relu1_out)
                conv2_out = conv2.forward_propagation(max_o1_out)
                relu2_out = relu2.forward_propagation(conv2_out)
                max_o2_out = max_o2.forward_propagation(relu2_out)
                flat_out = flat.forward_propagation(max_o2_out)
                fc1_out = fc1.forward_propagation(flat_out, 120)
                relu3_out = relu3.forward_propagation(fc1_out)
                fc2_out = fc2.forward_propagation(relu3_out, 84)
                relu4_out = relu4.forward_propagation(fc2_out)
                fc3_out = fc3.forward_propagation(relu4_out, 10)
                soft_out = soft.forward_propagation(fc3_out)

                y_pred[i] = soft_out

            print("validation accuracy: ", accuracy_score(np.argmax(labels_val, axis=1), np.argmax(y_pred, axis=1)))
            # accuracy = np.sum(np.argmax(soft_out, axis = 1) == np.argmax(labels_val, axis = 1)) / len(labels_val)
            # print("validation accuracy: ", accuracy)
            print("validation loss: ", log_loss(labels_val, y_pred))
            print("macro f1 score: ", f1_score(np.argmax(labels_val, axis=1), np.argmax(y_pred, axis=1) , average='macro'))
            print()

            log_file.write("validation accuracy: " + str(accuracy_score(np.argmax(labels_val, axis=1), np.argmax(y_pred, axis=1))) + "\n")
            log_file.write("validation loss: " + str(log_loss(labels_val, y_pred)) + "\n")
            log_file.write("macro f1 score: " + str(f1_score(np.argmax(labels_val, axis=1), np.argmax(y_pred, axis=1) , average='macro')) + "\n")
            log_file.write("\n")


        # save the weights and biases in a pickle file
        conv1_filters = conv1.get_filters()
        conv1_bias = conv1.get_bias()
        conv2_filters = conv2.get_filters()
        conv2_bias = conv2.get_bias()
        fc1_filters = fc1.get_filters()
        fc1_bias = fc1.get_bias()
        fc2_filters = fc2.get_filters()
        fc2_bias = fc2.get_bias()
        fc3_filters = fc3.get_filters()
        fc3_bias = fc3.get_bias()

        with open('weights_'+str(num_samples)+'_'+str(learning_rate)+'_epochs_15.pkl', 'wb') as f:
            pickle.dump([conv1_filters, conv1_bias, conv2_filters, conv2_bias, fc1_filters, fc1_bias, fc2_filters, fc2_bias, fc3_filters, fc3_bias], f)

    '''
    # Read the weights and biases from the pickle file
    with open('weights.pkl', 'rb') as f:
        conv1_filters, conv1_bias, conv2_filters, conv2_bias, fc1_filters, fc1_bias, fc2_filters, fc2_bias, fc3_filters, fc3_bias = pickle.load(f)


    # print('conv_grad_filter: ')
    # conv_filters = conv.get_filters()
    # print(conv_filters)
    # print('conv_grad_bias: ')
    # conv_bias = conv.get_bias()
    # print(conv_bias)
    # # print('fc_grad: ')
    # # fc_filters = fc.get_filters()
    # # print(fc_filters)
    # print('fc_grad_bias: ')
    # fc_bias = fc.get_bias()
    # print(fc_bias)

    batch_size = (1, in_channels, 28, 28)

    y_pred = np.zeros(len(images_test))

    for i in range(len(images_test)):
        # forward pass
        # conv_out = conv.forward_propagation((images_test[i:i+batch_size[0]]).reshape(batch_size))
        # relu_out = relu.forward_propagation(conv_out)
        # max_out = max_o.forward_propagation(relu_out)
        # flat_out = flat.forward_propagation(max_out)
        # fc_out = fc.forward_propagation(flat_out, no_of_classes)
        # soft_out = soft.forward_propagation(fc_out)

        conv1_out = conv1.forward_propagation((images_test[i:i+batch_size[0]]).reshape(batch_size))
        relu1_out = relu1.forward_propagation(conv1_out)
        max_o1_out = max_o1.forward_propagation(relu1_out)
        conv2_out = conv2.forward_propagation(max_o1_out)
        relu2_out = relu2.forward_propagation(conv2_out)
        max_o2_out = max_o2.forward_propagation(relu2_out)
        flat_out = flat.forward_propagation(max_o2_out)
        fc1_out = fc1.forward_propagation(flat_out, 120)
        relu3_out = relu3.forward_propagation(fc1_out)
        fc2_out = fc2.forward_propagation(relu3_out, 84)
        relu4_out = relu4.forward_propagation(fc2_out)
        fc3_out = fc3.forward_propagation(relu4_out, 10)
        soft_out = soft.forward_propagation(fc3_out)

        print('predicted: ', np.argmax(soft_out))
        print('actual: ', np.argmax(labels_test[i]))

        y_pred[i] = np.argmax(soft_out)

    # print how many images are correctly classified
    print('correctly classified: ', np.where(y_pred == np.argmax(labels_test, axis=1))[0].shape[0])

    accuracy = np.sum(y_pred == np.argmax(labels_test, axis = 1)) / len(labels_test)
    print('accuracy: ', accuracy*100)

    '''

if __name__ == '__main__':
    main()