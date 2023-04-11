import numpy as np
import pandas as pd
import cv2
import os
import pickle
import PIL
from PIL import Image, ImageFilter
import os
import matplotlib.pyplot as plt
import time
import seaborn as sns
import csv

from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, ConfusionMatrixDisplay

#########   Utility functions   #########

if(os.path.exists('log.txt')):
    os.remove('log.txt')
log_file = open('log.txt', 'w')

def preprocess_image(input_image):

    # convert to grayscale
    image = input_image.convert('L')
    # invert the image
    image = PIL.ImageOps.invert(image)
    # dilate the image
    image = image.filter(ImageFilter.MaxFilter(3))
    # # Apply erosion
    # image = image.filter(ImageFilter.MinFilter(3))
    # # Apply thresholding
    # image = image.point(lambda x: 0 if x < 100 else 255, '1')
    # # increase the contrast
    # image = image.point(lambda x: 0 if x < 200 else 255, '1')
    # resize the image
    # image = image.resize((16, 16))
    image = image.resize((28, 28))
    # convert to numpy array
    image = np.array(image)
    # normalize the image
    image = image / 255.0

    return image


def load_training_data():

    folder_paths_data = ['Dataset/NumtaDB_with_aug/training-a','Dataset/NumtaDB_with_aug/training-b','Dataset/NumtaDB_with_aug/training-c']
    # folder_paths = ['Dataset/NumtaDB_with_aug/training-b']
    folder_paths_label = ['Dataset/NumtaDB_with_aug/training-a.csv','Dataset/NumtaDB_with_aug/training-b.csv','Dataset/NumtaDB_with_aug/training-c.csv']
    # folder_paths = ['Dataset/NumtaDB_with_aug/training-b.csv']
    
    # image_paths = np.load('image_paths_train.npy', allow_pickle=True)
    # print(len(image_paths))
    # print(image_paths[-1])

    image_paths_train = []
    images = []

    for folder_path in folder_paths_data:
        image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        for image_path in image_paths:
            image = Image.open(image_path)
            image = preprocess_image(image)
            images.append(image)
            image_paths_train.append(image_path)

    np.save('image_paths_train.npy', image_paths_train)

    # image_paths_train = np.load('image_paths_train.npy', allow_pickle=True)
    # print(len(image_paths_train))
    # get the labels from the csv file

    labels = []
    for folder_path in folder_paths_label:
        df = pd.read_csv(folder_path)
        labels.extend(df['digit'])

    print('Training data loaded successfully')
    print('Number of training images: ', len(images))
    print('Shape of training images: ', images[0].shape)
    print('Number of training labels: ', len(labels))

    # save the images and labels
    np.save('images_train.npy', images)
    np.save('labels_train.npy', labels)

    training_data = np.load('images_train.npy', allow_pickle=True)
    training_label = np.load('labels_train.npy', allow_pickle=True)

    return training_data, training_label


def train_validation_split(split_factor, num_sample, training_data, training_label):

    # shuffle the data
    random_indices = np.random.choice(len(training_data), num_sample, replace=False)
    training_data_full = training_data[random_indices]
    training_label_full = training_label[random_indices]

    split_index = int(split_factor * len(training_data_full))

    training_data = training_data_full[:split_index]
    training_label = training_label_full[:split_index]

    validation_data = training_data_full[split_index:]
    validation_label = training_label_full[split_index:]

    print('Training data size: ', len(training_data))
    log_file.write('Training data size: ' + str(len(training_data)) + '\n')
    print('Validation data size: ', len(validation_data))
    log_file.write('Validation data size: ' + str(len(validation_data)) + '\n')

    return training_data, training_label, validation_data, validation_label
    
    
def load_dataset(split_factor = 0.9, num_sample = 0, no_of_classes = 10, load_from_file = False):

    print('Loading training data...')

    if load_from_file:
        training_data = np.load('images_train.npy')
        training_label = np.load('labels_train.npy')
    else:
        training_data, training_label = load_training_data()

    print('Training data loaded successfully')
    print('Number of training images: ', len(training_data))

    if num_sample == 0:
        num_sample = len(training_data)

    training_data, training_label, validation_data, validation_label = train_validation_split(split_factor, num_sample, training_data, training_label)

    # Standard normalization
    training_data = (training_data - np.mean(training_data)) / np.std(training_data)
    validation_data = (validation_data - np.mean(validation_data)) / np.std(validation_data)

    # One hot encoding
    training_label = np.eye(no_of_classes)[training_label]
    validation_label = np.eye(no_of_classes)[validation_label]

    return training_data, training_label, validation_data, validation_label


def load_test_data(path, for_predict = False):

    folder_paths_data = [path]
    folder_paths_label = [path + '.csv']

    for folder_path in folder_paths_data:
        image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))] ## changed here
    np.save('image_paths_test.npy', image_paths)

    images = []
    filenames = []

    for image_path in image_paths:

        # extract the filename
        filename = image_path.split('\\')[-1]
        filenames.append(filename)

        image = Image.open(image_path)
        image = preprocess_image(image)
        images.append(image)
 
    # get the labels from the csv file
    labels = []

    if(for_predict == False):
        for folder_path in folder_paths_label:
            df = pd.read_csv(folder_path)
            labels.extend(df['digit'])
            # filenames.extend(df['filename'])

    print('Test data loaded successfully')
    print('Number of test images: ', len(images))
    print('Shape of test images: ', images[0].shape)
    print('Number of test labels: ', len(labels))
    print('Number of test filenames: ', len(filenames))

    # save the images and labels
    np.save('images_test.npy', images)
    np.save('labels_test.npy', labels)
    np.save('filenames_test.npy', filenames)

    test_data = np.load('images_test.npy', allow_pickle=True)
    test_label = np.load('labels_test.npy', allow_pickle=True)
    test_filenames = np.load('filenames_test.npy', allow_pickle=True)

    return test_data, test_label, test_filenames


def load_dataset_test(path, num_sample = 0, no_of_classes = 10, load_from_file = False, for_predict = False):

    print('Loading test data...')

    if load_from_file:
        test_data = np.load('images_test.npy')
        if for_predict == False:
            test_label = np.load('labels_test.npy')
        test_filenames = np.load('filenames_test.npy')
    else:
        test_data, test_label, test_filenames = load_test_data(path, for_predict)

    print('Test data loaded successfully')
    print('Number of test images: ', len(test_data))

    if num_sample == 0:
        num_sample = len(test_data)

    # print(test_filenames[-1])

    # test_data = test_data[:num_sample]
    # if not for_predict :
    #     test_label = test_label[:num_sample]
    # test_filenames = test_filenames[:num_sample]

    random_indices = np.random.choice(len(test_data), num_sample, replace=False)
    test_data = test_data[random_indices]
    if not for_predict :
        test_label = test_label[random_indices]
    test_filenames = test_filenames[random_indices]

    # Standard normalization
    test_data = (test_data - np.mean(test_data)) / np.std(test_data)

    # One hot encoding
    if not for_predict :
        test_label = np.eye(no_of_classes)[test_label]

    return test_data, test_label, test_filenames


def modify_gradients(gradients, overflow_threshold=1000.0, underflow_threshold=1e-15): #changed here

    overflow_indices = np.where(np.abs(gradients) > overflow_threshold)
    gradients[overflow_indices] = np.sign(gradients[overflow_indices]) * overflow_threshold

    zero_indices = np.where(np.abs(gradients) == 0)
    underflow_indices = np.where(np.abs(gradients) < underflow_threshold)
    gradients[underflow_indices] = np.sign(gradients[underflow_indices]) * underflow_threshold
    gradients[zero_indices] = 1e-3

    gradients = [np.nan_to_num(gradient) for gradient in gradients]

    return gradients


########## Parent Model Class to handle all Component related functions ##########

class Convolutional_neural_network:

    def __init__(self):

        self.layers = []
        self.no_of_parameters = 0


    def set_parameters(self, no_of_classes = 10, learning_rate = 0.001, samples_in_batch = 16, epochs = 20, in_channels = 1):

        self.no_of_classes = no_of_classes
        self.learning_rate = learning_rate
        self.samples_in_batch = samples_in_batch
        self.epochs = epochs
        self.num_samples = 0
        self.batch_size_train = (0,0,0,0)
        self.batch_size_val = (0,0,0,0)
        self.in_channels = in_channels
    

    def add_component(self, component):

        self.layers.append(component)


    def build_architecture(self):

        conv_layer1 = Convolutional_layer(no_output_chnl=6, filter_dim=(5, 5), stride=1, pad=2)
        self.add_component(conv_layer1)

        relu_layer1 = ReLU_layer()
        self.add_component(relu_layer1)

        maxpool_layer1 = Max_pooling_layer(pool_dim=(2, 2), stride=2, pad=0)
        self.add_component(maxpool_layer1)

        conv_layer2 = Convolutional_layer(no_output_chnl=16, filter_dim=(5, 5), stride=1, pad=0)
        self.add_component(conv_layer2)

        relu_layer2 = ReLU_layer()
        self.add_component(relu_layer2)

        maxpool_layer2 = Max_pooling_layer(pool_dim=(2, 2), stride=2, pad=0)
        self.add_component(maxpool_layer2)
        
        flatten_layer = Flatten_layer()
        self.add_component(flatten_layer)

        fc_layer1 = Fully_connected_layer(no_output_nodes=120)
        self.add_component(fc_layer1)

        relu_layer3 = ReLU_layer()
        self.add_component(relu_layer3)

        fc_layer2 = Fully_connected_layer(no_output_nodes=84)
        self.add_component(fc_layer2)

        relu_layer4 = ReLU_layer()
        self.add_component(relu_layer4)

        fc_layer3 = Fully_connected_layer(no_output_nodes=self.no_of_classes)
        self.add_component(fc_layer3)

        softmax_layer = Softmax_layer(self.no_of_classes)
        self.add_component(softmax_layer)


    def forward_propagation(self, input):

        output = input
        for layer in self.layers:
            output = layer.forward_propagation(output)
        return output


    def backward_propagation(self, previous_gradient, learning_rate):

        for layer in reversed(self.layers):
            previous_gradient = layer.backward_propagation(previous_gradient, learning_rate)
        return previous_gradient


    def train(self, training_data, training_label, validation_data, validation_label):

        self.num_samples = len(training_data)

        for epoch in range(self.epochs):

            self.batch_size_train = (self.samples_in_batch, self.in_channels, training_data.shape[1], training_data.shape[2])
            self.batch_size_val = (1, self.in_channels, validation_data.shape[1], validation_data.shape[2])

            training_output = np.zeros((self.num_samples, self.no_of_classes))

            for i in range(0, self.num_samples, self.samples_in_batch):

                if(self.num_samples - i < self.samples_in_batch):
                    self.batch_size_train = (self.num_samples - i, self.in_channels, training_data.shape[1], training_data.shape[2])

                training_batch = training_data[i:i+self.batch_size_train[0]].reshape(self.batch_size_train)

                batch_out = self.forward_propagation(training_batch)

                loss = batch_out - training_label[i:i+self.batch_size_train[0]]

                training_output[i:i+self.batch_size_train[0]] = batch_out

                self.backward_propagation(loss, self.learning_rate)

            print("Epoch:", epoch+1)
            log_file.write("Epoch: " + str(epoch+1) + "\nn")

            print("Training Accuracy:", accuracy_score(np.argmax(training_label, axis=1), np.argmax(training_output, axis=1)))
            log_file.write("Training Accuracy: " + str(accuracy_score(np.argmax(training_label, axis=1), np.argmax(training_output, axis=1))) + "\n")

            print("Training loss: ", log_loss(training_label, training_output))
            log_file.write("Training loss: " + str(log_loss(training_label, training_output)) + "\n")

            self.validate(validation_data, validation_label, self.batch_size_val)

        print("Training Done")
        log_file.write("Training Done\n")


    def validate(self, validation_data, validation_label, batch_size_val):


        validation_output = np.zeros((len(validation_data), self.no_of_classes))

        for i in range(0, len(validation_data), batch_size_val[0]):
            if(len(validation_data) - i < batch_size_val[0]):
                batch_size_val = (len(validation_data) - i, self.in_channels, validation_data.shape[1], validation_data.shape[2])

            validation_batch = validation_data[i:i+self.batch_size_val[0]].reshape(self.batch_size_val)

            batch_out = self.forward_propagation(validation_batch)

            validation_output[i:i+self.batch_size_val[0]] = batch_out

        print("Validation Accuracy:", accuracy_score(np.argmax(validation_label, axis=1), np.argmax(validation_output, axis=1)))
        log_file.write("Validation Accuracy: " + str(accuracy_score(np.argmax(validation_label, axis=1), np.argmax(validation_output, axis=1))) + "\n")

        print("Validation loss: ", log_loss(validation_label, validation_output))
        log_file.write("Validation loss: " + str(log_loss(validation_label, validation_output)) + "\n")

        print("Macro F1 score: ", f1_score(np.argmax(validation_label, axis=1), np.argmax(validation_output, axis=1), average='macro'))
        log_file.write("Macro F1 score: " + str(f1_score(np.argmax(validation_label, axis=1), np.argmax(validation_output, axis=1), average='macro')) + "\n")

        print()
        log_file.write("\n")


    def save_model(self):

        parameters = []

        for layer in self.layers:
            if(layer.has_params):
                filters, bias = layer.get_params()
                parameters.append(filters)
                parameters.append(bias)

        self.no_of_parameters = len(parameters)
        print("No of parameters: ", self.no_of_parameters)

        # with open('1705071_model_'+str(self.learning_rate)+'.pickle', 'wb') as f:
        with open('1705071_model.pickle', 'wb') as f:
            pickle.dump(parameters, f)


    def load_model(self):

        with open('1705071_model.pickle', 'rb') as f:
            parameters = pickle.load(f)

        print("No of parameters: ", len(parameters))

        for layer in self.layers:
            if(layer.has_params):
                layer.set_params(parameters.pop(0), parameters.pop(0))
                # print("No of parameters left: ", len(parameters))


    def test(self, test_data, test_label, test_filename, path):

        # log_file = open("1705071_test_log.txt", "w")

        print("Testing...")
        batch_size_test = (1, self.in_channels, test_data.shape[1], test_data.shape[2])

        test_output = np.zeros((len(test_data), self.no_of_classes))

        for i in range(0, len(test_data), batch_size_test[0]):
            if(len(test_data) - i < batch_size_test[0]):
                batch_size_test = (len(test_data) - i, self.in_channels, test_data.shape[1], test_data.shape[2])

            test_batch = test_data[i:i+batch_size_test[0]].reshape(batch_size_test)

            batch_out = self.forward_propagation(test_batch)

            test_output[i:i+batch_size_test[0]] = batch_out

        print("Test Accuracy:", accuracy_score(np.argmax(test_label, axis=1), np.argmax(test_output, axis=1)))
        log_file.write("Test Accuracy: " + str(accuracy_score(np.argmax(test_label, axis=1), np.argmax(test_output, axis=1))) + "\n")

        print("Macro F1 score: ", f1_score(np.argmax(test_label, axis=1), np.argmax(test_output, axis=1), average='macro'))
        log_file.write("Macro F1 score: " + str(f1_score(np.argmax(test_label, axis=1), np.argmax(test_output, axis=1), average='macro')) + "\n")

        print()
        log_file.write("\n")

        # confusion matrix
        cm = confusion_matrix(np.argmax(test_label, axis=1), np.argmax(test_output, axis=1))
        
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', cbar=False, xticklabels=np.arange(10), yticklabels=np.arange(10), annot_kws={'size': 5})
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix for lr = ' + str(self.learning_rate) + ', epoch = ' + str(self.epochs) + ', batch_size = ' + str(self.samples_in_batch))
        # plt.title('Confusion Matrix for lr = 0.03 epoch = 20 batch_size = 16')
        plt.savefig('1705071_confusion_matrix.png')
        plt.show()


    def predict(self, test_data, test_filename, path):

        print("Predicting...")

        batch_size_test = (1, self.in_channels, test_data.shape[1], test_data.shape[2])

        test_output = np.zeros((len(test_data), self.no_of_classes))

        for i in range(0, len(test_data), batch_size_test[0]):
            if(len(test_data) - i < batch_size_test[0]):
                batch_size_test = (len(test_data) - i, self.in_channels, test_data.shape[1], test_data.shape[2])

            test_batch = test_data[i:i+batch_size_test[0]].reshape(batch_size_test)

            batch_out = self.forward_propagation(test_batch)

            test_output[i:i+batch_size_test[0]] = batch_out

        # write the test filenames and corresponding predicted labels in a csv file
        # print(path)
        csv_filename = path+ '/1705071_prediction.csv'
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['FileName', 'Digit'])
            for i in range(len(test_filename)):
                writer.writerow([test_filename[i], np.argmax(test_output[i])])

        

############# Component Classes #############

############# Convolutional Layer #############

class Convolutional_layer:

    def __init__(self, no_output_chnl, filter_dim, stride, pad):

        self.no_output_chnl = no_output_chnl
        self.filter_dim = filter_dim
        self.filter_height = filter_dim[0]
        self.filter_width = filter_dim[1]
        self.stride = stride
        self.pad = pad
        self.filters = None
        self.bias = None
        
        self.has_params = True

    def get_all_windows(self, input, output_dim, stride):

        # extract necessary information from the input and output_dim
        batch_strides, channel_strides, height_strides, width_strides = input.strides

        batch_size, input_channels, input_height, input_width = input.shape

        output_height, output_width = output_dim[2], output_dim[3]

        return np.lib.stride_tricks.as_strided(input,
        (batch_size, input_channels, output_height, output_width, self.filter_height, self.filter_width),
        (batch_strides, channel_strides, height_strides * stride, width_strides * stride, height_strides, width_strides))

    def dilate_input(self, input):

        dilated_input = input
        dilated_input = np.insert(dilated_input, range(1, input.shape[2]), 0, axis=2)
        dilated_input = np.insert(dilated_input, range(1, input.shape[3]), 0, axis=3)
        return dilated_input

    def forward_propagation(self, input):

        self.input_shape = input.shape
        self.batch_size = input.shape[0]
        self.input_channels = input.shape[1]
        self.input_height = input.shape[2]
        self.input_width = input.shape[3]

        # initialize the filters and bias
        if self.filters is None:
            self.filters = np.random.randn(self.no_output_chnl, self.input_channels, self.filter_height, self.filter_width) * np.sqrt(2 / (self.input_channels * self.filter_height * self.filter_width))
            self.bias = np.zeros(self.no_output_chnl) * np.sqrt(2 / (self.input_channels * self.filter_height * self.filter_width))

        # calculate the output shape with padding and stride
        self.output_dim = (self.batch_size,
                        self.no_output_chnl,
                        int((self.input_height - self.filter_height + 2 * self.pad) / self.stride + 1),
                        int((self.input_width - self.filter_width + 2 * self.pad) / self.stride + 1))

        # initialize the output
        output = np.zeros(self.output_dim)

        # apply padding to the input
        input = np.pad(input, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), 'constant')

        # get all the windows
        self.all_input_windows = self.get_all_windows(input, self.output_dim, self.stride)

        # use the windows to calculate the output
        output = np.einsum('bihwkl,oikl->bohw', self.all_input_windows, self.filters) + self.bias[None, :, None, None]

        return output

    def backward_propagation(self, previous_gradient, learning_rate):
        
        # initialize the gradient of the filters and bias
        filters_gradient = np.zeros(self.filters.shape)
        bias_gradient = np.zeros(self.bias.shape)

        # calculate the gradient of the filters
        filters_gradient = np.einsum('bihwkl,bohw->oikl', self.all_input_windows, previous_gradient)

        # calculate the gradient of the bias
        bias_gradient = previous_gradient.sum(axis=(0, 2, 3)) # changed here

        # calculate the gradient of the input

        # dilate the previous gradient
        dilated_previous_gradient = self.dilate_input(previous_gradient)

        # pad the dilated previous gradient
        if self.pad != 0:
            pad_height = self.pad
            pad_width = self.pad
        else:
            pad_height = self.filter_height - 1
            pad_width = self.filter_width - 1

        dilated_padded_previous_gradient = np.pad(dilated_previous_gradient, ((0, 0), (0, 0), (pad_height, pad_height), (pad_width, pad_width)), 'constant')

        # get all the windows
        all_dilated_padded_previous_gradient_windows = self.get_all_windows(input=dilated_padded_previous_gradient, output_dim=self.input_shape, stride=1)

        # rotate the filters by 180 degrees
        rotated_filters = np.rot90(self.filters, 2, (2, 3))

        # use the windows to calculate the gradient of the input
        input_gradient = np.einsum('oikl,bohwkl->bihw', rotated_filters, all_dilated_padded_previous_gradient_windows)

        # modify the input gradient
        input_gradient = np.copy(modify_gradients(gradients=input_gradient))

        # update the filters and bias
        self.filters -= learning_rate * filters_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient

    def get_params(self):

        return self.filters, self.bias

    def set_params(self, filters, bias):

        self.filters = filters
        self.bias = bias

############# ReLU Layer #############

class ReLU_layer:

    def __init__(self):

        self.mask = None

        self.has_params = False

    def forward_propagation(self, input):
        self.mask = (input <= 0)
        output = input.copy()
        output[self.mask] = 0

        return output

    def backward_propagation(self, previous_gradient, learning_rate):

        previous_gradient[self.mask] = 0

        return previous_gradient

    def get_params(self):
        
        return None

    def set_params(self, filters, bias):

        pass

############# Max Pooling Layer #############

class Max_pooling_layer:

    def __init__(self, pool_dim, stride, pad):
        self.pool_dim = pool_dim
        self.pool_height = pool_dim[0]
        self.pool_width = pool_dim[1]
        self.stride = stride
        self.pad = pad
        self.max_indices = None

        self.has_params = False

    def get_all_windows(self, input, output_dim, stride):

        # extract necessary information from the input and output_dim
        batch_strides, channel_strides, height_strides, width_strides = input.strides

        batch_size, input_channels, input_height, input_width = input.shape

        output_height, output_width = output_dim[2], output_dim[3]

        return np.lib.stride_tricks.as_strided(input,
        (batch_size, input_channels, output_height, output_width, self.pool_height, self.pool_width),
        (batch_strides, channel_strides, height_strides * stride, width_strides * stride, height_strides, width_strides))

    def forward_propagation(self, input):

        self.input_shape = input.shape

        self.batch_size = input.shape[0]
        self.input_channels = input.shape[1]
        self.input_height = input.shape[2]
        self.input_width = input.shape[3]

        # calculate the output shape with padding and stride
        self.output_dim = (self.batch_size,
                        self.input_channels,
                        int((self.input_height - self.pool_height + 2 * self.pad) / self.stride) + 1,
                        int((self.input_width - self.pool_width + 2 * self.pad) / self.stride) + 1)

        # initialize the output
        output = np.zeros(self.output_dim)

        # apply padding to the input
        input = np.pad(input, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), 'constant')

        # get all the windows
        self.all_input_windows = self.get_all_windows(input, self.output_dim, self.stride)
        output = self.all_input_windows.max(axis=(4, 5))

        # initialize the max_indices
        self.max_indices = np.zeros(self.output_dim).astype(int)

        # get the indices of the max values
        self.out_batch, self.out_channel, self.out_height, self.out_width = self.output_dim

        for batch in range(self.out_batch):
            for chnl in range(self.out_channel):
                for hght in range(self.out_height):
                    for wdth in range(self.out_width):
                        self.max_indices[batch, chnl, hght, wdth] = np.argmax(input[batch, chnl, hght * self.stride: hght * self.stride + self.pool_height, wdth * self.stride: wdth * self.stride + self.pool_width])

        return output

    def backward_propagation(self, previous_gradient, learning_rate):
            
            # initialize the input gradient
            input_gradient = np.zeros(self.input_shape)
    
            # get the indices of the max values
            self.out_batch, self.out_channel, self.out_height, self.out_width = self.output_dim
    
            for batch in range(self.out_batch):
                for chnl in range(self.out_channel):
                    for hght in range(self.out_height):
                        for wdth in range(self.out_width):
                            # check if we can place the pool filter at the current position
                            if hght * self.stride + self.pool_height <= self.input_height and wdth * self.stride + self.pool_width <= self.input_width:
                                input_gradient[batch, chnl, hght * self.stride: hght * self.stride + self.pool_height, wdth * self.stride: wdth * self.stride + self.pool_width] = np.zeros((self.pool_height, self.pool_width))
                                input_gradient[batch, chnl, hght * self.stride: hght * self.stride + self.pool_height, wdth * self.stride: wdth * self.stride + self.pool_width].flat[int(self.max_indices[batch, chnl, hght, wdth])] = previous_gradient[batch, chnl, hght, wdth]
    
            return input_gradient

    def get_params(self):
            
        return None

    def set_params(self, filters, bias):

        pass

############# Flatten Layer #############

class Flatten_layer:

    def __init__(self):

        self.has_params = False

    def forward_propagation(self, input):

        self.input_shape = input.shape

        return input.reshape(input.shape[0], -1)

    def backward_propagation(self, previous_gradient, learning_rate):

        return previous_gradient.reshape(self.input_shape)

    def get_params(self):
        
        return None

    def set_params(self, filters, bias):

        pass

############# Fully Connected Layer #############

class Fully_connected_layer:

    def __init__(self, no_output_nodes):

        self.filters = None
        self.bias = None
        self.no_output_nodes = no_output_nodes

        self.has_params = True

    def forward_propagation(self, input):

        self.input = input
        self.no_input_channels = input.shape[1]

        if self.filters is None:
            self.filters = np.random.randn(self.no_input_channels, self.no_output_nodes) * np.sqrt(2 / self.no_input_channels)
            
            ###_changed_here
            self.bias = np.random.randn(self.no_output_nodes) * np.sqrt(2 / self.no_input_channels)

        output = input.dot(self.filters) + self.bias
        return output

    def backward_propagation(self, previous_gradient, learning_rate):

        # calculate the gradient of the filters
        filters_gradient = self.input.T.dot(previous_gradient)

        # calculate the gradient of the bias
        bias_gradient = previous_gradient.sum(axis=0)  # changed here

        # calculate the gradient of the input
        input_gradient = previous_gradient.dot(self.filters.T)

        # modify the input gradient
        input_gradient = np.copy(modify_gradients(input_gradient))

        # update the filters and bias
        self.filters -= learning_rate * filters_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient

    def get_params(self):
        
        return self.filters, self.bias

    def set_params(self, filters, bias):

        self.filters = filters
        self.bias = bias

############# Softmax Layer #############

class Softmax_layer:

    def __init__(self, no_of_classes):

        self.no_of_classes = no_of_classes

        self.has_params = False

    def forward_propagation(self, input):

        # get unnormalized probabilities
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))

        # normalize them for each sample
        softmax_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        return softmax_values

    def backward_propagation(self, previous_gradient, learning_rate):

        return np.copy(previous_gradient)

    def get_params(self):
        
        return None

    def set_params(self, filters, bias):

        pass


############# main function #############

def main():

    # learning_rates = [0.0005, 0.0008, 0.001, 0.003, 0.005, 0.008, 0.1, 0.5]
    # learning_rates = [0.001, 0.0001, 0.0009, 0.0003, 0.005, 0.003, 0.01, 0.03, 0.0005, 0.0008, 0.008, 0.05]
    learning_rates = [0.001]

    for learning_rate in learning_rates:

        print("Learning rate: ", learning_rate)
        log_file.write("Learning rate: " + str(learning_rate) + "\n")

        # Load the dataset 
        x_train, y_train, x_validation, y_validation = load_dataset(num_sample = 0, load_from_file = True, split_factor = 0.95)

        # Create the model
        cnn = Convolutional_neural_network()

        # set parameters
        cnn.set_parameters(no_of_classes=10, learning_rate=learning_rate, samples_in_batch=16, epochs=20, in_channels = 1)

        # build the model
        cnn.build_architecture()

        # train the model
        cnn.train(x_train, y_train, x_validation, y_validation)

        # save the model
        cnn.save_model()

if __name__ == '__main__':
    main()