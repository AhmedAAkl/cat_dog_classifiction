# A keras implementation to solve cats vs dogs classification problem using CNN 


This is a simple classification model to solve the problem of an input image is a cat or a dog using CNN.
The model consists of 4 convolutional layers each one followed by a max pooling layer and finally 2 fully connected layers.
The convolutional layers use a filter size 3 × 3 and pooling layer use 2 ×  2 which reduces the size to 1/2. 

The model validation accuracy nearly equal 89%

#Tools for improving CNN performance
The following techniques are employed to imporve performance of CNN.

Train

1. Data augmentation
The number of train-data is increased to 5 times by means of

Random rotation : each image is rotated by random degree.
Random shift : each image is randomly shifted by a value.
Zero-centered normalization : a pixel value is subtracted by (PIXEL_DEPTH/2) and divided by PIXEL_DEPTH.

2. Parameter initializers
Weight initializer : xaiver initializer
Bias initializer : constant (zero) initializer

3. Batch normalization
All convolution/fully-connected layers use batch normalization.

4. Dropout
The third fully-connected layer employes dropout technique.

5. Exponentially decayed learning rate
A learning rate is decayed every after one-epoch.

Test
1. Ensemble prediction
Every model makes a prediction (votes) for each test instance and the final output prediction is the one that receives the highest number of votes.
