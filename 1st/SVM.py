import random

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as ppl
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False, reshape = False, validation_size = 0)

train_images = mnist.train.images
train_labels = mnist.train.labels
test_images = mnist.test.images
test_labels = mnist.test.labels
val_images = mnist.validation.images
val_labels = mnist.validation.labels

print(datetime.now())

clf = svm.SVC()

test_dataset_size = len (test_images)
train_dataset_size = len(train_images)

clf.fit (train_images.reshape(train_dataset_size, -1), train_labels)
array = clf.predict(test_images.reshape(test_dataset_size, -1))


n_mistakes = 0
i = 0
array_length = len(array)
for element in array:
    if test_labels[i] != array[i]:
         n_mistakes = n_mistakes + 1
    i = i + 1
print (array_length)
print (n_mistakes)
accuracy = (array_length - n_mistakes) / (n_mistakes)
print('accuracy: ', accuracy)

print(datetime.now())


train_images.shape, train_labels.shape, test_images.shape, test_labels.shape, val_images.shape, val_labels.shape

shape = (4, 10)
width = 15
height = 7
ppl.figure(figsize=(width, height))
for i in range(shape[0] * shape[1]):
    idx = random.randint(0, train_images.shape[0])
    img = ppl.subplot(shape[0], shape[1], i + 1)
    img.imshow(train_images[idx].reshape((28, 28)), cmap='gray')
    img.set_title(train_labels[idx], )
    img.axes.get_yaxis().set_visible(False)
    img.axes.get_xaxis().set_visible(False)

ppl.hist(train_labels)
ppl.savefig('1.svg')
ppl.hist(test_labels)
ppl.savefig('2.svg')
