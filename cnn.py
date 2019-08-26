import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

## Reading the whole genes dataset
# print("Reading data ...")
# df = pd.read_csv('notebooks/data/cleaned.csv')
# print("Getting genes from data ...")
# genes_indexes = df.columns[df.columns.str.startswith('ENSG')]
# genes = df[genes_indexes]
# labels = df['tumor_stage']

## Reading only 900 genes
# print("Reading data ...")
# df = pd.read_csv('notebooks/data/small_gens_cleaned.csv')
# genes_indexes = df.columns[df.columns.str.startswith('ENSG')]
# genes = df[genes_indexes]
# labels = df['tumor_stage']

# Reading 69 genes - from literature
print("Reading data ...")
df = pd.read_csv('notebooks/data/64_genes.csv')
genes_indexes = df.columns[df.columns.str.startswith('ENSG')]
genes = df[genes_indexes]
labels = df['tumor_stage']

image_gene_size = 8
# X_train, X_test, y_train, y_test = train_test_split(genes, labels, test_size=0.2, random_state=42)
np.random.seed(10)
random_indices = np.random.choice(genes.index, size=100, replace=False)
X_train, X_test, y_train, y_test = genes[~genes.index.isin(random_indices)], \
                                   genes[genes.index.isin(random_indices)], \
                                   labels[~labels.index.isin(random_indices)], \
                                   labels[labels.index.isin(random_indices)]

print('Training shape: ', X_train.shape)
print('Test shape: ', X_test.shape)
print('Training labels shape: ', y_train.shape)
print('Shape of a genes image: ', X_train.iloc[0].shape)
print('Example label: ', y_train.iloc[0])


# # Review a few images
# gene_list = X_train.iloc[0:9]
# gene_list_labels = y_train.iloc[0:9]
# fig = plt.figure(1, (5., 5.))
# grid = ImageGrid(fig, 111,  # similar to subplot(111)
#                  nrows_ncols=(3, 3),  # creates 2x2 grid of axes
#                  axes_pad=0.3,  # pad between axes in inch.
#                  )
#
# for i in range(len(gene_list)):
#     image = np.array(gene_list.iloc[i]).reshape(30, -1)
#     grid[i].imshow(image)
#     grid[i].set_title('Label: {0}'.format(gene_list_labels.iloc[i]))
#
# plt.show()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


sess = tf.InteractiveSession()

# Create placeholders nodes for images and label inputs
x = tf.placeholder(tf.float32, shape=[None, genes.shape[1]])
y = tf.placeholder(tf.float32, shape=[None, labels.nunique()])

# x_image = tf.reshape(x, [-1, image_gene_size, image_gene_size, 1])

# CONV layer
# W1 = weight_variable([5, 5, 1, 32])
# b1 = bias_variable([1, 32])
# x_conv1 = tf.nn.relu(tf.add(conv2d(x_image, W1), b1))
# x_pool1 = max_pooling_2x2(x_conv1)
#
# # Conv layer 2 - 64x5x5
# W2 = weight_variable([5, 5, 32, 64])
# b2 = bias_variable([1, 64])
# x_conv2 = tf.nn.relu(tf.add(conv2d(x_pool1, W2), b2))
# x_pool2 = max_pooling_2x2(x_conv2)
#
# Flatten
# x_flat = tf.reshape(x_conv1, [-1, (image_gene_size ** 2) * 32])

# Fully connected
hidden_size = 256
num_classes = 4

W_fc1 = weight_variable([(image_gene_size ** 2) , hidden_size])
b_fc1 = bias_variable([hidden_size])
x_fc1 = tf.nn.relu(tf.add(tf.matmul(x, W_fc1), b_fc1))

# Regularization with dropout
keep_prob = tf.placeholder(tf.float32)
x_fc1_drop = tf.nn.dropout(x_fc1, keep_prob)

# Classification layer
W_fc2 = weight_variable([hidden_size, num_classes])
b_fc2 = bias_variable([num_classes])
y_est = tf.add(tf.matmul(x_fc1_drop, W_fc2), b_fc2)

# Probabilities - output from model (not the same as logits)
y_ = tf.nn.softmax(y_est)

learning_rate = 1e-6
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_est))
Optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# Setup to test accuracy of model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
con_mat = tf.confusion_matrix(labels=tf.argmax(y, 1), predictions=tf.argmax(y_, 1), num_classes=4, dtype=tf.int32)

sess.run(tf.global_variables_initializer())


def get_one_hot(lb):
    labels_indexes = {
        'stage i': 0,
        'stage ii': 1,
        'stage iii': 2,
        'stage iv': 3,
    }
    label = np.zeros(4)
    label[labels_indexes[lb]] = 1
    return label


def get_genes_batch(batch_size, i):
    begin = i * batch_size
    end = begin + batch_size
    labels = np.array([a for a in y_train.iloc[begin:end].apply(get_one_hot)])

    batch = X_train.iloc[begin:end, :].values, labels
    return batch


# Train model
batch_size = 104
n_epochs = int(X_train.shape[0] / batch_size)
y_test = np.array([a for a in y_test.apply(get_one_hot)])
for i in range(n_epochs):
    batch = get_genes_batch(batch_size, i)
    img_genes = batch[0]
    lbls = batch[1]

    train_accuracy = accuracy.eval(feed_dict={x: img_genes, y: lbls, keep_prob: 1.0})
    print("step %d, training accuracy %g" % (i, train_accuracy))

    Optimizer.run(feed_dict={x: img_genes, y: lbls, keep_prob: .7})

print("test accuracy %g" % accuracy.eval(feed_dict={x: X_test,
                                                    y: y_test, keep_prob: 1.0}))

print('Confusion Matrix: \n\n', tf.Tensor.eval(con_mat, feed_dict={x: X_test,
                                                                   y: y_test, keep_prob: 1.0}, session=None))

print("Done")
