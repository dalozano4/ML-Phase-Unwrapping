import sys
import math
import time
import argparse
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import tensorflow as tf

slim = tf.contrib.slim


class Logger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_image(self, tag, img, step):
        """Original version Logs a list of images."""
        """Updated version logs one image"""

        # Changes that were made were to comment the loop over a list of
        # images.
        # Change the input from images to img since we are passing only one
        # image everytime the function is called

        im_summaries = []

        s = BytesIO()

        plt.imsave(s, img, format='png')

        # Create an Image object
        img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                   height=img.shape[0],
                                   width=img.shape[1])

        # Create a Summary value
        im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, 0),
                                             image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)

    ###########################################################################
    # def log_images(self, tag, images, step):
    #     """Logs a list of images."""
    #     im_summaries = []
    #     for nr, img in enumerate(images):
    #         # Write the image to a string
    #         # s = StringIO()
    #         s = BytesIO()
    #         plt.imsave(s, img, format='png')

    #         # Create an Image object
    #         img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
    #                                    height=img.shape[0],
    #                                    width=img.shape[1])
    #         # Create a Summary value
    #         im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
    #                                              image=img_sum))

    #     # Create and write Summary
    #     summary = tf.Summary(value=im_summaries)
    #     self.writer.add_summary(summary, step)
    ######################################################################

    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])

        self.writer.add_summary(summary, step)

        self.writer.flush()


def _read_and_decode_uint8(serialized):
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.
    features = \
        {
            'image_raw': tf.FixedLenFeature([], tf.string),
            'annotation_raw': tf.FixedLenFeature([], tf.string)
        }

    features = tf.parse_single_example(serialized=serialized,
                                       features=features)

    # Decode the raw bytes so it becomes a tensor with type.
    input_image = tf.decode_raw(features['image_raw'], tf.uint8)
    target_image = tf.decode_raw(features['annotation_raw'], tf.uint8)

    # Creating a variable that will normalize the input image
    norm = tf.constant(255, dtype=tf.float32)

    # When it is raw the data is a vector and we need to reshape it.
    input_image = tf.reshape(input_image, [512, 512, 3])
    input_image = tf.image.rgb_to_grayscale(input_image)
    # The type is now uint8 but we need it to be float.
    input_image = tf.cast(input_image, tf.float32)

    # Normalize the data before feeding it through. Optional.
    input_image = tf.divide(input_image, norm)
    input_image = tf.subtract(tf.multiply(input_image, 2), 1)

    # Repeat this process for the target segmentation.
    target_image = tf.reshape(target_image, [512, 512, 3])
    target_image = tf.image.rgb_to_grayscale(target_image)
    target_image = tf.cast(target_image, tf.float32)
    target_image = tf.divide(target_image, norm)
    target_image = tf.subtract(tf.multiply(target_image, 2), 1)
    # The image and label are now correct TensorFlow types.
    return input_image, target_image


def _read_and_decode_float32(serialized):
    # Wtih tf.FixedLenFeature I kept getting a
    # ValueError: Cannot reshape a tensor with 1 elements.
    # Instead tf.FixedLenSequenceFeature works not sure why ???
    features = \
        {
            'wrapped': tf.FixedLenSequenceFeature([],
                                                  tf.float32,
                                                  allow_missing=True),
            'unwrapped': tf.FixedLenSequenceFeature([],
                                                    tf.float32,
                                                    allow_missing=True)
        }

    # Parse the serialized data so we get a dict with our data.
    features = tf.parse_single_example(serialized=serialized,
                                       features=features)

    # Get the Input matrix, which was flatten.
    input_matrix = features['wrapped']

    # Reshape it to its original size.
    input_matrix = tf.reshape(input_matrix, [512, 512, 1])

    # Wrapped phz images are bounded by pi and -pi so we know their min and max
    input_matrix_min = tf.constant(-math.pi, dtype=tf.float32)
    input_matrix_max = tf.constant(math.pi, dtype=tf.float32)

    # Determine the range between the min and max
    input_matrix_range = tf.subtract(input_matrix_max, input_matrix_min)

    # Normalize the matrix by subtracting my the min and dividing by the range [0,1]
    input_matrix = tf.divide(tf.subtract(input_matrix, input_matrix_min), input_matrix_range)

    # Multiply by 2 to range [0, 2] and subtract by 1 to normalize to [-1, 1]
    input_matrix = tf.subtract(tf.multiply(input_matrix, 2), 1)

    # Do the same for the Target image
    target_matrix = features['unwrapped']
    target_matrix = tf.reshape(target_matrix, [512, 512, 1])
    target_matrix_min = tf.constant(-278.51978182, dtype=tf.float32)
    target_matrix_max = tf.constant(268.47228778, dtype=tf.float32)
    target_matrix_range = tf.subtract(target_matrix_max, target_matrix_min)
    target_matrix = tf.divide(tf.subtract(target_matrix, target_matrix_min), target_matrix_range)
    target_matrix = tf.subtract(tf.multiply(target_matrix, 2), 1)

    # The input matrix and target matrix are now correct TensorFlow types.
    return input_matrix, target_matrix


class DatasetTFRecords(object):

    def __init__(self, path_tfrecords_train, path_tfrecords_valid, data_type, batch_size):

        train_dataset = tf.data.TFRecordDataset(filenames=path_tfrecords_train)
        # Parse the serialized data in the TFRecords files.
        # This returns TensorFlow tensors for the input image and target image.
        if data_type == 'float32':
            train_dataset = train_dataset.map(_read_and_decode_float32)

        elif data_type == 'uint8':
            train_dataset = train_dataset.map(_read_and_decode_uint8)

        else:
            print('data_type has to be either float32 or uint8')

        # String together various operations to apply to the data
        train_dataset = train_dataset.shuffle(1000)
        train_dataset = train_dataset.batch(batch_size)
        # Will iterate through the data once before throwing an OutOfRangeError
        self._train_iterator = train_dataset.make_one_shot_iterator()

        self._train_init_op = self._train_iterator.make_initializer(train_dataset)

        validation_dataset = tf.data.TFRecordDataset(filenames=path_tfrecords_valid)

        if data_type == 'float32':
            validation_dataset = validation_dataset.map(_read_and_decode_float32)

        elif data_type == 'uint8':
            validation_dataset = validation_dataset.map(_read_and_decode_uint8)

        validation_dataset = validation_dataset.shuffle(1000)
        validation_dataset = validation_dataset.batch(batch_size)
        self._validation_iterator = validation_dataset.make_one_shot_iterator()

        self._validation_init_op = self._validation_iterator.make_initializer(validation_dataset)

        # Create a placeholder that can be dynamically changed between train
        # and test.
        self._handle = tf.placeholder(tf.string, shape=[])

        # Define a generic iterator using the shape of the dataset
        iterator = tf.data.Iterator.from_string_handle(self._handle,
                                                       train_dataset.output_types,
                                                       train_dataset.output_shapes)

        self._next_element = iterator.get_next()
        self._train_handle = []
        self._validation_handle = []

    def initialize_training_iterator(self, sess):
        sess.run(self._train_init_op)

    def initialize_validation_iterator(self, sess):
        sess.run(self._validation_init_op)

    def get_next_training_element(self, sess):
        # The `Iterator.string_handle()` method returns a tensor that can be
        # evaluated and used to feed the `handle` placeholder.
        self._train_handle = sess.run(self._train_iterator.string_handle())
        feed_dict = {self._handle: self._train_handle}
        elements = sess.run(self._next_element, feed_dict=feed_dict)
        return elements

    def get_next_validation_element(self, sess):
        self._validation_handle = sess.run(self._validation_iterator.string_handle())
        feed_dict = {self._handle: self._validation_handle}
        elements = sess.run(self._next_element, feed_dict=feed_dict)
        return elements


class Pix2Pix(object):
    def __init__(self,
                 image_size=512,
                 generator_filters=64,
                 discriminator_filters=64,
                 num_channels=1):

        self.image_size = image_size

        self.generator_filters = generator_filters

        self.discriminator_filters = discriminator_filters

        self.num_channels = num_channels

        self.x_placeholder = tf.placeholder(tf.float32, [None, image_size, image_size, num_channels])

        self.y_placeholder = tf.placeholder(tf.float32, [None, image_size, image_size, num_channels])

    def discriminator(self, image):

        h1 = slim.conv2d(image, self.discriminator_filters, [4, 4], stride=2,
                         activation_fn=tf.nn.leaky_relu, scope='dis_layer_1')

        h2 = slim.conv2d(h1, self.discriminator_filters * 2, [4, 4], stride=2,
                         activation_fn=None, scope='dis_layer_2')
        h2 = slim.batch_norm(h2, decay=0.9)

        h3 = slim.conv2d(tf.nn.leaky_relu(h2), self.discriminator_filters * 4, [4, 4], stride=2,
                         activation_fn=None, scope='dis_layer_3')
        h3 = slim.batch_norm(h3, decay=0.9)

        h4 = slim.conv2d(tf.nn.leaky_relu(h3), self.discriminator_filters * 8, [4, 4], stride=1,
                         activation_fn=None, scope='dis_layer_4')
        h4 = slim.batch_norm(h4, decay=0.9)
        h4 = tf.nn.leaky_relu(h4)

        h5 = slim.conv2d(tf.nn.leaky_relu(h4), self.num_channels, [4, 4], stride=1,
                         activation_fn=None, scope='dis_layer_5')

        return tf.nn.sigmoid(h5), h5

    def generator(self):
        encoder_1 = slim.conv2d(self.x_placeholder, self.generator_filters, [4, 4], stride=2,
                                activation_fn=None, scope='gen_encoder_1')

        encoder_2 = slim.conv2d(tf.nn.leaky_relu(encoder_1), self.generator_filters * 2, [4, 4],
                                stride=2, activation_fn=None, scope='gen_encoder_2')
        encoder_2 = slim.batch_norm(encoder_2, decay=0.9)

        encoder_3 = slim.conv2d(tf.nn.leaky_relu(encoder_2), self.generator_filters * 4, [4, 4],
                                stride=2, activation_fn=None, scope='gen_encoder_3')
        encoder_3 = slim.batch_norm(encoder_3, decay=0.9)

        encoder_4 = slim.conv2d(tf.nn.leaky_relu(encoder_3), self.generator_filters * 8, [4, 4],
                                stride=2, activation_fn=None, scope='gen_encoder_4')
        encoder_4 = slim.batch_norm(encoder_4, decay=0.9)

        encoder_5 = slim.conv2d(tf.nn.leaky_relu(encoder_4), self.generator_filters * 8, [4, 4],
                                stride=2, activation_fn=None, scope='gen_encoder_5')
        encoder_5 = slim.batch_norm(encoder_5, decay=0.9)

        encoder_6 = slim.conv2d(tf.nn.leaky_relu(encoder_5), self.generator_filters * 8, [4, 4],
                                stride=2, activation_fn=None, scope='gen_encoder_6')
        encoder_6 = slim.batch_norm(encoder_6, decay=0.9)

        encoder_7 = slim.conv2d(tf.nn.leaky_relu(encoder_6), self.generator_filters * 8, [4, 4],
                                stride=2, activation_fn=None, scope='gen_encoder_7')
        encoder_7 = slim.batch_norm(encoder_7, decay=0.9)

        encoder_8 = slim.conv2d(tf.nn.leaky_relu(encoder_7), self.generator_filters * 8, [4, 4],
                                stride=2, activation_fn=None, scope='gen_encoder_8')
        encoder_8 = slim.batch_norm(encoder_8, decay=0.9)

        decoder_1 = slim.convolution2d_transpose(tf.nn.relu(encoder_8), self.generator_filters * 8, [4, 4],
                                                 stride=2, activation_fn=None, scope='gen_decoder_1')
        decoder_1 = slim.batch_norm(decoder_1, decay=0.9)
        decoder_1 = tf.nn.dropout(decoder_1, keep_prob=0.5)
        decoder_1 = tf.concat([decoder_1, encoder_7], 3)

        decoder_2 = slim.convolution2d_transpose(tf.nn.relu(decoder_1), self.generator_filters * 8, [4, 4],
                                                 stride=2, activation_fn=None, scope='gen_decoder_2')
        decoder_2 = slim.batch_norm(decoder_2, decay=0.9)
        decoder_2 = tf.nn.dropout(decoder_2, keep_prob=0.5)
        decoder_2 = tf.concat([decoder_2, encoder_6], 3)

        decoder_3 = slim.convolution2d_transpose(tf.nn.relu(decoder_2), self.generator_filters * 8, [4, 4],
                                                 stride=2, activation_fn=None, scope='gen_decoder_3')
        decoder_3 = slim.batch_norm(decoder_3, decay=0.9)
        decoder_3 = tf.nn.dropout(decoder_3, keep_prob=0.5)
        decoder_3 = tf.concat([decoder_3, encoder_5], 3)

        decoder_4 = slim.convolution2d_transpose(tf.nn.relu(decoder_3), self.generator_filters * 8, [4, 4],
                                                 stride=2, activation_fn=None, scope='gen_decoder_4')
        decoder_4 = slim.batch_norm(decoder_4, decay=0.9)
        decoder_4 = tf.concat([decoder_4, encoder_4], 3)

        decoder_5 = slim.convolution2d_transpose(tf.nn.relu(decoder_4), self.generator_filters * 4, [4, 4],
                                                 stride=2, activation_fn=None, scope='gen_decoder_5')
        decoder_5 = slim.batch_norm(decoder_5, decay=0.9)
        decoder_5 = tf.concat([decoder_5, encoder_3], 3)

        decoder_6 = slim.convolution2d_transpose(tf.nn.relu(decoder_5), self.generator_filters * 2, [4, 4],
                                                 stride=2, activation_fn=None, scope='gen_decoder_6')
        decoder_6 = slim.batch_norm(decoder_6, decay=0.9)
        decoder_6 = tf.concat([decoder_6, encoder_2], 3)

        decoder_7 = slim.convolution2d_transpose(tf.nn.relu(decoder_6), self.generator_filters, [4, 4],
                                                 stride=2, activation_fn=None, scope='gen_decoder_7')
        decoder_7 = slim.batch_norm(decoder_7, decay=0.9)
        decoder_7 = tf.concat([decoder_7, encoder_1], 3)

        decoder_8 = slim.convolution2d_transpose(tf.nn.relu(decoder_7), self.num_channels, [4, 4],
                                                 stride=2, activation_fn=None, scope='gen_decoder_8')
        return tf.nn.tanh(decoder_8)


def log_weights_bias():
    for variable in slim.get_model_variables():
        with tf.name_scope(variable.op.name):
            tf.summary.scalar('mean', tf.reduce_mean(variable))

            tf.summary.scalar('max', tf.reduce_max(variable))

            tf.summary.scalar('min', tf.reduce_min(variable))

            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(variable - tf.reduce_mean(variable))))

            tf.summary.scalar('stddev', stddev)

            tf.summary.histogram('histogram', variable)


def phase_unwrapping_tensorflow_model(_):

    # Clear the log directory, if it exists.
    if tf.gfile.Exists(FLAGS.log_dir):

        tf.gfile.DeleteRecursively(FLAGS.log_dir)

    # Create a log directory, if it exists.
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Reset the default graph.
    tf.reset_default_graph()

    model = Pix2Pix(image_size=512,
                    generator_filters=16,
                    discriminator_filters=16,
                    num_channels=1)

    dataset = DatasetTFRecords(path_tfrecords_train=FLAGS.path_tfrecords_train,
                               path_tfrecords_valid=FLAGS.path_tfrecords_valid,
                               data_type=FLAGS.data_type,
                               batch_size=FLAGS.batch_size)

    logger = Logger(FLAGS.log_dir)

    with tf.variable_scope("generator"):
        # We call the generator prediction Tensor.
        gen_output_tensor = model.generator()

    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # call the discriminator that compares the input vs target
            input_target = tf.concat([model.x_placeholder, model.y_placeholder], 3)
            predict_real, real_logit = model.discriminator(input_target)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # call the discriminator that compares the input vs generator output
            input_output = tf.concat([model.x_placeholder, gen_output_tensor], 3)
            predict_fake, fake_logit = model.discriminator(input_output)

    # predict_real => 1
    # predict_fake => 0
    # The discriminator should maximize the probability of the training data
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit,
                                                                         labels=tf.ones_like(predict_real)))

    # The discriminator should minimize the probability of the data is from the generator
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit,
                                                                         labels=tf.zeros_like(predict_fake)))

    discriminator_loss = d_loss_real + d_loss_fake

    # Generator needs to maximize the probability that the discriminator won't detect generated samples
    gan_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit,
                                                                      labels=tf.ones_like(predict_fake)))
    # The generator is also tasked to be near the ground truth output
    l1_loss = FLAGS.L1_lambda * tf.reduce_mean(tf.abs(model.y_placeholder - gen_output_tensor))

    generator_loss = gan_loss + l1_loss

    log_weights_bias()

    variables_to_restore = slim.get_model_variables()

    dis_vars = [var for var in variables_to_restore if 'dis_' in var.name]
    gen_vars = [var for var in variables_to_restore if 'gen_' in var.name]

    dis_optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate,
                                           beta1=FLAGS.beta1).minimize(discriminator_loss, var_list=dis_vars)

    gen_optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate,
                                           beta1=FLAGS.beta1).minimize(generator_loss, var_list=gen_vars)

    summary = tf.summary.merge_all()

    sv = tf.train.Supervisor(logdir=FLAGS.log_dir,
                             summary_op=None)

    with sv.managed_session() as sess:
        for i in range(FLAGS.epochs):

            print("<----------------- Training Sessions ------------------>")

            dataset.initialize_training_iterator(sess)

            # Train for one epoch.
            while True:

                try:

                    start_time = time.time()

                    input_matrix, target_matrix = dataset.get_next_training_element(sess)

                    # Make a dict to load the batch onto the placeholders.
                    feed_dict = {model.x_placeholder: input_matrix,
                                 model.y_placeholder: target_matrix}

                    # Update Discriminator
                    train_dis_loss = sess.run(discriminator_loss, feed_dict=feed_dict)

                    sess.run(dis_optimizer, feed_dict=feed_dict)

                    # Update Generator
                    train_gen_loss = sess.run(generator_loss, feed_dict=feed_dict)

                    sess.run(gen_optimizer, feed_dict=feed_dict)

                    print("Running time = " + str(time.time() - start_time))

                except tf.errors.OutOfRangeError:

                    break

            print("<------------- Training Loss -------------->")

            print("Training Epoch: {}, Discriminator_Loss: {:.8f}, Generator_Loss: {:.8f}"
                  .format(i, train_dis_loss, train_gen_loss))

            logger.log_scalar('train_discriminator_loss', train_dis_loss, i)

            logger.log_scalar('train_generator_loss', train_gen_loss, i)

            if i % FLAGS.save_count == 0:

                print("<------------- Saving Training Images -------------->")

                predictions = sess.run(gen_output_tensor, feed_dict=feed_dict)

                # Since this is a cGAN our data is normalize from [-1, 1]
                # I am not sure if matplotlib will throw a fit but just in case saved images will be within [0, 1]
                norm_input_matrix = (input_matrix[0, :, :, 0] + 1) / 2
                norm_target_matrix = (target_matrix[0, :, :, 0] + 1) / 2
                norm_predictions = (predictions[0, :, :, 0] + 1) / 2

                logger.log_image('train_input_matrix', norm_input_matrix, i)

                logger.log_image('train_target_matrix', norm_target_matrix, i)

                logger.log_image('train_predicted_image', norm_predictions, i)

            print("<---------------- Validation Session ----------------->")

            dataset.initialize_validation_iterator(sess)
            while True:

                try:

                    input_matrix, target_matrix = dataset.get_next_validation_element(sess)

                    # Make a dict to load the batch onto the placeholders.
                    feed_dict = {model.x_placeholder: input_matrix,
                                 model.y_placeholder: target_matrix}

                    valid_dis_loss = sess.run(discriminator_loss, feed_dict=feed_dict)
                    valid_gen_loss = sess.run(generator_loss, feed_dict=feed_dict)

                except tf.errors.OutOfRangeError:

                    break

            print("<---------------- Validation Loss ----------------->")

            print("Training Epoch: {}, Discriminator_Loss: {:.8f}, Generator_Loss: {:.8f}"
                  .format(i, valid_dis_loss, valid_gen_loss))

            logger.log_scalar('valid_discriminator_loss', valid_dis_loss, i)

            logger.log_scalar('valid_generator_loss', valid_gen_loss, i)

            if i % FLAGS.save_count == 0:

                print("<------------- Saving Validation Images -------------->")

                predictions = sess.run(gen_output_tensor, feed_dict=feed_dict)

                norm_input_matrix = (input_matrix[0, :, :, 0] + 1) / 2
                norm_target_matrix = (target_matrix[0, :, :, 0] + 1) / 2
                norm_predictions = (predictions[0, :, :, 0] + 1) / 2

                logger.log_image('valid_input_matrix', norm_input_matrix, i)

                logger.log_image('valid_target_matrix', norm_target_matrix, i)

                logger.log_image('valid_predicted_image', norm_predictions, i)

            print("<------------- Saving Weights and Bias -------------->")

            summary_str = sess.run(summary, feed_dict=feed_dict)
            logger.writer.add_summary(summary_str, i)

        sv.stop()
        sess.close()


if __name__ == '__main__':
    # Instantiate an arg parser.
    parser = argparse.ArgumentParser()

    # Establish default arguements.

    # These flags are often, but not always, overwritten by the launcher.
    parser.add_argument('--path_tfrecords_train', type=str,
                        default='C:\\Users\\Diego Lozano\\AFRL_Project\\train_data\\HDF5_train.tfrecords',
                        help='Location of the training data set which is in .tfrecords format.')

    parser.add_argument('--path_tfrecords_valid', type=str,
                        default='C:\\Users\\Diego Lozano\\AFRL_Project\\validation_data\\HDF5_validation.tfrecords',
                        help='Location of the test data set which is in .tfrecords format.')

    parser.add_argument('--log_dir', type=str,
                        default='C:\\Users\\Diego Lozano\\AFRL_Project\\templog',
                        help='Summaries log directory.')

    parser.add_argument('--data_type', type=str,
                        default='float32',
                        help='Summaries log directory.')

    parser.add_argument('--save_count', type=int, default=5,
                        help='Save variables at every set count.')

    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Initial learning rate.')

    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 value for Adam Optimizer as given in cGan optimizers.')

    parser.add_argument('--L1_lambda', type=float, default=100,
                        help=' Weight that will be multiplied by the L1_loss in the generator total loss')

    parser.add_argument('--batch_size', type=int, default=2,
                        help='Training set batch size.')

    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to run trainer.')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=phase_unwrapping_tensorflow_model, argv=[sys.argv[0]] + unparsed)

    # tensorboard --logdir="C:\\Users\\Diego Lozano\\AFRL_Project\\Summer_2018\\templog"
