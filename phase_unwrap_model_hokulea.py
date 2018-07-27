import sys
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

    def log_images(self, tag, img, step):
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
        hist.sum_squares = float(np.sum(values**2))

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


def _read_and_decode(serialized):
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
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    annotation = tf.decode_raw(features['annotation_raw'], tf.uint8)

    # Creating a variable that will normalize the input image
    norm = tf.constant(255, dtype=tf.float32)

    # When it is raw the data is a vector and we need to reshape it.
    image = tf.reshape(image, [512, 512, 3])

    # The type is now uint8 but we need it to be float.
    image = tf.cast(image, tf.float32)

    # Normalize the data before feeding it through. Optional.
    image = tf.divide(image, norm)

    # Repeat this process for the target segmentation.
    annotation = tf.reshape(annotation, [512, 512, 3])
    annotation = tf.cast(annotation, tf.float32)
    annotation = tf.divide(annotation, norm)

    # The image and label are now correct TensorFlow types.
    return image, annotation


class Dataset_TFRecords(object):

    def __init__(self, path_tfrecords_train, path_tfrecords_valid, batch_size):

        train_dataset = tf.data.TFRecordDataset(filenames=path_tfrecords_train)
        # Parse the serialized data in the TFRecords files.
        # This returns TensorFlow tensors for the image and labels.
        train_dataset = train_dataset.map(_read_and_decode)
        # String together various operations to apply to the data
        train_dataset = train_dataset.shuffle(1000)
        train_dataset = train_dataset.batch(batch_size)
        # Will iterate through the data once before throwing an OutOfRangeError
        self._train_iterator = train_dataset.make_one_shot_iterator()

        self._train_init_op = self._train_iterator.make_initializer(train_dataset)

        validation_dataset = tf.data.TFRecordDataset(filenames=path_tfrecords_valid)
        validation_dataset = validation_dataset.map(_read_and_decode)
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
        #evaluated and used to feed the `handle` placeholder.
        self._train_handle = sess.run(self._train_iterator.string_handle())
        feed_dict = {self._handle: self._train_handle}
        elements = sess.run(self._next_element, feed_dict=feed_dict)
        return(elements)

    def get_next_validation_element(self, sess):

        self._validation_handle = sess.run(self._validation_iterator.string_handle())
        feed_dict = {self._handle: self._validation_handle}
        elements = sess.run(self._next_element, feed_dict=feed_dict)
        return(elements)


class Modified_Unet(object):

    def __init__(self):
        # Build placeholders values which change during execution.
        self.x_placeholder = tf.placeholder(tf.float32, [None, 512, 512, 3])
        self.y_placeholder = tf.placeholder(tf.float32, [None, 512, 512, 3])

    def unet_resize_conv(self):
        conv_1 = slim.repeat(self.x_placeholder, 2, slim.conv2d, 64, [3, 3],
                             scope='conv1')
        pool_1 = slim.max_pool2d(conv_1, [2, 2], scope='pool1')

        conv_2 = slim.repeat(pool_1, 2, slim.conv2d, 128, [3, 3],
                             scope='conv2')
        pool_2 = slim.max_pool2d(conv_2, [2, 2], scope='pool2')

        conv_3 = slim.repeat(pool_2, 2, slim.conv2d, 256, [3, 3],
                             scope='conv3')
        pool_3 = slim.max_pool2d(conv_3, [2, 2], scope='pool3')

        conv_4 = slim.repeat(pool_3, 2, slim.conv2d, 512, [3, 3],
                             scope='conv4')
        pool_4 = slim.max_pool2d(conv_4, [2, 2], scope='pool4')

        conv_5 = slim.repeat(pool_4, 2, slim.conv2d, 1024, [3, 3],
                             scope='conv5')

        resize_nn4 = tf.image.resize_nearest_neighbor(conv_5, size=[64, 64])
        resize_nn4 = slim.conv2d(resize_nn4, 512, [2, 2], scope='resize_conv4')

        concat4 = tf.concat([resize_nn4, conv_4], axis=3)

        conv_4 = slim.repeat(concat4, 2, slim.conv2d, 512, [3, 3],
                             scope='uconv4')

        resize_nn3 = tf.image.resize_nearest_neighbor(conv_4, size=[128, 128])
        resize_nn3 = slim.conv2d(resize_nn3, 256, [2, 2], scope='resize_conv3')

        concat3 = tf.concat([resize_nn3, conv_3], axis=3)
        conv_3 = slim.repeat(concat3, 2, slim.conv2d, 256, [3, 3],
                             scope='uconv3')

        resize_nn2 = tf.image.resize_nearest_neighbor(conv_3, size=[256, 256])
        resize_nn2 = slim.conv2d(resize_nn2, 256, [2, 2], scope='resize_conv2')

        concat2 = tf.concat([resize_nn2, conv_2], axis=3)
        conv_2 = slim.repeat(concat2, 2, slim.conv2d, 128, [3, 3],
                             scope='uconv2')

        resize_nn1 = tf.image.resize_nearest_neighbor(conv_2, size=[512, 512])
        resize_nn1 = slim.conv2d(resize_nn1, 256, [2, 2], scope='resize_conv1')

        concat1 = tf.concat([resize_nn1, conv_1], axis=3)
        conv_1 = slim.repeat(concat1, 2, slim.conv2d, 64, [3, 3],
                             scope='uconv1')

        final_layer = slim.conv2d(conv_1, 3, [1, 1], scope='uconv1/uconv1_3')

        for variable in slim.get_model_variables():
            self.log_weights_bias(variable)

        return final_layer

    def unet_transpose_conv(self):

        # changed to 32 from 64
        conv_1 = slim.repeat(self.x_placeholder, 2, slim.conv2d, 16, [3, 3],
                             scope='conv1')
        pool_1 = slim.max_pool2d(conv_1, [2, 2], scope='pool1')

        conv_2 = slim.repeat(pool_1, 2, slim.conv2d, 32, [3, 3],
                             scope='conv2')
        pool_2 = slim.max_pool2d(conv_2, [2, 2], scope='pool2')

        conv_3 = slim.repeat(pool_2, 2, slim.conv2d, 64, [3, 3],
                             scope='conv3')
        pool_3 = slim.max_pool2d(conv_3, [2, 2], scope='pool3')

        conv_4 = slim.repeat(pool_3, 2, slim.conv2d, 128, [3, 3],
                             scope='conv4')
        pool_4 = slim.max_pool2d(conv_4, [2, 2], scope='pool4')

        conv_5 = slim.repeat(pool_4, 2, slim.conv2d, 256, [3, 3],
                             scope='conv5')

        deconv_4 = slim.convolution2d_transpose(conv_5, 128, [2, 2], [2, 2],
                                                padding='VALID',
                                                scope='tr_conv4')

        deconv_4 = tf.concat([deconv_4, conv_4], axis=3)
        conv_4 = slim.repeat(deconv_4, 2, slim.conv2d, 128, [3, 3],
                             scope='uconv4')

        deconv_3 = slim.convolution2d_transpose(conv_4, 64, [2, 2], [2, 2],
                                                padding='VALID',
                                                scope='tr_conv3')
        deconv_3 = tf.concat([deconv_3, conv_3], axis=3)
        conv_3 = slim.repeat(deconv_3, 2, slim.conv2d, 64, [3, 3],
                             scope='uconv3')

        deconv_2 = slim.convolution2d_transpose(conv_3, 32, [2, 2], [2, 2],
                                                padding='VALID',
                                                scope='tr_conv2')
        deconv_2 = tf.concat([deconv_2, conv_2], axis=3)
        conv_2 = slim.repeat(deconv_2, 2, slim.conv2d, 32, [3, 3],
                             scope='uconv2')

        deconv_1 = slim.convolution2d_transpose(conv_2, 16, [2, 2], [2, 2],
                                                padding='VALID',
                                                scope='tr_conv1')
        deconv_1 = tf.concat([deconv_1, conv_1], axis=3)
        conv_1 = slim.repeat(deconv_1, 2, slim.conv2d, 16, [3, 3],
                             scope='uconv1')

        final_conv_layer = slim.conv2d(conv_1, 3, [1, 1],
                                       scope='uconv1/uconv1_3')

        final_layer = tf.nn.softmax(final_conv_layer)

        for variable in slim.get_model_variables():
            self.log_weights_bias(variable)

        return final_layer

    def unet_inference(self):

        power_basis = FLAGS.power_basis

        # changed to 32 from 64
        conv_1 = slim.repeat(self.x_placeholder,
                             2,
                             slim.conv2d,
                             2**power_basis,
                             [3, 3],
                             scope='conv1')
        pool_1 = slim.max_pool2d(conv_1, [2, 2], scope='pool1')

        conv_2 = slim.repeat(pool_1, 2, slim.conv2d, 2 ** (power_basis + 1),
                             [3, 3],
                             scope='conv2')
        pool_2 = slim.max_pool2d(conv_2, [2, 2], scope='pool2')

        conv_3 = slim.repeat(pool_2, 2, slim.conv2d, 2 ** (power_basis + 2),
                             [3, 3],
                             scope='conv3')
        pool_3 = slim.max_pool2d(conv_3, [2, 2], scope='pool3')

        conv_4 = slim.repeat(pool_3, 2, slim.conv2d, 2 ** (power_basis + 3),
                             [3, 3],
                             scope='conv4')
        pool_4 = slim.max_pool2d(conv_4, [2, 2], scope='pool4')

        conv_5 = slim.repeat(pool_4, 2, slim.conv2d, 2 ** (power_basis + 4),
                             [3, 3],
                             scope='conv5')

        deconv_4 = slim.convolution2d_transpose(conv_5, 2 ** (power_basis + 3),
                                                [2, 2], [2, 2],
                                                padding='VALID',
                                                scope='tr_conv4')

        deconv_4 = tf.concat([deconv_4, conv_4], axis=3)
        conv_4 = slim.repeat(deconv_4, 2, slim.conv2d, 2 ** (power_basis + 3),
                             [3, 3],
                             scope='uconv4')

        deconv_3 = slim.convolution2d_transpose(conv_4, 2 ** (power_basis + 2),
                                                [2, 2], [2, 2],
                                                padding='VALID',
                                                scope='tr_conv3')
        deconv_3 = tf.concat([deconv_3, conv_3], axis=3)
        conv_3 = slim.repeat(deconv_3, 2, slim.conv2d, 2 ** (power_basis + 2),
                             [3, 3],
                             scope='uconv3')

        deconv_2 = slim.convolution2d_transpose(conv_3, 2 ** (power_basis + 1),
                                                [2, 2], [2, 2],
                                                padding='VALID',
                                                scope='tr_conv2')
        deconv_2 = tf.concat([deconv_2, conv_2], axis=3)
        conv_2 = slim.repeat(deconv_2, 2, slim.conv2d, 2 ** (power_basis + 1),
                             [3, 3],
                             scope='uconv2')

        deconv_1 = slim.convolution2d_transpose(conv_2, 2 ** (power_basis),
                                                [2, 2], [2, 2],
                                                padding='VALID',
                                                scope='tr_conv1')
        deconv_1 = tf.concat([deconv_1, conv_1], axis=3)
        conv_1 = slim.repeat(deconv_1, 2, slim.conv2d, 16, [3, 3],
                             scope='uconv1')

        final_conv_layer = slim.conv2d(conv_1, 3, [1, 1],
                                       scope='uconv1/uconv1_3')

        final_layer = tf.nn.softmax(final_conv_layer)

        for variable in slim.get_model_variables():
            self.log_weights_bias(variable)

        return final_layer

    def log_weights_bias(self, variable):

            with tf.name_scope(variable.op.name):

                tf.summary.scalar('mean', tf.reduce_mean(variable))

                tf.summary.scalar('max', tf.reduce_max(variable))

                tf.summary.scalar('min', tf.reduce_min(variable))

                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(
                                     variable - tf.reduce_mean(variable))))

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

    model = Modified_Unet()

    dataset = Dataset_TFRecords(path_tfrecords_train=FLAGS.path_tfrecords_train,
                                path_tfrecords_valid=FLAGS.path_tfrecords_valid,
                                batch_size = FLAGS.batch_size)

    logger = Logger(FLAGS.log_dir)

    # Now, we construct a loss function.
    # model_output_tensor = model.unet_transpose_conv()
    model_output_tensor = model.unet_inference()
#    model_output_tensor = model.unet_resize_conv()

    mean_squared_error = tf.squared_difference(model_output_tensor,
                                               model.y_placeholder)
    loss = tf.reduce_mean(mean_squared_error, name='mse_mean')

    # Next, add an optimizet to the graph.
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

    summary = tf.summary.merge_all()

    sv = tf.train.Supervisor(logdir=FLAGS.log_dir, save_summaries_secs=240.0)

    with sv.managed_session() as sess:

        for i in range(FLAGS.epochs):

            print("<------------Epoch" + str(i) + "------------->")

            train_steps = 0
            dataset.initialize_training_iterator(sess)
            print("<------------Training Output------------->")

            # Train for one epoch.
            while True:

                try:

                    start_time = time.time()

                    train_steps += 1

                    images, annotations = dataset.get_next_training_element(sess)

                    # Make a dict to load the batch onto the placeholders.
                    feed_dict = {model.x_placeholder: images,
                                 model.y_placeholder: annotations}

                    # Not a train loss.
                    train_loss = sess.run(loss, feed_dict=feed_dict)

                    sess.run(optimizer, feed_dict=feed_dict)

                    print("Running time = " + str(time.time() - start_time))

                except tf.errors.OutOfRangeError:

                    break

            print("<-------------Saving Training Data-------------->")

            predictions = sess.run(model_output_tensor, feed_dict=feed_dict)

            logger.log_images('train_input_image', images[0], i)

            logger.log_images('train_annotation_image', annotations[0], i)

            logger.log_images('train_predicted_image', predictions[0], i)

            logger.log_scalar('train_loss', train_loss, i)

            print("Training Epoch: {}, Mean Square Error: {:.3f}".format(i, train_loss))

            logger.log_scalar('training_loss', train_loss, i)

            dataset.initialize_validation_iterator(sess)
            print("<------------Validation Output------------->")
            while True:

                    try:

                        images, annotations = dataset.get_next_validation_element(sess)

                        # Make a dict to load the batch onto the placeholders.
                        feed_dict = {model.x_placeholder: images,
                                     model.y_placeholder: annotations}

                        validation_loss = sess.run(loss, feed_dict=feed_dict)

                    except tf.errors.OutOfRangeError:

                        break

            print("<-------------Saving Validation Data-------------->")

            print("Validation Epoch: {}, Mean Square Error: {:.3f}".format(i, validation_loss))

            predictions = sess.run(model_output_tensor,
                                   feed_dict=feed_dict)

            logger.log_images('validation_input_image', images[0], i)

            logger.log_images('validation_annotation_image', annotations[0], i)

            logger.log_images('validation_predicted_image', predictions[0], i)

            logger.log_scalar('validation_loss', validation_loss, i)

            summary_str = sess.run(summary, feed_dict=feed_dict)

            logger.writer.add_summary(summary_str, i)

        sv.stop()
        sess.close()


if __name__ == '__main__':

    # Instantiate an arg parser.
    parser = argparse.ArgumentParser()

    # Establish default arguements.

    # These flags are often, but not always, overwritten by the launcher.
    # parser.add_argument('--path_tfrecords_train', type=str,
    #                     default='/gpfs/projects/ml/data/phase_unwrapping/train_img3200.tfrecords',
    #                     help='Location of the training data set which is in .tfrecords format.')

    # parser.add_argument('--path_tfrecords_valid', type=str,
    #                     default='/gpfs/projects/ml/data/phase_unwrapping/validation_img800.tfrecords',
    #                     help='Location of the test data set which is in .tfrecords format.')

    # parser.add_argument('--log_dir', type=str,
    #                     default='/gpfs/projects/ml/phase_unwrapping_dnn/logs',
    #                     help='Summaries log directory.')


    # parser.add_argument('--batch_size', type=int, default=48,
    #                     help='Training set batch size.')

    # parser.add_argument('--epochs', type=int, default=10000000,
    #                     help='Number of epochs to run trainer.')

    # These flags are often, but not always, overwritten by the launcher.
    parser.add_argument('--path_tfrecords_train', type=str,
                        default='C:\\Users\\Justin Fletcher\\Research\\data\\phase_unwrapping\\train_img1.tfrecords',
                        help='Location of the training data set which is in .tfrecords format.')

    parser.add_argument('--path_tfrecords_valid', type=str,
                        default='C:\\Users\\Justin Fletcher\\Research\\data\\phase_unwrapping\\validation_img1.tfrecords',
                        help='Location of the test data set which is in .tfrecords format.')

    parser.add_argument('--log_dir', type=str,
                        default='C:\\Users\\Justin Fletcher\\Research\\data\\phase_unwrapping\\logs\\',
                        help='Summaries log directory.')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='Training set batch size.')

    parser.add_argument('--epochs', type=int, default=10000000,
                        help='Number of epochs to run trainer.')

    parser.add_argument('--ckpt_filename', type=str,
                        default='model-20180726.ckpt',
                        help='Summaries log directory.')

    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate.')

    parser.add_argument('--power_basis', type=float, default=4,
                        help='Power basis controlling unet expansion.')


    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=phase_unwrapping_tensorflow_model, argv=[sys.argv[0]] + unparsed)

#    tensorboard --logdir="C:\\Users\\Diego Lozano\\AFRL_Project\\Summer_2018\\templog"