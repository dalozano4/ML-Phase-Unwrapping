import tensorflow as tf


def _read_and_decode(serialized):

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
    wrapped = features['wrapped']

    # Reshape it to its original size.
    wrapped = tf.reshape(wrapped, [512, 512, 1])

    # Do the same for the Target image
    unwrapped = features['unwrapped']
    unwrapped = tf.reshape(unwrapped, [512, 512, 1])

    # The input matrix and target matrix are now correct TensorFlow types.
    return wrapped, unwrapped

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


class FLAGS(object):

    def __init__(self, 
                 batch_size,
                 path_tfrecords_train,
                 path_tfrecords_validation): 

        self.batch_size = batch_size
        self.path_tfrecords_train = path_tfrecords_train
        self.path_tfrecords_validation = path_tfrecords_validation


if __name__ == '__main__':

    flags = FLAGS(batch_size = 1,
                  path_tfrecords_train = 'C:\\Users\\Diego Lozano\\AFRL_Project\\Summer_2018\\train_data\\HDF5_train.tfrecords',
                  path_tfrecords_validation = 'C:\\Users\\Diego Lozano\\AFRL_Project\\Summer_2018\\validation_data\\HDF5_validation.tfrecords')

    dataset = Dataset_TFRecords(path_tfrecords_train = flags.path_tfrecords_train, 
                                path_tfrecords_valid = flags.path_tfrecords_validation,  
                                batch_size = flags.batch_size)

    with tf.Session() as sess:

        dataset.initialize_training_iterator(sess)

        images, annotations = dataset.get_next_training_element(sess)