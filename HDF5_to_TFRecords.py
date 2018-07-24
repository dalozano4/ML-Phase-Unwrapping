import h5py
import sys
import glob
from random import shuffle
import numpy as np
import tensorflow as tf


# I have n-number of HDF5 files in a directory and I want so save all these
# files into one TFRecord file

def _read_HDF5_files(input_file):
        # Calling the HDF5 file 
        hdf5 = h5py.File(input_file, 'r')

        # We are determining the keys/directories within the HDF5 file
        sets_to_read = list(hdf5.keys())

        # Using the keys we can open the contents within the HDF5 file.
        directory = {dictionaries: np.array(hdf5[dictionaries]) for dictionaries in sets_to_read}

        # In MATLAB, the memory arrangement is down the columns,
        # so the next item in memory after X(I,J) is X(I+1,J) rather than X(I,J+1).
        # After opening we need to transpose the matrix. 
        max_intensity_location = np.transpose(directory[sets_to_read[0]])
        unwrapped_phz =np.transpose(directory[sets_to_read[1]])
        wrapped_phz = np.transpose(directory[sets_to_read[2]])

        hdf5.close()

        return wrapped_phz, unwrapped_phz


def list_files_in_directory_by_ext(file_directory, ext):                   

    file_directory = file_directory + '\\*' + ext

    # This'll walk your directories recursively and return all absolute 
    # pathnames to matching ext files.
    files_path = glob.glob(file_directory)
    
    # Shuffle the data
    for i in range(100):
        shuffle(files_path) 

    return files_path

def print_progress(count, total):
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


#TensorFlow Tutorial #18
#TFRecords & Dataset API
#by Magnus Erik Hvass Pedersen
def _convert_HDF5_into_TFRecords(files_path, out_path):
    
    # Args:
    # file path   List of file-paths for the HDF5 files.
    # out_path    File-path for the TFRecords output file.

    # Number of images. Used when printing the progress.
    num_files = len(files_path)

    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:

        # Iterate over all the file-paths.
        for i, file in enumerate(files_path, 1):

            # Print the percentage-progress.
            print_progress(count=i, total=num_files)

            # Load the numpy array from the HDF5 files.
            wrapped, unwrapped = _read_HDF5_files(file)

            # Flatten out the images because tf.train.Feature class expects 1-D arrays.
            wrapped_vector = wrapped.flatten()
            unwrapped_vector = unwrapped.flatten()

            # Convert array to a list because tf.train.Feature class only supports lists
            wrapped_list_vector = wrapped_vector.tolist()
            unwrapped_list_vector = unwrapped_vector.tolist()

            # Create a dict with the data we want to save in the
            # TFRecords file. You can add more relevant data here.
            example = tf.train.Example(features=tf.train.Features(feature={
                    'wrapped':_floats_feature(wrapped_list_vector),
                    'unwrapped': _floats_feature(unwrapped_list_vector)}))

            # Serialize the data.
            serialized = example.SerializeToString()

            # Write the serialized data to the TFRecords file.
            writer.write(serialized)


class FLAGS(object):
  
    def __init__(self, 
                 file_directory,
                 ext,
                 path_tfrecords_train,
                 path_tfrecords_validation): 

        self.file_directory = file_directory
        self.ext = ext
        self.path_tfrecords_train = path_tfrecords_train
        self.path_tfrecords_validation = path_tfrecords_validation


if __name__ == '__main__':

    flags = FLAGS(file_directory = 'C:\\Users\\Diego Lozano\\AFRL_Project\\Summer_2018\\HDF5_files',
                  ext = '.h5',
                  path_tfrecords_train = 'C:\\Users\\Diego Lozano\\AFRL_Project\\Summer_2018\\train_data\\HDF5_train.tfrecords',
                  path_tfrecords_validation = 'C:\\Users\\Diego Lozano\\AFRL_Project\\Summer_2018\\validation_data\\HDF5_validation.tfrecords')

    # Get a list of path location for all the HDF5 files in the directory.
    HDF5_path = list_files_in_directory_by_ext(flags.file_directory, flags.ext)

    # We seperate the data into training and validation set.
    training_path = HDF5_path[:int(len(HDF5_path)*0.8)]
    validation_path = HDF5_path[int(len(HDF5_path)*0.8):]

    _convert_HDF5_into_TFRecords(files_path = training_path, 
                                 out_path = flags.path_tfrecords_train)

#    _convert_HDF5_into_TFRecords(files_path = validation_path, 
#                                 out_path = flags.path_tfrecords_validation)