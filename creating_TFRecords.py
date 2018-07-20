import sys
import cv2
import glob
import tensorflow as tf


def list_images_in_directory(image_path, annotation_path):                   
    
    image_path = image_path + '\\*.png' 
    annotation_path = annotation_path + '\\*.png' 
    
    # This'll walk your directories recursively and return all absolute pathnames to matching .png files.
    image_address = sorted(glob.glob(image_path),key=len)
    annotation_address = sorted(glob.glob(annotation_path),key=len)
  
    return image_address, annotation_address


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

#TensorFlow Tutorial #18
#TFRecords & Dataset API
#by Magnus Erik Hvass Pedersen
def convert_image_into_TFRecords(image_paths, 
                                 annotation_paths, 
                                 out_path):
    # Args:
    # image_paths   List of file-paths for the images.
    # labels        Class-labels for the images.
    # out_path      File-path for the TFRecords output file.
    # Number of images. Used when printing the progress.
    num_images = len(image_paths)
    
    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:
        
        # Iterate over all the image-paths and class-labels.
        for i, (img_path, ann_path) in enumerate(zip(image_paths, annotation_paths), 1):
            # Print the percentage-progress.
            print_progress(count=i, total=num_images)

            # Load the image-file using matplotlib's imread function.
            image = cv2.imread(img_path)
            annotation = cv2.imread(ann_path)
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            annotation = cv2.cvtColor(annotation, cv2.COLOR_BGR2RGB)
            
            # Convert the image to raw bytes.
            image_raw = image.tostring()
            annotation_raw = annotation.tostring()
            # Create a dict with the data we want to save in the
            # TFRecords file. You can add more relevant data here.
            example = tf.train.Example(features=tf.train.Features(feature={
                    'image_raw': _bytes_feature(image_raw),
                    'annotation_raw': _bytes_feature(annotation_raw)}))

            # Serialize the data.
            serialized = example.SerializeToString()
            
            # Write the serialized data to the TFRecords file.
            writer.write(serialized)

     
class FLAGS(object):
  
    def __init__(self, 
                 train_image_path,
                 train_annotation_path,
                 path_tfrecords_train,
                 validation_image_path, 
                 validation_annotation_path,
                 path_tfrecords_validation): 

        self.train_image_path = train_image_path
        self.train_annotation_path = train_annotation_path
        self.path_tfrecords_train = path_tfrecords_train
        self.validation_image_path = validation_image_path
        self.validation_annotation_path = validation_annotation_path
        self.path_tfrecords_validation = path_tfrecords_validation

      
      
if __name__ == '__main__':
    # Define path to data and hyperparameters for training the model.
    flags = FLAGS(train_image_path = 'C:\\Users\\Diego Lozano\\AFRL_Project\\Summer_2018\\train_data\\image',
                  train_annotation_path = 'C:\\Users\\Diego Lozano\\AFRL_Project\\Summer_2018\\train_data\\annotation',
                  path_tfrecords_train = 'C:\\Users\\Diego Lozano\\AFRL_Project\\Summer_2018\\train_data\\trial_train.tfrecords',
                  validation_image_path = 'C:\\Users\\Diego Lozano\\AFRL_Project\\Summer_2018\\validation_data\\image',
                  validation_annotation_path = 'C:\\Users\\Diego Lozano\\AFRL_Project\\Summer_2018\\validation_data\\annotation',
                  path_tfrecords_validation = 'C:\\Users\\Diego Lozano\\AFRL_Project\\Summer_2018\\validation_data\\trial_validation.tfrecords')
    
    train_image_address, train_annotation_address = list_images_in_directory(
            flags.train_image_path, flags.train_annotation_path)
    
    validation_image_address, validation_annotation_address = list_images_in_directory(
            flags.validation_image_path, flags.validation_annotation_path)
    
    convert_image_into_TFRecords(train_image_address[:15], 
                                 train_annotation_address, 
                                 flags.path_tfrecords_train)
    
#    convert_image_into_TFRecords(validation_image_address[:5], 
#                                 validation_annotation_address, 
#                                 flags.path_tfrecords_validation)