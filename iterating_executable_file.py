import subprocess
import numpy as np
import h5py
import argparse
import os
from matplotlib import image


def saving_nparray_as_png(input_file, array):
    
    image.imsave(input_file, array)


def create_png_files_from_HDF5_files(input_file):
    # When going from Matlab HDF5 to Python the matrices seemed to be transposed 
    f = h5py.File(input_file, 'r')
    determining_keys_of_HDF5 = list(f.keys())
    
    directory_max_intensity = f[determining_keys_of_HDF5[0]]
    directory_unwrapped_phz = f[determining_keys_of_HDF5[1]]
    directory_wrapped_phz = f[determining_keys_of_HDF5[2]]
    
    open_contents_max_intesity = list(directory_max_intensity)
    open_contents_unwrapped_phz = list(directory_unwrapped_phz)
    open_contents_wrapped_phz = list(directory_wrapped_phz)
    
    max_intensity_location = np.transpose(open_contents_max_intesity) 
    unwrapped_phz = np.transpose(open_contents_unwrapped_phz) 
    wrapped_phz = np.transpose(open_contents_wrapped_phz)
    
    
    splitting_input_file = os.path.splitext(input_file)
    renaming_input_file = splitting_input_file[0]
    saving_nparray_as_png(renaming_input_file + '_unwrapped.png', unwrapped_phz)
    saving_nparray_as_png(renaming_input_file + '_wrapped.png', wrapped_phz)


def serializing_output_file_name(output_file):
    
    splitting_output_file = os.path.splitext(output_file)
    
    root_output_file = splitting_output_file[0]
    extension_output_file = splitting_output_file[1]
    
    serialize_output_file = root_output_file + '_' + str(serializing_output_file_name.count) + extension_output_file
    serializing_output_file_name.count+=1
    
    return serialize_output_file


def  run_propagation_simulation(num_iteration,
                                rytov_value,
                                wavelength_m,
                                inner_scale_length_m,
                                outer_scale_length_m,
                                propagation_length_m,
                                grid_height_width,
                                grid_spacing_height_width_m,
                                num_screens,
                                Power_Spectral_Density_Model,
                                output_file_location,
                                output_file_name):

    output_file = output_file_location + output_file_name
    executable_file_location = 'C:\\Program Files\\data_gen_overwrite\\application\\data_gen_overwrite.exe'
    
    for i in range(num_iteration):
        
        serialized_output_file = serializing_output_file_name(output_file)
        
        suprocess_command = []
        suprocess_command.append(executable_file_location)
        suprocess_command.append(str(rytov_value))
        suprocess_command.append(str(wavelength_m))
        suprocess_command.append(str(inner_scale_length_m))
        suprocess_command.append(str(outer_scale_length_m))
        suprocess_command.append(str(propagation_length_m))
        suprocess_command.append(str(grid_height_width))
        suprocess_command.append(str(grid_spacing_height_width_m))
        suprocess_command.append(str(num_screens))
        suprocess_command.append(Power_Spectral_Density_Model)
        suprocess_command.append(serialized_output_file)
        
        subprocess.check_call(suprocess_command)
        create_png_files_from_HDF5_files(input_file = serialized_output_file)


def main(FLAGS):

    run_propagation_simulation(2,
                               FLAGS.rytov_value,
                               FLAGS.wavelength_m,
                               FLAGS.inner_scale_length_m,
                               FLAGS.outer_scale_length_m,
                               FLAGS.propagation_length_m,
                               FLAGS.grid_height_width,
                               FLAGS.grid_spacing_height_width_m,
                               FLAGS.num_screens,
                               FLAGS.Power_Spectral_Density_Model,
                               FLAGS.output_file_location,
                               FLAGS.output_file_name)
    

if __name__ == '__main__':

    # Instantiate an arg parser.
    parser = argparse.ArgumentParser()

    # Establish default arguements.

    # These flags are often, but not always, overwritten by the launcher.
    parser.add_argument('--rytov_value', type=float, default = 0.1,
                        help='.')

    parser.add_argument('--wavelength_m', type=float, default = 0.798e-6,
                        help='.')

    parser.add_argument('--inner_scale_length_m', type=float, default = 1e-3,
                        help='.')

    parser.add_argument('--outer_scale_length_m', type=float, default = 5.0,
                        help='.')

    parser.add_argument('--propagation_length_m', type=float,default=10e3,
                        help='.')

    parser.add_argument('--grid_height_width', type=float,default=512,
                        help='.')

    parser.add_argument('--grid_spacing_height_width_m', type=float, default=5e-3,
                        help='.')

    parser.add_argument('--num_screens', type=float, default=21,
                        help='.')

    parser.add_argument('--Power_Spectral_Density_Model', type=str, default='vk',
                        help='.')

    parser.add_argument('--output_file_location', type=str,
                        default='C:\\Users\\Diego Lozano\\AFRL_Project\\Output\\',
                        help='.')
    
    parser.add_argument('--output_file_name', type=str,
                        default='June27.h5',
                        help='.')


    # Parse known arguements.
    FLAGS, unparsed = parser.parse_known_args()

    serializing_output_file_name.count = 0
    main(FLAGS)