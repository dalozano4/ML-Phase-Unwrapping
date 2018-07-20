import subprocess
import numpy as np
import h5py
import argparse
import os
from matplotlib import image

class Propagation_Simulation():
    
    
    def _move_file_by_extention(self,current_location,new_location, ext):
        
        # Produces a list, containing all the names within the directory
        current_file_directory = sorted(os.listdir(current_location),key=len)
        
        # Will loop through all the names inside the list
        for items in current_file_directory:
            
            # If the files have an extension of '.h5' it will return a TRUE statement
            if items.endswith(ext):
                # These files will be moved
                current_path = os.path.join(current_location, items)
                new_path = os.path.join(new_location,items)
                # These files will be moved
                os.rename(current_path, new_path)
    
    
    def _create_png_files_from_HDF5_files(self,input_file):
        # Calling the HDF5 file 
        f = h5py.File(input_file, 'r')
        
        # We are determining the keys/directories within the HDF5 file
        determining_keys_of_HDF5 = list(f.keys())
        
        # Using the keys we can open the contents within the HDF5 file.
        open_max_intensity = f[determining_keys_of_HDF5[0]]
        open_unwrapped_phz = f[determining_keys_of_HDF5[1]]
        open_wrapped_phz = f[determining_keys_of_HDF5[2]]
        
        # We need to list the contents inside. 
        contents_max_intesity = list(open_max_intensity)
        contents_unwrapped_phz = list(open_unwrapped_phz)
        contents_wrapped_phz = list(open_wrapped_phz)
        
        # In MATLAB, the memory arrangement is down the columns,
        # so the next item in memory after X(I,J) is X(I+1,J) rather than X(I,J+1).
        # After opening we need to transpose the matrix. 
        max_intensity_location = np.transpose(contents_max_intesity) 
        unwrapped_phz = np.transpose(contents_unwrapped_phz) 
        wrapped_phz = np.transpose(contents_wrapped_phz)
        
        # Removing initial extention in the input file
        splitting_input_file = os.path.splitext(input_file)
        renaming_input_file = splitting_input_file[0]
        
        # Saving the numpy array as a png
        image.imsave(renaming_input_file + '_unwrapped.png', unwrapped_phz)
        image.imsave(renaming_input_file + '_wrapped.png', wrapped_phz)
    
    
    def _serializing_output_file_name(self,output_file, count):
        
        # Does a reverse search for '.' and returns the (root, extension) portion.
        splitting_output_file = os.path.splitext(output_file)
        root = splitting_output_file[0]
        extension = splitting_output_file[1]
        
        # Serialize the output file name by a count.
        serialize_output_file = root + '_' + str(count) + extension
        
        return serialize_output_file
    
    
    def  _run_propagation_simulation(self,
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
                                    output_file_name,
                                    HDF5_file_location,
                                    num_iteration=555):
    
        output_file = output_file_location + output_file_name
        executable_file_location = 'C:\\Program Files\\data_gen_overwrite\\application\\data_gen_overwrite.exe'
        count = 444
        
        for i in range(num_iteration):
            
            serialized_output_file = self._serializing_output_file_name(output_file, count)
            
            # Generate the subprocess args needed to call the .exe file.
            # Fist value will always be the executable file location,
            # anything after will be considered input arguments to the executable.
            # However, all values are inserted as strings in subprocess.
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
            
            # Calls the executable file.
            subprocess.check_call(suprocess_command)
            self._create_png_files_from_HDF5_files(input_file = serialized_output_file)
            count+=1
            
        self._move_file_by_extention(current_location = output_file_location,
                                     new_location = HDF5_file_location,
                                     ext = ".h5")
    
    
    def _main(self, FLAGS):
    
        self._run_propagation_simulation(FLAGS.rytov_value,
                                    FLAGS.wavelength_m,
                                    FLAGS.inner_scale_length_m,
                                    FLAGS.outer_scale_length_m,
                                    FLAGS.propagation_length_m,
                                    FLAGS.grid_height_width,
                                    FLAGS.grid_spacing_height_width_m,
                                    FLAGS.num_screens,
                                    FLAGS.Power_Spectral_Density_Model,
                                    FLAGS.output_file_location,
                                    FLAGS.output_file_name,
                                    FLAGS.HDF5_file_location)

    
if __name__ == '__main__':

    # Instantiate an arg parser.
    parser = argparse.ArgumentParser()

    # Establish default arguements.

    # These flags are often, but not always, overwritten by the launcher.
    parser.add_argument('--rytov_value', type=float, default = 0.2,
                        help='The log-amplitude variance, describes the strength of scintillations.')

    parser.add_argument('--wavelength_m', type=float, default = 1.064e-6,
                        help='Distance over a waves shape repeats.')

    parser.add_argument('--inner_scale_length_m', type=float, default = 1e-3,
                        help='The average size of the largest eddies.')

    parser.add_argument('--outer_scale_length_m', type=float, default = 5.0,
                        help='The average size of the smallest turbulent eddies.')

    parser.add_argument('--propagation_length_m', type=float,default=10e3,
                        help='Over all distance which a laser will propagate.')

    parser.add_argument('--grid_height_width', type=float,default=512,
                        help='Number of samples.')

    parser.add_argument('--grid_spacing_height_width_m', type=float, default=1e-2,
                        help='Sample spacing.')

    parser.add_argument('--num_screens', type=float, default=21,
                        help='Number of turbulent induced phase screens.')

    parser.add_argument('--Power_Spectral_Density_Model', type=str, default='vk',
                        help='A spectral description of refractive-index fluctuation.')

    parser.add_argument('--output_file_location', type=str,
                        default='C:\\Users\\Diego Lozano\\AFRL_Project\\Summer_2018\\Data\\',
                        help='Directory where output files will be placed.')
    
    parser.add_argument('--output_file_name', type=str,
                        default='July19.h5',
                        help='Name of the output file.')
    
    parser.add_argument('--HDF5_file_location', type=str,
                        default='C:\\Users\\Diego Lozano\\AFRL_Project\\Summer_2018\\HDF5_files',
                        help='Directory where HDF5 files will be placed.')
    
    # Parse known arguements.
    FLAGS, unparsed = parser.parse_known_args()
    Simulation = Propagation_Simulation()
    Simulation._main(FLAGS)