import subprocess
import numpy as np
import h5py


Rytov = float(input("Rytov? "))
wvl = float(input("Wavelegth? "))
Lout = float(input("Outer Scale Length? "))
Dz = float(input("Propagation Length? "))
N = float(input("Number of Samples ? "))
deltan = float(input("Grid Spacing? "))
nscreen = float(input("Number of screens? "))+1
ntrial = float(input("Number of Trials? "))

loc = [r"C:\Program Files\Test\application\Test.exe","0.1","0.532e-6","5.0","30.0e3","1024.0","5e-3","21.0","5.0"]
#loc = [r"C:\Program Files\Test\application\Test.exe","Rytov","wvl","Lout","Dz","N","deltan","nscreen","ntrial"]
# [r"C:\Program Files\Data_Gen\application\Data_Gen.exe","Rytov","wvl","Lout","Dz","N","deltan","nscreen","ntrial"]
subprocess.check_call(loc, stdin=None, stdout=None, stderr=None, shell=False)

# When going from Matlab HDF5 to Python the matrices seemed to be transposed 
f = h5py.File('datafile.h5', 'r')

im_max = np.transpose(list(f[list(f.keys())[0]])) # Location of the brightest images 
unw_phz = np.transpose(list(f[list(f.keys())[1]])) # Unwrapped Images (Continuous)
w_phz = np.transpose(list(f[list(f.keys())[2]])) # Wraped Images (Discontinuous -pi<phz<pi)

im_max = np.split(im_max,ntrial)
unw_phz = np.split(unw_phz,ntrial)
w_phz = np.split(w_phz,ntrial)