import os
import numpy as np
from scipy.io import savemat

# Folder containing .npy files
# folder = "Analysis/Comparative_Analysis/VRFB"
# folder = "Analysis/Performance_Analysis/Concated_epochs/VRFB"
# folder = "Analysis/KF_Analysis/VRFB"
folder = "Analysis/KF_PERF/VRFB"
#
# Loop through all files
for file in os.listdir(folder):
    if file.endswith(".npy"):
        npy_path = os.path.join(folder, file)

        # Load numpy array
        data = np.load(npy_path, allow_pickle=True)

        # Output .mat path
        mat_name = os.path.splitext(file)[0] + ".mat"
        mat_path = os.path.join(folder, mat_name)

        # Variable name inside MATLAB
        var_name = os.path.splitext(file)[0]

        # Save as .mat
        savemat(mat_path, {var_name: data})

        print(f"Converted: {file}  -->  {mat_name}")

print("\nAll .npy files have been converted to .mat successfully.")