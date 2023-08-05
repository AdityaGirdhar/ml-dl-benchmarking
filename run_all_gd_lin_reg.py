import sys
import subprocess
from datetime import datetime

# Specify the file name or path of the Python file to run
file_pytorch = "pytorch/gd_lin_reg.py"
#file_mxnet = "mxnet/gd_lin_reg_cpu.py"
file_keras = "keras/lin_regression.py"
file_numpy = "numpy/gd_lin_reg.py"
#file_tensorflow = "tensorflow/linear-reg-batch.py"


lst_files = [file_pytorch, file_keras, file_numpy]


# Run the Python file and capture the output
result = subprocess.run(["python", file_pytorch], capture_output=True, text=True)

# # Store the output in a variable
output = result.stdout

# # Print the output
print(output)


