import subprocess
subprocess.call(['sudo', 'apt-get', 'update', '-y'])
subprocess.call(['sudo', 'apt-get', 'upgrade', '-y'])
# Install NumPy
subprocess.call(['sudo', 'apt-get', 'install', '-y', 'python-numpy'])
# Install Pandas
subprocess.call(['sudo', 'apt-get', 'install', '-y', 'python-pandas'])
# Install SciPy
subprocess.call(['sudo', 'apt-get', 'install', '-y', 'python-scipy'])
# Install OpenCV
subprocess.call(['sudo', 'apt-get', 'install', '-y', 'python-opencv'])
# Datasets from sklearn
subprocess.call(['sudo', 'apt-get', 'install', '-y', 'sklearn.datasets'])
# Install Visual Studio Code
subprocess.call(['sudo', 'apt-get', 'install', '-y', 'code'])
