import subprocess
# update and upgrade os
# Enable the universe repository
subprocess.call(['sudo', 'add-apt-repository', 'universe'])

# Update the package list
subprocess.call(['sudo', 'apt-get', 'update'])

# Install pip
subprocess.call(['sudo', 'apt-get', 'install', '-y', 'python-pip'])
subprocess.call(['sudo', 'apt-get', 'update', '-y'])
subprocess.call(['sudo', 'apt', 'install', '-y', 'python-pip'])
subprocess.call(['sudo', 'apt-get', 'upgrade', '-y'])

# Install the packages
subprocess.call(['sudo', 'apt-get', 'install', '-y', 'make', 'libssl-dev', 'libghc-zlib-dev', 'libcurl4-gnutls-dev', 'libexpat1-dev', 'gettext'])
# AI packages
subprocess.call(['sudo', 'pip', 'install', '-y', 'numpy', 'pandas', 'scipy'])
# Install OpenCV
subprocess.call(['sudo', 'pip', 'install', '-y', 'python-opencv'])
# Install scikit-learn
subprocess.call(['sudo', 'pip', 'install', 'scikit-learn'])
# Install Visual Studio Code
subprocess.call(['sudo', 'apt-get', 'install', '-y', 'code'])
