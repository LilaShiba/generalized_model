import subprocess
# update and upgrade os
# Enable the universe repository
subprocess.call(['sudo', 'add-apt-repository', 'universe'])
subprocess.call(['sudo', 'apt-get', '-y', 'rpi-update'])
# Update the package list
subprocess.call(['sudo', 'apt-get', 'update'])
# Install pip and packages needed for cv2
subprocess.call(['sudo', 'apt-get', 'install', '-y', 'python3-pip'])
subprocess.call(['sudo', 'apt-get', 'install', '-y', 'build-essential', 'cmake', 'pkg-config', 'libjpeg-dev', 'libtiff5-dev', 'libpng-dev', 'libavcodec-dev', 'libavformat-dev', 'libswscale-dev', 'libv4l-dev', 'libxvidcore-dev', 'libx264-dev', 'libgtk-3-dev', 'libatlas-base-dev', 'gfortran', 'python3-dev'])
subprocess.call(['sudo', 'apt-get', 'upgrade', '-y'])
subprocess.call(['sudo', 'apt-get', 'install', '-y', 'make', 'libssl-dev', 'libghc-zlib-dev', 'libcurl4-gnutls-dev', 'libexpat1-dev', 'gettext'])
# AI packages
subprocess.call(['sudo', 'pip', 'install', 'numpy', 'pandas', 'scipy'])
# Install OpenCV with apt-get & headless
subprocess.call(['sudo', 'apt-get', 'install', '-y', ' python3-opencv'])
# Install scikit-learn
subprocess.call(['sudo', 'pip', 'install', 'scikit-learn'])
# Install Visual Studio Code
#subprocess.call(['sudo', 'apt-get', 'install', '-y', 'code'])
subprocess.call(['sudo','reboot'])
