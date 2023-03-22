#!/bin/bash

# Purge the installed packages
sudo apt-get purge -y python3 python3-pip libhdf5-dev libhdf5-serial-dev libatlas-base-dev libjasper-dev libqtgui4 libqt4-test motion

# Remove any remaining configuration files
sudo rm -rf /etc/motion

# Remove any remaining Python libraries
sudo pip3 uninstall -y opencv-python-headless numpy pandas scipy pillow

# Update the package list and upgrade the installed packages
sudo apt-get update && sudo apt-get upgrade -y

# Install Python and pip
sudo apt-get install python3 python3-pip -y

# Install OpenCV dependencies
sudo apt-get install libhdf5-dev libhdf5-serial-dev libatlas-base-dev libjasper-dev libqtgui4 libqt4-test -y

# Install OpenCV
sudo pip3 install opencv-python-headless

# Install numpy, pandas, and scipy
sudo pip3 install numpy pandas scipy

# Install motion
sudo apt-get install motion -y

# Install PIL
sudo pip3 install pillow

echo "Purge and reinstallation complete!"
