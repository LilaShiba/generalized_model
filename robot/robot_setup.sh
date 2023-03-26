#!/bin/bash

sudo apt-get-update
sudo apt full-upgrade
# Base Packages
sudo apt-get install python3-picamera libatlas-base-dev eog git vlc code realvnc-vnc-server realvnc-vnc-viewer -y
# Install OpenCV dependencies
sudo apt-get install python3-dev python3-pip python3-numpy libjpeg-dev libtiff5-dev libjasper-dev libpng-dev libwebp-dev libopenexr-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libfontconfig1-dev libcairo2-dev libgdk-pixbuf2.0-dev libpango1.0-dev libgtk2.0-dev libgtk-3-dev libgtkglext1-dev libatlas-base-dev gfortran libhdf5-dev libhdf5-serial-dev libhdf5-103
# Python Basics
sudo pip install -y numpy pandas scipy pillow matplotlib
# Install CV2
sudo pip install -y opencv-python-headless
echo "Purge and Reinstallation complete!"
