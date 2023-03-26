#!/bin/bash

sudo apt-get-update
sudo apt full-upgrade
sudo apt-get install python3-picamera eog git vlc -y
# Install OpenCV dependencies
sudo apt-get install libhdf5-dev libhdf5-serial-dev libatlas-base-dev libjasper-dev libqtgui4 libqt4-test -y
# Remove any remaining Python libraries
sudo pip install -y opencv-python-headless numpy pandas scipy pillow
# Install PIL
sudo pip install pillow

echo "Purge and Reinstallation complete!"
