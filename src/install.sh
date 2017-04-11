#!/bin/bash
# Install enviroment
# Author: Victor Segura Tirado
# mail: victor.seguratir@e-campus.uab.cat

# Delete files form last installation
rm lib*
rm CMakeCache.txt
rm -r CMakeFiles
rm cmake_install.cmake

# Make again
cmake -DCMAKE_INSTALL_PREFIX="." ..
