# ValidationTools : Configuration
#   
# Developers:
#   Victor E. Bazterra
#   Kenneth James Smith
#
# Descrition:
#   Full configuration for ValidationTools

import os

variables = {}

# initial directory
variables['HomeDirectory'] =  "/uscms_data/d1/kjsmith/CMSSW_2_1_0_pre6/src/Validation/Generator/"

# Directory where references are stored
variables['ReleaseDirectory'] = "/uscms_data/d1/kjsmith/CMSSW_2_1_0_pre6/src/Validation/Generator/DropBox/releases/"

# web directory
#variables['WebDirectory'] = "/afs/fnal.gov/files/home/room2/kjsmith/public_html/"
variables['WebDirectory'] = "/uscms_data/d1/kjsmith/CMSSW_2_1_0_pre6/src/Validation/Generator/DropBox/web/"

variables['HTTPLocation'] = "http://home.fnal.gov/~kjsmith/"

# Set port number
variables["PortNumber"] = 44030

# set root directory
variables['VTRoot'] = variables['HomeDirectory']+"root/"

