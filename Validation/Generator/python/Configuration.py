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
variables['HomeDirectory'] =  "/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/CMSSW_2_1_0_pre6/src/MC/GeneratorValidation/"

# web directory
variables['WebDirectory'] = "/afs/fnal.gov/files/home/room2/kjsmith/public_html/"
variables['HTTPLocation'] = "http://home.fnal.gov/~kjsmith/"

# Set port number
variables["PortNumber"] = 44030

# set root directory
variables['VTRoot'] = variables['HomeDirectory']+"root/"

