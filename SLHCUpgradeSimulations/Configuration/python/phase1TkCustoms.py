import FWCore.ParameterSet.Config as cms

def customise(process):
    raise Exception("""
########################################
#
# -- Warning! You are using a deprecated customisation function. --
#
# You have to update your configuration file by
#   If using cmsDriver:
#       1) add the option "--era Run2_2017" 
#   If using a pre-made configuration file:
#       1) add "from Configuration.Eras.Era_Run2_2017_cff import Run2_2017" to the TOP of the config file (above
#          the process declaration).
#       2) add "Run2_2017" as a parameter to the process object, e.g. "process = cms.Process('HLT',Run2_2017)" 
#
# If you are targeting something else than 2017 and have to use a
# customize function from SLHC-times, you have to modify the customize
# function to take phase1Pixel customizations via era, and the rest via
# the customize function. If you are working on something that should
# get integrated to the releases, it is highly recommended to migrate
# the customize function to eras.
#
# There is more information at https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideCmsDriverEras
#
########################################
""")
