#! /usr/bin/env python3

import os
import shutil
import sys
import fileinput
import string

##############################################################
# Input parameters
#
#
# Should we execute the code?
#
Submit=True

#
# Should we publish the results?
#
Publish=True
Publish_rootfile=False

# + Location of the AFS place where to put the PDFs
WebRepository = '/eos/user/c/cmsmupog/www/Validation/'

# User enabled to write in the official repository
User='cprieels'

#
# Information about the new release
#
NewParams = dict(
    # Type of parameters
    Type='New',
    
    # Releases to compare
    Release='CMSSW_10_6_0',

    # Conditions of the sample
    Condition='106X_upgrade2021_realistic_v4_rsb',

    # 'no' if no pileup, otherwise set BX spacing
    PileUp='no',
    #PileUp='25ns',
    #PileUp='',      # for HeavyIons

    # 13 or 14TeV?
    Energy = '14TeV',

    Version='v1',

    Format='DQMIO',

    # If True use Fastsim, else use Fullsim
    FastSim=False,
    #FastSim=True,

    # for HeavyIons samples (few folders are not there) 
    HeavyIons=False,
    #HeavyIons=True,

    # needed if you copy any root file from the DQM GUI.
    # See GetLabel function for more details
    Label='',

    # Where to get the root file from. Possible values
    # * WEB: Take root files from the MuonPOG Validation repo on the web
    # * GUI: Copy root files from the DQM GUI server
    # * EOS: copy root files from Muon POG users area
    # By default, if the root files are already in the local area,
    # they won't be overwritten
    GetFilesFrom='GUI',
    #GetFilesFrom='EOS',

    # Base URL of the DQM GUI repository
    #DqmGuiBaseRepo='https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/',
    DqmGuiBaseRepo='https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/',

    # Base repository on EOS
    EOSBaseRepository='/eos/cms/store/group/phys_muon/abbiendi/RelVal/'
)


#
# Information about the reference release.
# We only need to set the variables than are different from the new release
#
RefParams = dict(
    # Type of parameters
    Type='Ref',

    # Parameters for the reference
    Release='CMSSW_10_6_0',
    Condition='106X_upgrade2021_realistic_v4_rsb',
    Energy = '13',
    Version='v1',
    Label=''
)

#
# Optional plots to be made:
#  for FastSim HLT,DQM are set automatically to False
#  for HeavyIons HLT,RECO are skipped if HeavyIons = True
#
ValidateHLT  = True
ValidateRECO = True
ValidateISO  = True
ValidateDQM  = True

# Samples for Validation

# For FullSim No PU
#samples = ['RelValSingleMuPt10','RelValSingleMuPt100','RelValSingleMuPt1000',
#           'RelValZMM', 'RelValWM', 'RelValJpsiMuMu_Pt-8', 'RelValTTbar',
#           'RelValZpMM', 'RelValWpM',
#           'RelValDisplacedSUSY_stopToBottom_M_300_1000mm']

# For FastSim/FullSim PU 25ns
samples = ['RelValTTbar']
#samples = ['RelValZMM', 'RelValTTbar']

# For HeavyIons FullSim
#samples = ['RelValZEEMM_13_HI']
