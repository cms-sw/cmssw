#! /usr/bin/env python

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
WebRepository = '/eos/cms/store/group/phys_muon/Validation/'

# User enabled to write in the afs area
User='giovanni'

#
# Information about the new release
#
NewParams = dict(
    # Type of parameters
    Type='New',
    
    # Releases to compare
    Release='CMSSW_10_1_0_pre1',

    # Conditions of the sample
    #
    #    FullSim in CMSSW_10_1_0_pre1 
    Condition='100X_upgrade2018_realistic_v10',
    #
    #    FastSim in CMSSW_10_1_0_pre1
    #Condition='100X_mcRun2_asymptotic_v2',

    # 'no' if no pileup, otherwise set BX spacing
    PileUp='25ns',
    #PileUp='',      # for HeavyIons
    #PileUp='no',

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

    Release='CMSSW_10_0_0',

    # Conditions for Special RelVals in CMSSW_10_0_0
    #
    #    FullSim NoPU in CMSSW_10_0_0
    #Condition='100X_upgrade2018_realistic_v6_muVal', 
    #Condition='100X_upgrade2018_realistic_v6_mahiON', #standard RelVals (v1 and v2)
    #
    #    FullSim PU25ns in CMSSW_10_0_0
    Condition='100X_upgrade2018_realistic_v6_muVal_resubwith4cores',
    #
    #    FastSim in CMSSW_10_0_0
    #Condition='100X_mcRun2_asymptotic_v2_muVal',
    #Condition='100X_mcRun2_asymptotic_v2', #standard RelVals (v1)

    Version='v1',
    #Version='v2',

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
#           'RelValZMM_13', 'RelValWM_13', 'RelValJpsiMuMu_Pt-8', 'RelValTTbar_13',
#           'RelValZpMM_13', 'RelValWpM_13',
#           'RelValDisplacedSUSY_stopToBottom_M_300_1000mm_13']

# For FullSim PU 25ns
samples = ['RelValZMM_13', 'RelValTTbar_13']

# For HeavyIons FullSim
#samples = ['RelValZEEMM_13_HI']

# For FastSim No PU
#samples = ['RelValSingleMuPt10_UP15', 'RelValSingleMuPt100_UP15',
#           'RelValZMM_13','RelValTTbar_13']

# For FastSim PU 25ns
#samples = ['RelValZMM_13','RelValTTbar_13']

