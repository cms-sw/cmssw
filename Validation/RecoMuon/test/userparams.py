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

# + Location of the AFS place where to put the PDFs
WebRepository = '/afs/cern.ch/cms/Physics/muon/CMSSW/Performance/RecoMuon/Validation/val'

# User enabled to write in the afs area
User='ndefilip'

#
# Information about the new release
#
NewParams = dict(
    # Type of parameters
    Type='New',
    
    # Releases to compare
    Release='CMSSW_7_3_0_pre1',

    # Conditions of the sample: MC, STARTUP, POSTLS1...
    # Condition='POSTLS1',
    Condition='PRE_LS1',

    # 'no' if no pileup, otherwise set spacing
    PileUp='50ns',
    #PileUp='no',

    # If True use Fastsim, else use Fullsim
    FastSim=False,

    # Where to get the root file from. Possible values
    # * WEB: Take root files from the MuonPOG Validation repo on the web
    # * CASTOR: Copy root files from castor
    # * GUI: Copy root files from the DQM GUI server
    # By default, if the root files are already in the local area,
    # they won't be overwritten

    GetFilesFrom='GUI',

    # Base URL of the DQM GUI repository
    DqmGuiBaseRepo='https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/',
    #     DqmGuiRepository='https://cmsweb.cern.ch/dqm/dev/data/browse/Development/RelVal/',
    #     DqmGuiRepository='https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/RelVal/CMSSW_4_3_x/',

    # Location of Castor repository
    CastorRepository = '/castor/cern.ch/user/a/aperrott/ValidationRecoMuon',

    # These are only needed if you copy any root file from the DQM GUI.
    # See GetLabel function for more details
    #Label='70_V6_corrHARV',
    Label='72_V16',

    # ???
    Format='DQMIO',
    #Format='GEN-SIM-RECO',

    # Minor Version
    Version='v1'
)


#
# Information about the reference release.
# We only need to set the variables than are different from the new release
#
RefParams = dict(
    Condition='PRE_LS1',
    Type='Ref',
    Release='CMSSW_7_2_0',
    #Label='70_V6_AlcaCSA14',
    Label='72_V16',
    Version='v2'
)

#
# ???
#
ValidateHLT  = True
ValidateRECO = True
ValidateISO  = True
ValidateDQM  = True

# Samples for Validation

# For No PU
# samples= ['RelValSingleMuPt1','RelValSingleMuPt10','RelValSingleMuPt100','RelValSingleMuPt1000','RelValTTbar','RelValZMM','RelValJpsiMM','RelValZpMM_2250_13TeV_Tauola']
# samples= ['RelValZpMM_2250_13TeV_Tauola']

# For PU 25 ns and 50 ns
samples= ['RelValTTbar','RelValZMM','RelValZmumuJets_Pt_20_300']

# For No PU FastSim
# samples=['RelValSingleMuPt10','RelValSingleMuPt100','RelValTTbar']

#############################################################

