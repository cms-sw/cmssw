# This test config file that needs to be modified to have two files being compared
#

import FWCore.ParameterSet.Config as cms

###########################
# Load Files to be compared
###########################

loadTau = cms.EDAnalyzer("DQMFileLoader",
  test = cms.PSet(
    inputFileNames = cms.vstring('/afs/cern.ch/user/v/vasquez/scratch0/CMSSW_2_2_3/src/Validation/RecoTau/test/CMSSW_2_2_3_tauGenJets.root'),
    scaleFactor = cms.double(1.),
    dqmDirectory_store = cms.string('test')
  ),
  reference = cms.PSet(
    inputFileNames = cms.vstring('/afs/cern.ch/user/v/vasquez/scratch0/CMSSW_2_2_0/src/Validation/RecoTau/test/CMSSW_2_2_0_tauGenJets.root'),
    scaleFactor = cms.double(1.),
    dqmDirectory_store = cms.string('reference')
  )
)

#########################################################
# Give the correct test and reference labels on the plots
#########################################################

test = cms.string('CMSSW_2_2_3_Summer08')
reference = cms.string('CMSSW_2_2_0')
