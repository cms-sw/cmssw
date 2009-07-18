# This test config file that needs to be modified to have two files being compared
#

import FWCore.ParameterSet.Config as cms

###########################
# Load Files to be compared
###########################

loadTau = cms.EDAnalyzer("DQMFileLoader",
  test = cms.PSet(
    inputFileNames = cms.vstring('/afs/cern.ch/user/v/vasquez/scratch0/removavableTest/CMSSW_3_1_0_pre2/src/Validation/RecoTau/test/CMSSW_3_1_0_pre2_RelValQCD_FlatPt_15_3000.root'),
    scaleFactor = cms.double(1.),
    dqmDirectory_store = cms.string('test')
  ),
  reference = cms.PSet(
    inputFileNames = cms.vstring('/afs/cern.ch/user/v/vasquez/scratch0/CMSSW_3_1_0_pre1/src/Validation/RecoTau/test/CMSSW_3_1_0_pre1_RelValQCD_FlatPt_15_3000.root'),
    scaleFactor = cms.double(1.),
    dqmDirectory_store = cms.string('reference')
  )
)

#########################################################
# Give the correct test and reference labels on the plots
#########################################################

test = cms.string('CMSSW_3_1_0_pre2 RelValQCD_FlatPt_15_3000')
reference = cms.string('CMSSW_3_1_0_pre1 RelValQCD_FlatPt')
