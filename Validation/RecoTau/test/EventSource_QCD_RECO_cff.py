import FWCore.ParameterSet.Config as cms

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
    'rfio:/castor/cern.ch/user/f/friis/CMSSW_4_1_x/skims/QCD/QCD_PU/QCD_PU_10_1_Z95.root',
#    'rfio:/castor/cern.ch/user/f/friis/CMSSW_4_1_x/skims/QCD/QCD_PU/QCD_PU_1_1_NzT.root',
    'rfio:/castor/cern.ch/user/f/friis/CMSSW_4_1_x/skims/QCD/QCD_PU/QCD_PU_2_1_uK6.root',
    'rfio:/castor/cern.ch/user/f/friis/CMSSW_4_1_x/skims/QCD/QCD_PU/QCD_PU_3_1_aGq.root',
    'rfio:/castor/cern.ch/user/f/friis/CMSSW_4_1_x/skims/QCD/QCD_PU/QCD_PU_4_1_gCd.root',
    'rfio:/castor/cern.ch/user/f/friis/CMSSW_4_1_x/skims/QCD/QCD_PU/QCD_PU_5_1_8Lv.root',
    'rfio:/castor/cern.ch/user/f/friis/CMSSW_4_1_x/skims/QCD/QCD_PU/QCD_PU_6_1_416.root',
    'rfio:/castor/cern.ch/user/f/friis/CMSSW_4_1_x/skims/QCD/QCD_PU/QCD_PU_7_1_3ie.root',
    'rfio:/castor/cern.ch/user/f/friis/CMSSW_4_1_x/skims/QCD/QCD_PU/QCD_PU_8_1_tnV.root',
    'rfio:/castor/cern.ch/user/f/friis/CMSSW_4_1_x/skims/QCD/QCD_PU/QCD_PU_9_1_x0t.root'
] )



secFiles.extend( [
   ] )

