import FWCore.ParameterSet.Config as cms

hcalsimhitsClient = cms.EDAnalyzer("HcalSimHitsClient", 
     DQMDirName = cms.string("/"), # root directory
     Verbosity  = cms.untracked.bool(False),
)
